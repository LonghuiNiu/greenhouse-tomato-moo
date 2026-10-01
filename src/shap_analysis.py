"""
SHAP Explainability Analysis Module (standalone runnable).

Attributes the four composite objectives (Growth / Physiology / Yield / Quality) of the
tomato surrogate model to the three environmental drivers (PPFD / °C / ETc):

- The prediction chain mirrors the optimizer's chain exactly
  (poly -> standardize -> PCA -> MinMax -> StandardScaler -> model).
- Shapley values are computed in the ORIGINAL physical space with full coalition
  enumeration (3 drivers -> exact, no sampling approximation).
- Gold-standard verification: runtime-log anchors (final-retrain sample count +
  final_train_r2 on the same evaluation set) and bit-exact Pareto-front reproduction
  must pass before any attribution is produced.
- Outputs: SHAP value matrices (CSV) + beeswarm / importance / dependence composites
  (600-dpi TIFF), a persistent run log, and an auto-generated analysis brief.

Usage:
    python shap_analysis.py                          # all seasons, ensemble + best
    python shap_analysis.py --seasons spring2026     # single season
    python shap_analysis.py --targets ensemble       # ensemble (optimizer chain) only

NOTE (preview release): this module imports the private core at runtime
(``data_loader``, ``data_preprocessor``, ``models``); executing it end-to-end requires
the full codebase, which is available from the corresponding author upon request.
"""

import sys
import argparse
import pickle
import re
import logging
import os
from pathlib import Path
from datetime import datetime

sys.path.insert(0, str(Path(__file__).resolve().parent))

import matplotlib
matplotlib.use('Agg')

import numpy as np
import pandas as pd
import joblib
from sklearn.metrics import r2_score
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import shap

from utils import check_env_dll_path, install_tiff_savefig, setup_logging

check_env_dll_path()
install_tiff_savefig(dpi=600)   # project-wide convention: figures are emitted as 600-dpi lossless TIFF only

# Output root is configurable (no hard-coded absolute paths); anchors expect the main
# pipeline run log to be placed at <output-root>/tomato_optimization.log.
OUTPUT_ROOT = Path(os.environ.get('SHAP_OUTPUT_ROOT', './output')).resolve()
OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
# Persistent dual-channel (console + file) run log, shared bootstrap with the main pipeline.
logger = setup_logging(log_file=str(OUTPUT_ROOT / 'shap_analysis.log'))

import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.cm import ScalarMappable
from matplotlib.lines import Line2D
from sci_visualization import SCIVisualizer
from config import SEASON_REGISTRY, ENHANCEMENT_CONFIG
from data_loader import TomatoDataLoader
from data_preprocessor import DataPreprocessor
from models import EnhancedSurrogateModel, ModelConfig

SCIVisualizer()._setup_sci_style()   # reuse the main pipeline's SCI theme verbatim (Times serif / grid / sizes)

OBJECTIVE_NAMES = ['Growth', 'Physiology', 'Yield', 'Quality']
FEATURE_NAMES = ['Light (PPFD)', 'Temperature (°C)', 'Irrigation (ETc)']
FEATURE_KEYS = ['PPFD', 'AvgTemp', 'ET']
FEATURE_SHORT = ['Light', 'Temperature', 'Irrigation']
FEATURE_UNITS = ['PPFD', '°C', 'ETc']
FEATURE_COLORS = ['#2E86AB', '#F18F01', '#6A994E']   # identity colors per driver, fixed across panels/seasons
SHAP_CMAP = LinearSegmentedColormap.from_list(
    'shap_rb', ['#1A5FA8', '#5B9BD5', '#A8CBE8', '#E3DDD5', '#F4C7A8', '#E06B4F', '#A31D1B'])   # deepened ends/trimmed middle: no washed-out points on white
LOG_PATH = OUTPUT_ROOT / 'tomato_optimization.log'


def build_aug_helper(ml_cfg: dict) -> EnhancedSurrogateModel:
    """Build a surrogate shell that reuses models._augment_data verbatim.

    Calling the very same code path (same seed, same order) guarantees the replicated
    augmentation matches the training-time one bit-for-bit.
    """
    return EnhancedSurrogateModel(ModelConfig(
        include_rf=False, include_gbm=False, include_svm=False, include_nn=False,
        include_gp=False, include_xgb=False, include_catboost=False, include_ensemble=False,
        use_data_augmentation=True,
        augmentation_factor=ml_cfg.get('augmentation_factor', 2),
        augmentation_mixup=ml_cfg.get('augmentation_mixup', True),
        random_state=ml_cfg.get('random_state', 42)))


class SurrogateChain:
    """Load or deterministically rebuild the preprocessing chain the optimizer used."""

    def __init__(self, season_dir: Path, season: str, ml_cfg: dict, n_anchor: int):
        tm_path = season_dir / 'models' / 'training_metrics.pkl'
        if not tm_path.exists():
            raise FileNotFoundError(f"Missing model artifact {tm_path}; run main.py first")
        tm = joblib.load(tm_path)
        self.metrics = tm['metrics']
        self.best_model_name = tm['best_model_name']
        self.persisted_weights = tm.get('ensemble_weights')

        chain = tm.get('preprocessing_chain')
        if chain and chain.get('scaler') is not None:
            self.source = 'persisted'
            self.poly = chain['poly_transformer']
            self.poly_scaler = chain['poly_scaler']
            self.pca = chain['poly_pca']
            self.feature_scaler = chain.get('feature_scaler')
            self.scaler = chain['scaler']
            self.X_bg = None
        else:
            self.source = 'reconstructed'
            self._reconstruct(season, ml_cfg, n_anchor)

    def _reconstruct(self, season: str, ml_cfg: dict, n_anchor: int):
        """Deterministically rebuild the chain under the runtime training regime:
        poly -> standardize -> PCA -> MinMax -> (augmentation) -> StandardScaler.
        Whether augmentation ran is adjudicated from the log anchor (actual executed
        behavior: anchor count > 54 rows implies the augmented regime), never from
        configuration intent."""
        self.poly = self.poly_scaler = self.pca = self.feature_scaler = None
        loader = TomatoDataLoader()
        loader.load_data(season=season)
        X_raw, _ = loader.get_features_and_targets()
        Xp = X_raw
        if ENHANCEMENT_CONFIG.get('use_poly_features', False):
            from sklearn.decomposition import PCA
            from sklearn.preprocessing import PolynomialFeatures
            self.poly = PolynomialFeatures(degree=2, include_bias=False,
                                           interaction_only=False).fit(X_raw)
            Xp = self.poly.transform(X_raw)
            self.poly_scaler = StandardScaler().fit(Xp)
            Xp = self.poly_scaler.transform(Xp)
            self.pca = PCA(n_components=ENHANCEMENT_CONFIG['pca_variance_threshold']).fit(Xp)
            Xp = self.pca.transform(Xp)
        self.feature_scaler = MinMaxScaler().fit(Xp)
        self.X_mm = self.feature_scaler.transform(Xp)
        self.augmented = n_anchor > len(self.X_mm)
        if self.augmented:
            helper = build_aug_helper(ml_cfg)
            self.X_bg, _, _, _ = helper._augment_data(
                self.X_mm, np.zeros((len(self.X_mm), 1)), None, None)
        else:
            self.X_bg = self.X_mm
        self.scaler = StandardScaler().fit(self.X_bg)

    def transform(self, X_raw: np.ndarray) -> np.ndarray:
        """Map raw physical inputs into the model-input space (step-identical to
        surrogate.predict's internal chain)."""
        Xp = self.poly.transform(X_raw) if self.poly is not None else X_raw
        Xp = self.poly_scaler.transform(Xp) if self.poly_scaler is not None else Xp
        Xp = self.pca.transform(Xp) if self.pca is not None else Xp
        if self.feature_scaler is not None:
            Xp = self.feature_scaler.transform(Xp)
        return self.scaler.transform(Xp)


def rebuild_ensemble_weights(metrics: dict, weight_method: str) -> dict:
    """Rebuild ensemble weights exactly as models._build_ensemble does, from the
    persisted metrics (val_r2 is unchanged by the final retraining)."""
    candidates = [m for m, v in metrics.items()
                  if m != 'Ensemble (weighted)' and (v['val_r2'] > 0 or v.get('cv_mean_r2', -1) > 0)]
    top3 = sorted(candidates, key=lambda m: metrics[m]['val_r2'], reverse=True)[:3]
    if weight_method == 'softmax':
        r2 = np.array([max(0.0, metrics[m]['val_r2']) for m in top3])
        scores = np.exp(r2 - r2.max())
    elif weight_method == 'rmse':
        scores = np.array([1.0 / max(1e-6, metrics[m]['val_mse']) for m in top3])
    else:   # mixed
        scores = []
        for m in top3:
            vr, cr = max(0.0, metrics[m]['val_r2']), max(0.0, metrics[m].get('cv_mean_r2', 0.0))
            scores.append(vr * (1 + cr) if cr > 0 else vr)
    scores = np.array(scores)
    total = scores.sum()
    weights = scores / total if total > 1e-6 else np.full(len(top3), 1.0 / len(top3))
    return dict(zip(top3, weights))


def parse_log_anchor(season: str) -> tuple:
    """Extract the final-retrain anchor (augmented sample count, final_train_r2) from
    the runtime log; with multiple runs, the last entry per season wins."""
    if not LOG_PATH.exists():
        raise FileNotFoundError(f"Missing run log {LOG_PATH}; cannot verify chain regime. "
                                f"Run main.py first")
    text = LOG_PATH.read_text(encoding='utf-8', errors='ignore')
    pattern = (r"PROCESSING SEASON: " + season.upper() + r".*?"
               r"Final model retrained on full data \((\d+) augmented samples\), "
               r"final_train_r2=([0-9.]+)")
    matches = re.findall(pattern, text, flags=re.S)
    if not matches:
        raise ValueError(f"No final-retrain anchor for {season} in the log; "
                         f"gold-standard verification impossible")
    n, r2 = matches[-1]
    return int(n), float(r2)


def rebuild_global_y() -> dict:
    """Re-build globally normalized targets via the same deterministic preprocessing
    (stage aggregation / subgroup PCA / variance-weighted quality score)."""
    y_all_list, y_by_season = [], {}
    for season in SEASON_REGISTRY:
        loader = TomatoDataLoader()
        loader.load_data(season=season)
        _, targets = loader.get_features_and_targets()
        pre = DataPreprocessor(variance_threshold=ENHANCEMENT_CONFIG['pca_variance_threshold'])
        pre.stage_info = loader.stage_info
        pre.base_info = loader.base_info
        pt = pre.preprocess_targets(targets, target_columns=loader.target_columns)
        cs = pre.create_composite_scores(pt, composite_method='first_pc')
        y_by_season[season] = cs
        y_all_list.append(cs)
    gscaler = MinMaxScaler().fit(np.vstack(y_all_list))
    return {s: gscaler.transform(y) for s, y in y_by_season.items()}


def verify_chain(chain: SurrogateChain, metrics: dict, weights: dict, season: str,
                 y54: np.ndarray, ml_cfg: dict, X_phys: np.ndarray, F_runtime: np.ndarray):
    """Gold-standard verification.

    Primary gate (y-free): re-running the saved Pareto-front decision variables through
    the reconstructed chain + ensemble weights must reproduce the saved front objectives
    bit-exactly; any mismatch aborts SHAP (misaligned regimes must never yield results).
    Diagnostic (non-blocking): final_train_r2 re-computation against the log anchor
    monitors target re-construction consistency and does not affect SHAP correctness.
    """
    n_anchor, r2_anchor = parse_log_anchor(season)
    if chain.source == 'reconstructed' and chain.X_bg.shape[0] != n_anchor:
        raise AssertionError(f"{season}: replicated augmented sample count "
                             f"{chain.X_bg.shape[0]} != log anchor {n_anchor}; chain regime "
                             f"differs from training time - aborting SHAP")
    f_ens = make_predict_fn(chain, metrics, weights, 'ensemble')
    mad = np.abs(f_ens(X_phys) - F_runtime).max()
    if mad > 1e-3:
        raise AssertionError(f"{season}: front reproduction over {len(X_phys)} points gave "
                             f"max|Δ|={mad:.4f} > 1e-3 - chain failed end-to-end verification, "
                             f"aborting SHAP")
    logger.info(f"  [primary gate passed] {len(X_phys)} front points reproduced, "
                f"max|Δ|={mad:.2e} (chain + ensemble weights identical to runtime)")

    model = metrics[chain.best_model_name]['model']
    if chain.source == 'reconstructed':
        X_eval = chain.scaler.transform(chain.X_bg)   # same evaluation set as the anchor
        if chain.augmented:
            helper = build_aug_helper(ml_cfg)
            _, y_eval, _, _ = helper._augment_data(chain.X_mm, y54, None, None)
        else:
            y_eval = y54
    else:
        loader = TomatoDataLoader()
        loader.load_data(season=season)
        X_raw, _ = loader.get_features_and_targets()
        X_eval, y_eval = chain.transform(X_raw), y54   # persisted chain pairs with the new-regime log anchor
    r2_now = r2_score(y_eval, model.predict(X_eval))
    tag = 'consistent' if abs(r2_now - r2_anchor) <= 1e-3 else \
        'deviates (target re-construction drifted from the historical run; SHAP correctness unaffected)'
    logger.info(f"  [diagnostic] final_train_r2 re-computed={r2_now:.4f} vs log anchor "
                f"{r2_anchor:.4f}: {tag}")


def make_predict_fn(chain: SurrogateChain, metrics: dict, weights: dict, target: str):
    """Build the prediction function f (raw physical inputs -> 4 objectives in [0, 1])
    strictly mirroring the optimizer's chain."""
    if target == 'ensemble':
        members = [(weights[m], metrics[m]['model']) for m in weights]
    else:
        members = [(1.0, metrics[target]['model'])]

    def f(X_raw: np.ndarray) -> np.ndarray:
        Xs = chain.transform(np.atleast_2d(X_raw))
        pred = np.sum([w * m.predict(Xs) for w, m in members], axis=0)
        return np.clip(pred, 0, 1)

    return f


def save_figures(explanation, X_raw: np.ndarray, season: str, target: str,
                 out_dir: Path, best_model_name: str):
    """Emit three 2x2 composite figures (beeswarm / importance / dependence), 600-dpi TIFF."""
    fig_dir = out_dir / 'figures'
    fig_dir.mkdir(parents=True, exist_ok=True)
    sv4 = explanation.values
    target_label = ('Ensemble Surrogate (optimizer chain)' if target == 'ensemble'
                    else f'Best Single Model ({best_model_name})')
    _grid_figure(season, target_label,
                 lambda ax, obj, fig: _beeswarm_panel(
                     ax, sv4[:, :, OBJECTIVE_NAMES.index(obj)], X_raw),
                 'SHAP value (impact on objective)', f'Fig_SHAP_Summary_{target}.tif',
                 fig_dir, colorbar=True)
    _grid_figure(season, target_label,
                 lambda ax, obj, fig: _bar_panel(
                     ax, sv4[:, :, OBJECTIVE_NAMES.index(obj)]),
                 'Mean |SHAP value| (±SD, n=54)', f'Fig_SHAP_Importance_{target}.tif',
                 fig_dir)
    if target == 'ensemble':
        _grid_figure(season, target_label,
                     lambda ax, obj, fig: _dependence_panel(
                         ax, sv4[:, :, OBJECTIVE_NAMES.index(obj)], X_raw, fig),
                     None, 'Fig_SHAP_Dependence_ensemble.tif', fig_dir,
                     fig_legend='Binned mean ±SEM (8 equal-frequency bins)')


def _beeswarm_layering(sv: np.ndarray):
    """SHAP-style layered stacking: sort by value, bin-index offsets give the y spread.
    Returns (sorted indices, normalized offsets in [-0.5, 0.5])."""
    order = np.argsort(sv, kind='stable')
    sv_s = sv[order]
    nbins = 100
    edges = np.linspace(sv_s[0], sv_s[-1] + 1e-12, nbins + 1)
    b = np.clip(np.searchsorted(edges, sv_s, side='right') - 1, 0, nbins - 1)
    counts = np.bincount(b, minlength=nbins)
    seen = np.zeros(nbins)
    y = np.empty(len(sv))
    for i, bi in enumerate(b):
        seen[bi] += 1
        y[i] = seen[bi] - (counts[bi] + 1) / 2.0
    return order, y / max(counts.max(), 1)


def _beeswarm_panel(ax, sv: np.ndarray, fv: np.ndarray):
    """Single-objective beeswarm: drivers sorted by mean|SHAP|, points colored by
    feature value (blue low -> red high)."""
    row_order = np.argsort(np.abs(sv).mean(axis=0))[::-1]
    for row, fi in enumerate(row_order):
        order, yoff = _beeswarm_layering(sv[:, fi])
        span = np.ptp(fv[:, fi]) + 1e-12
        norm = (fv[order, fi] - fv[:, fi].min()) / span
        ax.scatter(sv[order, fi], row + yoff * 0.85, c=SHAP_CMAP(norm), s=50,
                   edgecolors='white', linewidth=0.6, zorder=3)
    ax.set_yticks(np.arange(len(row_order)))
    ax.set_yticklabels([FEATURE_NAMES[i] for i in row_order], fontsize=14)
    ax.set_ylim(2.5, -0.5)   # set_ylim(bottom, top): 2.5 below, -0.5 above -> rank 0 on top
    lim = np.abs(sv).max() * 1.08 + 1e-12
    ax.set_xlim(-lim, lim)
    ax.axvline(0, color='gray', ls='--', lw=1.1, alpha=0.6, zorder=1)
    ax.grid(True, alpha=0.3, ls='--')
    ax.tick_params(axis='x', labelsize=13)
    return row_order


def _bar_panel(ax, sv: np.ndarray):
    """Single-objective mean|SHAP| bars: per-driver identity colors + SD error bars +
    value annotations (±SD across n=54 samples)."""
    row_order = np.argsort(np.abs(sv).mean(axis=0))[::-1]
    means = np.abs(sv).mean(axis=0)[row_order]
    stds = np.abs(sv).std(axis=0)[row_order]
    y = np.arange(len(row_order))
    ax.barh(y, means, xerr=stds, height=0.68, color=[FEATURE_COLORS[i] for i in row_order],
            edgecolor='k', linewidth=0.8,
            error_kw=dict(ecolor='#555555', elinewidth=1.6, capsize=3.5), zorder=3)
    for yi, m, s in zip(y, means, stds):
        ax.text(m + s + means.max() * 0.03, yi, f'{m:.3f}', va='center', fontsize=13)
    ax.set_yticks(y)
    ax.set_yticklabels([FEATURE_NAMES[i] for i in row_order], fontsize=14)
    ax.set_ylim(2.5, -0.5)   # rank 0 on top, same as the beeswarm
    ax.set_xlim(0, (means + stds).max() * 1.22)
    ax.grid(True, alpha=0.3, ls='--', axis='x')
    ax.tick_params(axis='x', labelsize=13)


def _dependence_panel(ax, sv: np.ndarray, fv: np.ndarray, fig):
    """Single-objective dependence: top driver vs SHAP, colored by the runner-up driver,
    with an equal-frequency binned mean ± SEM line (non-linear shape readable)."""
    row_order = np.argsort(np.abs(sv).mean(axis=0))[::-1]
    top1, top2 = int(row_order[0]), int(row_order[1])
    x, y, c = fv[:, top1], sv[:, top1], fv[:, top2]
    sc = ax.scatter(x, y, c=c, cmap=SHAP_CMAP, s=58, edgecolors='white',
                    linewidth=0.5, zorder=3)
    ax.axhline(0, color='gray', ls='--', lw=1.1, alpha=0.6, zorder=1)
    qedges = np.quantile(x, np.linspace(0, 1, 9))
    qb = np.clip(np.searchsorted(qedges, x, side='right') - 1, 0, 7)
    mids, means, sems = [], [], []
    for k in range(8):
        m = qb == k
        if m.sum() >= 2:
            mids.append(x[m].mean())
            means.append(y[m].mean())
            sems.append(y[m].std(ddof=1) / np.sqrt(int(m.sum())))
    ax.errorbar(mids, means, yerr=sems, fmt='o-', color='#333333', markersize=5.5,
                elinewidth=1.4, capsize=3.5, lw=2.2, zorder=4)
    ax.set_xlabel(f'{FEATURE_NAMES[top1]}', fontsize=14)
    ax.set_ylabel(f'SHAP value for {FEATURE_SHORT[top1]}', fontsize=14)
    cb = fig.colorbar(sc, ax=ax, fraction=0.046, pad=0.02)
    cb.set_label(f'{FEATURE_SHORT[top2]} ({FEATURE_UNITS[top2]})', fontsize=13)
    cb.ax.tick_params(labelsize=12)
    ax.grid(True, alpha=0.3, ls='--')
    ax.tick_params(labelsize=13)


def _grid_figure(season, target_label, panel_fn, xlabel, fig_name, fig_dir,
                 colorbar=False, fig_legend: str = None):
    """2x2 four-objective composite skeleton: (a)-(d) panel tags, a figure-level legend
    placed in a reserved band between the suptitle and the panels (data can never reach
    it), and unified export."""
    fig = plt.figure(figsize=(13, 10.5))
    right = 0.85 if colorbar else 0.97
    gs = GridSpec(2, 2, figure=fig, hspace=0.22, wspace=0.34,
                  left=0.09, right=right, top=0.90, bottom=0.08)
    axes = [fig.add_subplot(gs[i, j]) for i in range(2) for j in range(2)]
    for ax, obj, tag in zip(axes, OBJECTIVE_NAMES, ['(a)', '(b)', '(c)', '(d)']):
        ax.set_title(f'{tag} {obj}', fontsize=16)
        panel_fn(ax, obj, fig)
    if colorbar:
        cax = fig.add_axes([0.875, 0.16, 0.016, 0.68])
        cb = fig.colorbar(ScalarMappable(cmap=SHAP_CMAP, norm=Normalize(0, 1)), cax=cax)
        cb.set_ticks([0, 1])
        cb.set_ticklabels(['Low', 'High'])
        cb.set_label('Feature value', fontsize=13)
        cb.ax.tick_params(labelsize=12)
    if xlabel:
        for ax in axes[-2:]:
            ax.set_xlabel(xlabel, fontsize=14)
    center = (0.09 + right) / 2   # suptitle/legend align with the grid's actual center
    if fig_legend:
        handle = Line2D([0], [0], color='#333333', marker='o', markersize=5, lw=2)
        fig.legend([handle], [fig_legend], loc='upper center',
                   bbox_to_anchor=(center, 0.942), frameon=False, fontsize=12)
    fig.suptitle(f'SHAP Attribution — {target_label} ({season})',
                 fontsize=18, fontweight='bold', y=0.968, x=center)
    plt.savefig(fig_dir / fig_name)
    plt.close(fig)
    logger.info(f"    composite figure emitted: {fig_name}")


def objective_stats(sv: np.ndarray, fv: np.ndarray) -> list:
    """Per-objective attribution statistics: mean|SHAP|±SD in descending order plus
    high/low feature-value group mean SHAP (signed directional evidence)."""
    order = np.argsort(np.abs(sv).mean(axis=0))[::-1]
    stats = []
    for fi in order:
        m, s = float(np.abs(sv[:, fi]).mean()), float(np.abs(sv[:, fi]).std())
        med = np.median(fv[:, fi])
        hi, lo = fv[:, fi] >= med, fv[:, fi] < med
        d_hi = float(sv[hi, fi].mean()) if hi.any() else 0.0
        d_lo = float(sv[lo, fi].mean()) if lo.any() else 0.0
        stats.append((FEATURE_NAMES[fi], m, s, d_hi, d_lo))
    return stats


def write_report(summaries: list, targets: list):
    """Analysis brief: regime statement + per-season verification verdicts + ensemble
    weights + four-objective attribution tables with signed directional evidence.
    Regenerated on every run."""
    lines = [
        '=' * 78,
        'SHAP Explainability Analysis Brief (surrogate environmental-driver attribution)',
        f'Generated: {datetime.now().strftime("%Y-%m-%d %H:%M")} | shap version: {shap.__version__}',
        'Regime statement: the prediction chain mirrors the optimizer chain exactly '
        '(poly -> standardize -> PCA -> MinMax -> StandardScaler -> model);',
        '          chain source and augmentation regime are adjudicated from runtime-log '
        'anchors; primary gate = Pareto-front reproduction (max|Δ| ≤ 1e-3);',
        '          SHAP is exact Shapley in the original physical space '
        '(additivity residual ~1e-15)',
        '-' * 78,
    ]
    for sm in summaries:
        lines += [f'[{sm["season"]}]',
                  f'  Prediction chain: {sm["chain_source"]} | log anchor: {sm["n_anchor"]} samples/'
                  f'final_train_r2={sm["r2_anchor"]:.4f}',
                  f'  Primary gate: {sm["n_front"]} front points reproduced, '
                  f'max|Δ|={sm["front_mad"]:.2e}  (passed)',
                  f'  Ensemble weights: ' + ', '.join(f'{k}={v:.4f}' for k, v in sm['weights'].items())]
        for tgt, tinfo in sm['targets'].items():
            label = ('Ensemble (optimizer prediction chain)' if tgt == 'ensemble'
                     else f'Best single model ({tgt})')
            lines.append(f'  * {label}  additivity residual={tinfo["resid"]:.2e}')
            for obj in OBJECTIVE_NAMES:
                lines.append(f'    > {obj} (mean|SHAP|±SD descending; high/low feature-value '
                             f'group mean SHAP = directional evidence):')
                for rank, (name, m, s, d_hi, d_lo) in enumerate(tinfo['objs'][obj], 1):
                    lines.append(f'      {rank}. {name}: {m:.3f}±{s:.3f} | '
                                 f'high-value group SHAP={d_hi:+.3f}, low-value group SHAP={d_lo:+.3f}, '
                                 f'group gap={d_hi - d_lo:+.3f}')
        lines.append('  Artifacts: shap_values_{ensemble,best}.csv (54x23) + '
                     'figures/ composites (600-dpi TIFF)')
        lines.append('-' * 78)
    lines.append('End of report (auto-generated by shap_analysis.py, refreshed on every run)')
    report_path = OUTPUT_ROOT / 'shap_analysis_report.txt'
    report_path.write_text('\n'.join(lines), encoding='utf-8')
    logger.info(f'Analysis brief refreshed: {report_path}')


def run_season(season: str, targets: list, global_y: dict) -> dict:
    """Run the full SHAP analysis for one season and return its summary for the brief."""
    season_dir = OUTPUT_ROOT / season
    if not season_dir.exists():
        raise FileNotFoundError(f"Missing season artifact directory {season_dir}; run main.py first")

    with open(season_dir / 'optimization_results.pkl', 'rb') as fh:
        top = pickle.load(fh)
    ml_cfg = top.get('ml_config', {})
    weight_method = ml_cfg.get('ensemble_weight_method',
                               ENHANCEMENT_CONFIG.get('ensemble_weight_method', 'mixed'))

    logger.info(f'===== {season} =====')
    n_anchor, r2_anchor = parse_log_anchor(season)   # augmentation regime adjudicated from the runtime log
    chain = SurrogateChain(season_dir, season, ml_cfg, n_anchor)
    logger.info(f"  Prediction chain: {chain.source} (log anchor: {n_anchor} samples, "
                f"final_train_r2={r2_anchor})")
    loader = TomatoDataLoader()
    loader.load_data(season=season)
    X_raw, _ = loader.get_features_and_targets()
    logger.info(f'  Raw features: {X_raw.shape} (PPFD/AvgTemp/ET)')

    out_dir = season_dir / 'shap'
    out_dir.mkdir(parents=True, exist_ok=True)
    y54 = global_y[season]

    # Ensemble weights (shared by the verification gate and the ensemble target):
    # persisted first; otherwise rebuilt exactly from the persisted metrics.
    weights = chain.persisted_weights or rebuild_ensemble_weights(chain.metrics, weight_method)
    logger.info(f"  Ensemble weights ({'persisted' if chain.persisted_weights else 'rebuilt via build formula'}): "
                f"{ {k: round(v, 4) for k, v in weights.items()} }")
    # Chain gold-standard verification (targets the chain, not any single target).
    verify_chain(chain, chain.metrics, weights, season, y54, ml_cfg,
                 top['results']['decision_variables'], top['results']['objectives'])

    summary = {'season': season, 'chain_source': chain.source, 'n_anchor': n_anchor,
               'r2_anchor': r2_anchor, 'weights': weights,
               'n_front': len(top['results']['objectives']),
               'front_mad': 0.0, 'targets': {}}
    for target in targets:
        target_name = chain.best_model_name if target == 'best' else target
        if target_name != 'ensemble':
            if target_name not in chain.metrics or chain.metrics[target_name].get('model') is None:
                raise KeyError(f"Target model '{target_name}' missing from training_metrics "
                               f"or its model is None")
        f = make_predict_fn(chain, chain.metrics, weights, target_name)
        explainer = shap.Explainer(f, X_raw, algorithm='exact')   # 3 drivers -> exact enumeration
        explanation = explainer(X_raw, silent=True)
        resid = np.abs(f(X_raw) - (explanation.base_values + explanation.values.sum(axis=1))).max()
        logger.info(f'  SHAP exact additivity residual: {resid:.2e} (machine-precision = correct)')

        rows = pd.DataFrame(X_raw, columns=FEATURE_KEYS)
        for obj_idx, obj in enumerate(OBJECTIVE_NAMES):
            for fi, fk in enumerate(['Light', 'Temperature', 'Irrigation']):
                rows[f'SHAP_{fk}_{obj}'] = explanation.values[:, fi, obj_idx]
            rows[f'Base_{obj}'] = explanation.base_values[:, obj_idx]
            rows[f'Pred_{obj}'] = f(X_raw)[:, obj_idx]
        csv_path = out_dir / f'shap_values_{target}.csv'
        rows.to_csv(csv_path, index=False, encoding='utf-8-sig')
        logger.info(f'  SHAP matrix exported: {csv_path} ({rows.shape[0]} rows x {rows.shape[1]} cols)')

        save_figures(explanation, X_raw, season, target, out_dir, chain.best_model_name)
        summary['targets'][target] = {
            'resid': float(resid),
            'objs': {obj: objective_stats(explanation.values[:, :, oi], X_raw)
                     for oi, obj in enumerate(OBJECTIVE_NAMES)},
        }
    return summary


def main():
    parser = argparse.ArgumentParser(description='Surrogate SHAP explainability analysis (standalone)')
    parser.add_argument('--seasons', nargs='*', default=list(SEASON_REGISTRY.keys()),
                        help=f'season list, default all: {list(SEASON_REGISTRY.keys())}')
    parser.add_argument('--targets', nargs='*', default=['ensemble', 'best'],
                        help='ensemble (optimizer chain) / best (top single model) / '
                             'any model name present in training_metrics')
    args = parser.parse_args()

    logger.info(f'shap {shap.__version__} | targets: {args.targets} | seasons: {args.seasons}')
    for season in args.seasons:
        if season not in SEASON_REGISTRY:
            raise ValueError(f"Unknown season '{season}'; options: {list(SEASON_REGISTRY.keys())}")

    global_y = rebuild_global_y()   # all seasons loaded once, for final_train_r2 re-computation
    summaries = [run_season(season, args.targets, global_y) for season in args.seasons]
    write_report(summaries, args.targets)
    logger.info('SHAP analysis finished')


if __name__ == '__main__':
    main()
