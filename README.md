# Greenhouse Tomato Multi-Objective Optimization — Code Preview

Preview release of the codebase accompanying a two-year, four-season greenhouse tomato
study. The full pipeline trains **machine-learning surrogate models** on measured
treatment responses and runs **multi-objective evolutionary optimization** (NSGA-III,
NSGA-II, MOEA/D, SMS-EMOA) over three controllable environmental drivers —
**light intensity (PPFD), temperature (°C), irrigation (ETc)** — against four composite
objectives: **Growth, Physiology, Yield, Quality**. Model decisions are explained with
**exact SHAP attributions in the original physical space**, guarded by a
gold-standard reproduction check against the optimizer's runtime behavior.

> **This is a partial preview release.** The surrogate-modeling core, the optimization
> engine, and the data pipeline are **not included** in this public repository; they are
> available from the corresponding author upon reasonable request.

## Repository structure

```
.
├── README.md
├── LICENSE
├── requirements.txt
├── docs/
│   └── architecture.md          # System architecture, data flow, verification design
└── src/
    ├── utils.py                 # Shared utilities: conda DLL self-check, 600-dpi TIFF
    │                            # savefig redirection, dual-channel logging
    ├── sci_visualization.py     # SCI-journal plotting toolkit (2×2 composites, 3-D
    │                            # Pareto fronts, radar, parallel coordinates, heatmaps)
    └── shap_analysis.py         # Exact-SHAP explainability module with gold-standard
                                 # chain verification (standalone runnable)
```

## Method highlights

- **Surrogate-assisted optimization** — tree ensembles (CatBoost / XGBoost /
  Gradient Boosting) approximate the greenhouse response surface so that evolutionary
  algorithms can evaluate thousands of candidate climates without new experiments.
- **Leak-free validation** — repeated-measures structure (18 treatments × 3 replicates)
  is respected via repeated-level leave-one-replicate CV; treatment-mean memory baselines
  and treatment-extrapolation scores are reported as controls against inflated R².
- **Gold-standard chain verification for SHAP** — the preprocessing chain consumed by
  SHAP must reproduce the optimizer's saved Pareto-front predictions bit-exactly
  (`max|Δ| = 0.0` over all front points) before any attribution is produced; the
  augmentation regime is adjudicated from the runtime log, not from configuration intent.
- **Exact Shapley in physical units** — with three environmental drivers, all coalitions
  are enumerated (`algorithm='exact'`), so attributions are exact, interpretable in
  PPFD / °C / ETc, and additive to machine precision (~1e-15).

## Preview modules

### `src/shap_analysis.py` (standalone runnable)

```bash
python src/shap_analysis.py                        # all seasons, ensemble + best model
python src/shap_analysis.py --seasons spring2026   # single season
python src/shap_analysis.py --targets ensemble     # ensemble (optimizer chain) only
```

Outputs per season: SHAP value matrices (`shap_values_{ensemble,best}.csv`, 54×23) and
publication-grade composite figures (2×2 beeswarm / importance / dependence, 600-dpi
TIFF). A persistent run log (`shap_analysis.log`) and an auto-generated analysis brief
(`shap_analysis_report.txt`) are written to the output root.

Set the output root with `SHAP_OUTPUT_ROOT` (defaults to `./output`); place the main
pipeline run log at `<output-root>/tomato_optimization.log` — it provides the
gold-standard anchors (final retrain sample count and `final_train_r2`) used to verify
the reconstructed preprocessing chain.

> The preview module imports the private core (`data_loader`, `data_preprocessor`,
> `models`) at runtime; to execute it end-to-end you need the full codebase (see
> *Requesting the full codebase*).

### `src/sci_visualization.py` (importable toolkit)

Journal-style figure factory used across the study: Times-serif theming, tuned
Pareto-front composites, decision-space analyses, trade-off matrices, and an
interactive Plotly dashboard.

## Requirements

See `requirements.txt`. Python ≥ 3.10 is recommended. Note that `pymoo==0.6.1.5` is
pinned for algorithm-behavior reproducibility, and `shap>=0.45` is required by the
explainability module.

## Requesting the full codebase

Core modules (`main.py`, `models.py`, `optimizer.py`, `config.py`, `data_loader.py`,
`data_preprocessor.py`) and the two-year experimental dataset are available for
research collaboration from the corresponding author upon reasonable request.

## Contact

- **Longhui Niu** — Northwest A&F University, China
- Email: nlonghui@163.com
- ORCID: [0000-0003-2767-1456](https://orcid.org/0000-0003-2767-1456)

## License

Released under the [MIT License](LICENSE).
