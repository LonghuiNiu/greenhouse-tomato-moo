"""
Scientific Visualization Toolkit (SCI journal style).

Publication-grade figure factory shared across the study:

- Enhanced Pareto-front composite (2-D projections + 3-D front + parallel coordinates)
- Trade-off analysis matrix (histograms / scatter with correlation annotations)
- Decision-space analysis (driver impact, temperature regimes, variable importance)
- Interactive Plotly dashboard

Theme: Times-serif type system, whitegrid background, per-driver identity colors, and
a process-wide 600-dpi TIFF export policy (see utils.install_tiff_savefig).
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import rcParams, gridspec
import seaborn as sns
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.patches import FancyArrowPatch
from mpl_toolkits.mplot3d import proj3d
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
try:
    import scienceplots  # optional; the code never calls its API, absence is tolerated
except ImportError:
    pass
from sklearn.ensemble import RandomForestRegressor
import warnings
from typing import Dict
from utils import install_tiff_savefig
warnings.filterwarnings('ignore')

install_tiff_savefig(dpi=600)   # standalone use of this toolkit also emits 600-dpi TIFF only


class Arrow3D(FancyArrowPatch):
    """3-D arrow rendered through the mplot3d projection."""

    def __init__(self, xs, ys, zs, *args, **kwargs):
        super().__init__((0, 0), (0, 0), *args, **kwargs)
        self._verts3d = xs, ys, zs

    def do_3d_projection(self, renderer=None):
        xs3d, ys3d, zs3d = self._verts3d
        xs, ys, zs = proj3d.proj_transform(xs3d, ys3d, zs3d, self.axes.M)
        self.set_positions((xs[0], ys[0]), (xs[1], ys[1]))
        return np.min(zs)


class SCIVisualizer:
    """Scientific visualizer with SCI-journal styling."""

    def __init__(self, results: Dict = None):
        self.results = results
        self._setup_sci_style()

    def _setup_sci_style(self):
        """Apply the journal type system: Times serif, whitegrid, fixed size scale."""
        # Force-disable LaTeX/mathtext exotic rendering for portability
        rcParams['text.usetex'] = False
        rcParams['mathtext.default'] = 'regular'
        rcParams['mathtext.fontset'] = 'stix'

        # Stable built-in styles (seaborn merged into matplotlib)
        try:
            plt.style.use('seaborn-v0_8-whitegrid')
        except Exception:
            try:
                plt.style.use('seaborn-whitegrid')
            except Exception:
                plt.style.use('default')

        # Times New Roman with graceful fallbacks
        rcParams['font.family'] = 'serif'
        rcParams['font.serif'] = ['Times New Roman', 'DejaVu Serif', 'serif']
        rcParams['font.size'] = 11
        rcParams['axes.titlesize'] = 12
        rcParams['axes.labelsize'] = 11
        rcParams['xtick.labelsize'] = 10
        rcParams['ytick.labelsize'] = 10
        rcParams['legend.fontsize'] = 10
        rcParams['figure.titlesize'] = 14
        rcParams['figure.dpi'] = 300
        rcParams['savefig.dpi'] = 300
        rcParams['savefig.format'] = 'png'
        rcParams['savefig.bbox'] = 'tight'
        rcParams['savefig.pad_inches'] = 0.05

        # Color palette for SCI publications
        self.colors = {
            'primary': ['#2E86AB', '#A23B72', '#F18F01', '#C73E1D', '#6A994E'],
            'sequential': ['#f7fbff', '#deebf7', '#c6dbef', '#9ecae1', '#6baed6',
                           '#4292c6', '#2171b5', '#08519c', '#08306b'],
            'diverging': ['#ca0020', '#f4a582', '#f7f7f7', '#92c5de', '#0571b0'],
            'categorical': ['#4C72B0', '#DD8452', '#55A868', '#C44E52', '#8172B3',
                            '#937860', '#DA8BC3', '#8C8C8C', '#CCB974', '#64B5CD']
        }

    def plot_enhanced_pareto_front(self, save_path: str = None):
        """Enhanced Pareto-front composite: two 2-D projections, one 3-D front, one
        parallel-coordinates panel, with per-projection color bars."""
        if self.results is None or 'objectives' not in self.results:
            raise ValueError("No optimization results available")

        objectives = self.results['objectives']
        n_obj = objectives.shape[1]

        # Taller bottom row so the 3-D square projection fills its cell without
        # the horizontal dead band a rectangular cell would force.
        fig = plt.figure(figsize=(12, 10))
        gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.26, wspace=0.34,
                               left=0.0, right=0.82, top=0.92, bottom=0.085,
                               height_ratios=(1, 1.35))

        ax1 = fig.add_subplot(gs[0, 0])
        self._plot_2d_pareto(ax1, objectives, 0, 1, 'Growth', 'Physiology')

        ax2 = fig.add_subplot(gs[0, 1])
        self._plot_2d_pareto(ax2, objectives, 2, 3, 'Yield', 'Quality')

        if n_obj >= 3:
            ax3 = fig.add_subplot(gs[1, 0], projection='3d')
            self._plot_3d_pareto(ax3, objectives)

        ax4 = fig.add_subplot(gs[1, 1])
        self._plot_parallel_coordinates(ax4, objectives)

        # Center the suptitle on the grid span, not the full figure canvas
        plt.suptitle('Multi-Objective Optimization: Pareto Front Analysis',
                     fontsize=14, fontweight='bold', y=0.975, x=0.41)
        if save_path:
            plt.savefig(save_path, dpi=600, bbox_inches='tight')
            print(f"Figure saved to {save_path}")

        plt.close()

    def _plot_2d_pareto(self, ax, objectives, idx1, idx2, label1, label2):
        """2-D projection panel with a color bar for a third objective and the best
        compromise solution highlighted."""
        scatter = ax.scatter(objectives[:, idx1], objectives[:, idx2],
                             c=objectives[:, (idx1 + 1) % objectives.shape[1]],
                             cmap='viridis',
                             s=50, alpha=0.8, edgecolors='k', linewidth=0.5)

        if 'best_solution' in self.results:
            best_obj = self.results['best_solution']['objectives']
            ax.scatter(best_obj[idx1], best_obj[idx2],
                       s=150, marker='*', color='red', edgecolors='k',
                       linewidth=1.5, zorder=5, label='Best Compromise')

        ax.set_xlabel(label1, fontsize=11)
        ax.set_ylabel(label2, fontsize=11)
        ax.set_title(f'{label1} vs {label2}', fontsize=12, pad=20)
        ax.grid(True, alpha=0.3, linestyle='--')
        # Horizontal legend above the panel keeps it off the scatter
        ax.legend(loc='lower center', bbox_to_anchor=(0.5, 1.0),
                  fontsize=9, frameon=False)

        plt.colorbar(scatter, ax=ax, label=f'Objective {(idx1 + 1) % objectives.shape[1] + 1}')

    def _plot_3d_pareto(self, ax, objectives):
        """3-D Pareto front with a bottom color bar; the axes rectangle is
        auto-fitted so the projected cube fills its grid cell."""
        if objectives.shape[1] < 3:
            return

        scatter = ax.scatter(objectives[:, 0], objectives[:, 1], objectives[:, 2],
                             c=objectives[:, 3] if objectives.shape[1] > 3 else objectives[:, 2],
                             cmap='plasma',
                             s=40, alpha=0.8, edgecolors='k', linewidth=0.5)

        if 'best_solution' in self.results:
            best_obj = self.results['best_solution']['objectives']
            ax.scatter(best_obj[0], best_obj[1], best_obj[2],
                       s=200, marker='*', color='red', edgecolors='k',
                       linewidth=1.5, zorder=5, label='Best Compromise')

        ax.xaxis.pane.fill = True
        ax.yaxis.pane.fill = True
        ax.zaxis.pane.fill = True
        ax.xaxis.pane.set_facecolor('whitesmoke')
        ax.yaxis.pane.set_facecolor('whitesmoke')
        ax.zaxis.pane.set_facecolor('whitesmoke')
        ax.grid(True, color='lightgray', alpha=0.3)

        # Uniform [0, 1] limits across the three axes (normalized objective space)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_zlim(0, 1)

        # mplot3d collapses the axes to a square via apply_aspect; zoom only sets the
        # initial scale, the final size is settled by the measurement-driven autofit.
        try:
            ax.set_box_aspect((1, 1, 0.85), zoom=1.12)
        except TypeError:
            ax.set_box_aspect((1, 1, 0.85))

        ax.xaxis.labelpad = 8
        ax.yaxis.labelpad = 8
        ax.zaxis.labelpad = 8
        ax.tick_params(axis='both', which='major', labelsize=10)

        ax.set_xlabel('Growth', fontsize=11)
        ax.set_ylabel('Physiology', fontsize=11)
        ax.set_zlabel('Yield', fontsize=11)
        ax.set_title('3D Pareto Front Projection', fontsize=12)
        # Best-solution note sits beside the bottom color bar (guaranteed conflict-free)
        ax.figure.text(0.42, 0.014, 'red star = Best Compromise', fontsize=11, color='red', va='bottom')

        fig = ax.figure
        cax = fig.add_axes([0.07, 0.010, 0.30, 0.015])
        cbar = fig.colorbar(scatter, cax=cax, orientation='horizontal', label='Quality')
        cbar.ax.tick_params(labelsize=9)

    def _plot_parallel_coordinates(self, ax, objectives):
        """Parallel-coordinates panel with the best compromise solution highlighted."""
        if len(objectives) == 0:
            return

        obj_vals = objectives
        for i in range(len(obj_vals)):
            ax.plot(range(obj_vals.shape[1]), obj_vals[i],
                    alpha=0.2, linewidth=0.5, color='gray')

        if 'best_solution' in self.results:
            best = self.results['best_solution']
            if 'index' in best:
                best_idx = best['index']
                if best_idx < len(obj_vals):
                    ax.plot(range(obj_vals.shape[1]), obj_vals[best_idx],
                            linewidth=3.0, color='red', marker='o',
                            markersize=8, label='Best Compromise', zorder=10)
            elif 'objectives' in best:
                best_obj = best['objectives']
                dists = np.linalg.norm(obj_vals - best_obj, axis=1)
                best_idx = np.argmin(dists)
                ax.plot(range(obj_vals.shape[1]), obj_vals[best_idx],
                        linewidth=3.0, color='red', marker='o',
                        markersize=8, label='Best Compromise', zorder=10)

        ax.set_xticks(range(obj_vals.shape[1]))
        ax.set_xticklabels(['Growth', 'Physiology', 'Yield', 'Quality'][:obj_vals.shape[1]])
        ax.set_ylabel('Normalized Objective Value', fontsize=11)
        ax.set_title('Parallel Coordinates Plot', fontsize=12)
        ax.grid(True, alpha=0.3, linestyle='--')
        ax.legend(loc='best')
        ax.set_ylim([0, 1.1])

    def plot_tradeoff_analysis_matrix(self, save_path: str = None):
        """Trade-off matrix: per-objective histograms on the diagonal, pairwise scatters
        with correlation annotations off-diagonal."""
        if self.results is None or 'objectives' not in self.results:
            raise ValueError("No optimization results available")

        objectives = self.results['objectives']
        n_obj = objectives.shape[1]

        fig, axes = plt.subplots(n_obj, n_obj, figsize=(14, 12))
        fig.subplots_adjust(hspace=0.15, wspace=0.15)

        objective_names = ['Growth', 'Physiology', 'Yield', 'Quality']

        for i in range(n_obj):
            for j in range(n_obj):
                ax = axes[i, j]

                if i == j:
                    ax.hist(objectives[:, i], bins=15,
                            color=self.colors['primary'][i % len(self.colors['primary'])],
                            alpha=0.7, edgecolor='k', linewidth=0.5)

                    mean_val = objectives[:, i].mean()
                    ax.axvline(mean_val, color='red', linestyle='--',
                               linewidth=1.5, alpha=0.8)

                    ax.text(0.05, 0.95, f'Mean: {mean_val:.3f}',
                            transform=ax.transAxes, fontsize=9,
                            verticalalignment='top',
                            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

                    ax.set_xlabel(objective_names[i], fontsize=10)
                    ax.set_ylabel('Frequency', fontsize=10)
                else:
                    scatter = ax.scatter(objectives[:, j], objectives[:, i],
                                         c=np.sum(objectives, axis=1),
                                         cmap='viridis',
                                         s=30, alpha=0.7, edgecolors='k',
                                         linewidth=0.3)

                    corr_coef = np.corrcoef(objectives[:, j], objectives[:, i])[0, 1]
                    ax.text(0.05, 0.95, f'r = {corr_coef:.3f}',
                            transform=ax.transAxes, fontsize=9,
                            verticalalignment='top',
                            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

                    ax.set_xlabel(objective_names[j], fontsize=10)
                    ax.set_ylabel(objective_names[i], fontsize=10)

                ax.tick_params(labelsize=9)
                ax.grid(True, alpha=0.2, linestyle='--')

        plt.suptitle('Trade-off Analysis Matrix: Objective Relationships',
                     fontsize=14, fontweight='bold', y=0.98)
        plt.subplots_adjust(top=0.9)

        if save_path:
            plt.savefig(save_path, dpi=600, bbox_inches='tight')
            print(f"Trade-off matrix saved to {save_path}")

        plt.close()

    def plot_decision_space_analysis(self, save_path: str = None):
        """Decision-space analysis: driver-impact scatter, temperature-regime boxplots,
        irrigation response with trend lines, and variable-importance bars."""
        if self.results is None or 'decision_variables' not in self.results:
            raise ValueError("No optimization results available")

        X = self.results['decision_variables']
        objectives = self.results['objectives']

        light_values = X[:, 0]
        temp_values = X[:, 1]
        water_values = X[:, 2]

        obj_names = ['Growth', 'Physiology', 'Yield', 'Quality']

        fig, axes = plt.subplots(2, 2, figsize=(12, 9))
        fig.subplots_adjust(left=0.08, right=0.92, hspace=0.34, wspace=0.25, top=0.86, bottom=0.08)

        # 1. Light intensity vs objectives
        for i in range(4):
            axes[0, 0].scatter(light_values, objectives[:, i],
                               s=40, alpha=0.7, label=obj_names[i],
                               color=self.colors['primary'][i])
        axes[0, 0].set_xlabel('Light Intensity', fontsize=11)
        axes[0, 0].set_ylabel('Objective Value', fontsize=11)
        axes[0, 0].set_title('Light Intensity Impact', fontsize=12, pad=24)
        axes[0, 0].legend(loc='lower center', bbox_to_anchor=(0.5, 1.01),
                          ncol=4, fontsize=8, frameon=False)
        axes[0, 0].grid(True, alpha=0.3, linestyle='--')

        # 2. Temperature regime boxplots (median split)
        temp_threshold = np.median(temp_values)
        temp_categories = ['Normal' if t < temp_threshold else 'High' for t in temp_values]
        temp_df = pd.DataFrame({
            'Temperature': temp_categories,
            'Growth': objectives[:, 0],
            'Physiology': objectives[:, 1],
            'Yield': objectives[:, 2],
            'Quality': objectives[:, 3]
        })
        temp_melted = temp_df.melt(id_vars=['Temperature'],
                                   value_vars=['Growth', 'Physiology', 'Yield', 'Quality'],
                                   var_name='Objective', value_name='Value')
        sns.boxplot(x='Temperature', y='Value', hue='Objective',
                    data=temp_melted, ax=axes[0, 1],
                    palette=self.colors['primary'])
        axes[0, 1].set_xlabel('Temperature Regime', fontsize=11)
        axes[0, 1].set_ylabel('Objective Value', fontsize=11)
        axes[0, 1].set_title('Temperature Impact', fontsize=12, pad=24)
        axes[0, 1].legend(loc='lower center', bbox_to_anchor=(0.5, 1.01),
                          ncol=4, fontsize=8, frameon=False)
        axes[0, 1].grid(True, alpha=0.3, linestyle='--')

        # 3. Irrigation vs objectives with linear trend lines
        for i in range(4):
            axes[1, 0].scatter(water_values, objectives[:, i],
                               s=40, alpha=0.7, label=obj_names[i],
                               color=self.colors['primary'][i])
            z = np.polyfit(water_values, objectives[:, i], 1)
            p = np.poly1d(z)
            axes[1, 0].plot(np.sort(water_values), p(np.sort(water_values)),
                            color=self.colors['primary'][i], linewidth=2, alpha=0.8)
        axes[1, 0].set_xlabel('Irrigation Level (ETc)', fontsize=11)
        axes[1, 0].set_ylabel('Objective Value', fontsize=11)
        axes[1, 0].set_title('Irrigation Impact', fontsize=12, pad=24)
        axes[1, 0].legend(loc='lower center', bbox_to_anchor=(0.5, 1.01),
                          ncol=4, fontsize=8, frameon=False)
        axes[1, 0].grid(True, alpha=0.3, linestyle='--')

        # 4. Variable importance (mean +/- SD over repeated runs)
        target_total = np.sum(objectives, axis=1)
        var_imp = self._compute_variable_importance_enhanced(X, target_total)
        self.results['variable_importance'] = var_imp

        features = var_imp['features']
        imp_mean = var_imp['combined']['mean']
        imp_std = var_imp['combined']['std']

        axes[1, 1].bar(features, imp_mean, yerr=imp_std, capsize=5,
                       color=self.colors['primary'][:3],
                       alpha=0.8, edgecolor='k', error_kw={'elinewidth': 1.5, 'ecolor': 'gray'})
        axes[1, 1].set_xlabel('Decision Variable', fontsize=11)
        axes[1, 1].set_ylabel('Importance Score', fontsize=11)
        axes[1, 1].set_title('Variable Importance Analysis (Mean ± Std)', fontsize=12)
        axes[1, 1].grid(True, alpha=0.3, linestyle='--', axis='y')
        for i, (v, err) in enumerate(zip(imp_mean, imp_std)):
            axes[1, 1].text(i, v + err + 0.02, f'{v:.3f}', ha='center', fontsize=9)

        plt.suptitle('Decision Space Analysis: Variable-Objective Relationships',
                     fontsize=14, fontweight='bold', y=0.98)

        if save_path:
            plt.savefig(save_path, dpi=600, bbox_inches='tight')
            print(f"Decision space analysis saved to {save_path}")

        plt.close()

    def _create_solution_summary(self) -> str:
        """Plain-text summary of the solution set with the best compromise detail."""
        if self.results is None:
            return "No results available"

        n_solutions = self.results.get('n_solutions', 0)
        best_solution = self.results.get('best_solution', {})

        summary = "Solution Summary\n"
        summary += "=" * 40 + "\n"
        summary += f"Total Pareto Solutions: {n_solutions}\n\n"

        if best_solution:
            if 'variables_original' in best_solution:
                vars_dict = best_solution['variables_original']
                light_desc = f"{vars_dict[0]:.1f} PPFD"
                temp_desc = f"{vars_dict[1]:.1f} °C"
                water_desc = f"{vars_dict[2]:.3f} ETc"
            else:
                vars_dict = best_solution.get('variables', [])
                light_desc = f"{vars_dict[0]:.3f}" if len(vars_dict) > 0 else "N/A"
                temp_desc = "High" if vars_dict[1] > 0.5 else "Normal" if len(vars_dict) > 1 else "N/A"
                water_desc = f"{vars_dict[2]:.3f} ETc" if len(vars_dict) > 2 else "N/A"

            obj_dict = best_solution.get('objectives', [])
            weighted_score = best_solution.get('weighted_score', None)

            summary += "Best Compromise Solution:\n"
            summary += f"  - Light: {light_desc}\n"
            summary += f"  - Temperature: {temp_desc}\n"
            summary += f"  - Irrigation: {water_desc}\n\n"

            summary += "Achieved Objectives:\n"
            objective_names = ['Growth', 'Physiology', 'Yield', 'Quality']
            for i, name in enumerate(objective_names):
                if i < len(obj_dict):
                    summary += f"  - {name}: {obj_dict[i]:.3f}\n"

            if weighted_score is not None:
                summary += f"\nWeighted Score: {weighted_score:.3f}"

        return summary

    def _compute_variable_importance_enhanced(self, X, y):
        """Enhanced variable importance: impurity-based and permutation importances,
        averaged over repeated runs; the two methods are combined by mean."""
        from sklearn.ensemble import RandomForestRegressor
        from sklearn.inspection import permutation_importance

        n_repeats = 10
        rf_importances = []
        perm_importances = []

        for seed in range(n_repeats):
            rf = RandomForestRegressor(n_estimators=100, random_state=seed, n_jobs=-1)
            rf.fit(X, y)
            rf_importances.append(rf.feature_importances_)

            perm = permutation_importance(rf, X, y, n_repeats=5, random_state=seed, n_jobs=-1)
            perm_importances.append(perm.importances_mean)

        rf_mean = np.mean(rf_importances, axis=0)
        rf_std = np.std(rf_importances, axis=0)
        perm_mean = np.mean(perm_importances, axis=0)
        perm_std = np.std(perm_importances, axis=0)

        combined_mean = (rf_mean + perm_mean) / 2
        combined_std = np.sqrt(rf_std ** 2 + perm_std ** 2) / 2

        return {
            'rf': {'mean': rf_mean, 'std': rf_std},
            'permutation': {'mean': perm_mean, 'std': perm_std},
            'combined': {'mean': combined_mean, 'std': combined_std},
            'features': ['Light', 'Temperature', 'Irrigation']
        }

    def create_interactive_dashboard(self, output_file: str = 'interactive_dashboard.html'):
        """Interactive Plotly dashboard: 3-D Pareto front, 3-D decision space, parallel
        coordinates, and variable importance."""
        if self.results is None:
            return

        objectives = self.results.get('objectives', np.array([]))
        X = self.results.get('decision_variables', np.array([]))

        if len(objectives) == 0:
            return

        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('3D Pareto Front', '3D Decision Space',
                            'Parallel Coordinates', 'Variable Importance'),
            specs=[[{'type': 'scatter3d'}, {'type': 'scatter3d'}],
                   [{'type': 'parcoords'}, {'type': 'bar'}]],
            vertical_spacing=0.1,
            horizontal_spacing=0.15,
            column_widths=[0.5, 0.5],
            row_heights=[0.6, 0.4]
        )

        if objectives.shape[1] >= 3:
            fig.add_trace(
                go.Scatter3d(
                    x=objectives[:, 0],
                    y=objectives[:, 1],
                    z=objectives[:, 2],
                    mode='markers',
                    marker=dict(
                        size=6,
                        color=objectives[:, 3] if objectives.shape[1] > 3 else objectives[:, 2],
                        colorscale='Plasma',
                        opacity=0.8,
                        showscale=True,
                        colorbar=dict(title="Quality", x=1.02, y=0.75, len=0.5)
                    ),
                    text=[f'Solution {i}' for i in range(len(objectives))],
                    name='Pareto Solutions'
                ),
                row=1, col=1
            )

        if X.shape[1] >= 3:
            fig.add_trace(
                go.Scatter3d(
                    x=X[:, 0],
                    y=X[:, 1],
                    z=X[:, 2],
                    mode='markers',
                    marker=dict(
                        size=5,
                        color=np.sum(objectives, axis=1),
                        colorscale='Viridis',
                        opacity=0.8,
                        showscale=True,
                        colorbar=dict(title="Total Score", x=1.02, y=0.25, len=0.5)
                    ),
                    text=[f'Total Score: {np.sum(objectives[i]):.3f}'
                          for i in range(len(objectives))],
                    name='Decision Space'
                ),
                row=1, col=2
            )

        dimensions = []
        for i in range(objectives.shape[1]):
            dimensions.append(
                dict(range=[objectives[:, i].min(), objectives[:, i].max()],
                     label=f'Obj {i + 1}', values=objectives[:, i])
            )
        fig.add_trace(
            go.Parcoords(
                line=dict(color=np.sum(objectives, axis=1),
                          colorscale='Viridis'),
                dimensions=dimensions
            ),
            row=2, col=1
        )

        from sklearn.ensemble import RandomForestRegressor
        rf = RandomForestRegressor(n_estimators=100, random_state=42)
        rf.fit(X, np.sum(objectives, axis=1))
        importance = rf.feature_importances_
        features = ['Light', 'Temperature', 'Irrigation']
        fig.add_trace(
            go.Bar(x=features, y=importance,
                   marker_color='#2E86AB',
                   text=[f'{v:.3f}' for v in importance],
                   textposition='outside'),
            row=2, col=2
        )

        fig.update_layout(
            title=dict(text='Multi-Objective Optimization Dashboard', x=0.5, font=dict(size=16)),
            height=800,
            showlegend=False,
            template='plotly_white',
            font=dict(family="Times New Roman", size=12)
        )

        fig.write_html(output_file)
        print(f"Interactive dashboard saved to {output_file}")

        return fig

    def generate_comprehensive_report(self, output_dir: str = 'sci_results'):
        """Generate the full figure set plus summary artifacts into ``output_dir``."""
        import os
        os.makedirs(output_dir, exist_ok=True)

        print(f"\n{'=' * 60}")
        print("GENERATING SCIENTIFIC VISUALIZATION REPORT")
        print(f"{'=' * 60}")

        plots = [
            ('enhanced_pareto_front.png', self.plot_enhanced_pareto_front),
            ('tradeoff_analysis_matrix.png', self.plot_tradeoff_analysis_matrix),
            ('decision_space_analysis.png', self.plot_decision_space_analysis)
        ]

        for filename, plot_func in plots:
            try:
                save_path = os.path.join(output_dir, filename)
                plot_func(save_path)
                print(f"OK {filename}")
            except Exception as e:
                print(f"FAILED {filename}: {e}")

        try:
            dashboard_path = os.path.join(output_dir, 'interactive_dashboard.html')
            self.create_interactive_dashboard(dashboard_path)
            print("OK interactive_dashboard.html")
        except Exception as e:
            print(f"FAILED interactive_dashboard.html: {e}")

        summary_path = os.path.join(output_dir, 'optimization_summary.txt')
        self._generate_summary_file(summary_path)
        print("OK optimization_summary.txt")

        print(f"\nReport generation complete! Results saved to: {output_dir}/")
        print(f"{'=' * 60}")

    def _generate_summary_file(self, filepath: str):
        """Write the optimization summary text (setup, metrics, best solution,
        objective statistics). Boundaries are read from config, never hard-coded."""
        from config import OPTIMIZATION_CONFIG
        _bounds = OPTIMIZATION_CONFIG['variable_bounds']
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write("MULTI-OBJECTIVE OPTIMIZATION SUMMARY REPORT\n")
            f.write("=" * 60 + "\n\n")

            f.write("1. EXPERIMENTAL SETUP\n")
            f.write("-" * 40 + "\n")
            f.write("Decision Variables:\n")
            f.write(f"  - Light Intensity: Continuous [{_bounds['light'][0]}, {_bounds['light'][1]}] PPFD\n")
            f.write(f"  - Temperature: Continuous [{_bounds['temperature'][0]}, {_bounds['temperature'][1]}] °C\n")
            f.write(f"  - Irrigation Level: Continuous [{_bounds['water'][0]}, {_bounds['water'][1]}] ETc\n\n")

            f.write("Objectives:\n")
            f.write("  - Growth: Maximize plant growth metrics\n")
            f.write("  - Physiology: Maximize physiological performance\n")
            f.write("  - Yield: Maximize fruit yield\n")
            f.write("  - Quality: Maximize fruit quality attributes\n\n")

            if self.results:
                f.write("2. OPTIMIZATION RESULTS\n")
                f.write("-" * 40 + "\n")

                n_solutions = self.results.get('n_solutions', 0)
                f.write(f"Total Pareto Optimal Solutions: {n_solutions}\n\n")

                if 'performance_metrics' in self.results:
                    perf = self.results['performance_metrics']
                    f.write("Performance Metrics:\n")
                    f.write(f"  - Hypervolume: {perf.get('hypervolume', 0):.4f}\n")
                    f.write(f"  - Spread: {perf.get('spread', 0):.4f}\n")
                    f.write(f"  - Spacing: {perf.get('spacing', 0):.4f}\n")
                    f.write(f"  - Cache Efficiency: {perf.get('cache_efficiency', 0):.2%}\n")
                    f.write(f"  - Total Evaluations: {perf.get('evaluations', 0)}\n\n")

                if 'best_solution' in self.results:
                    best = self.results['best_solution']
                    f.write("Best Compromise Solution:\n")
                    f.write(f"  - Light Intensity: {best['variables'][0]:.1f} PPFD\n")
                    f.write(f"  - Temperature: {best['variables'][1]:.1f} °C\n")
                    f.write(f"  - Irrigation: {best['variables'][2]:.3f} ETc\n\n")

                    f.write("Achieved Objective Values:\n")
                    obj_names = ['Growth', 'Physiology', 'Yield', 'Quality']
                    for i, name in enumerate(obj_names):
                        if i < len(best['objectives']):
                            f.write(f"  - {name}: {best['objectives'][i]:.3f}\n")

                    f.write(f"  - Total Score: {np.sum(best['objectives']):.3f}\n")
                    if 'weighted_score' in best:
                        f.write(f"  - Weighted Score: {best['weighted_score']:.3f}\n\n")

                if 'objectives' in self.results:
                    objectives = self.results['objectives']
                    if len(objectives) > 0:
                        f.write("\nObjective Statistics:\n")
                        f.write("  Metric      Min       Max       Mean      Std\n")
                        f.write("  " + "-" * 45 + "\n")

                        obj_names = ['Growth', 'Physiology', 'Yield', 'Quality']
                        for i, name in enumerate(obj_names[:objectives.shape[1]]):
                            col = objectives[:, i]
                            f.write(f"  {name:<10} {col.min():<9.3f} {col.max():<9.3f} "
                                    f"{col.mean():<9.3f} {col.std():<9.3f}\n")

            f.write("\n" + "=" * 60 + "\n")
            f.write("END OF REPORT\n")
