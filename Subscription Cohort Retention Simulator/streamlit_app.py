"""
Subscription Cohort Retention Simulator — Streamlit Web App
Deploy free on Streamlit Community Cloud: https://streamlit.io/cloud

Run locally with:  streamlit run streamlit_app.py
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import seaborn as sns
import streamlit as st

sns.set_style("whitegrid")

# ============================================================
# MODULE 1: DATA GENERATOR
# ============================================================

class SyntheticCohortGenerator:
    CHANNELS = {
        'organic': {'cac': 20, 'arpu': 15, 'weibull_shape': 1.3, 'weibull_scale': 25, 'monthly_signups': 800},
        'referral': {'cac': 25, 'arpu': 14, 'weibull_shape': 1.25, 'weibull_scale': 23, 'monthly_signups': 400},
        'paid_search': {'cac': 45, 'arpu': 16, 'weibull_shape': 0.95, 'weibull_scale': 15, 'monthly_signups': 600},
        'paid_social': {'cac': 70, 'arpu': 15, 'weibull_shape': 0.8, 'weibull_scale': 12, 'monthly_signups': 300},
        'direct': {'cac': 35, 'arpu': 17, 'weibull_shape': 1.15, 'weibull_scale': 20, 'monthly_signups': 500}
    }

    def __init__(self, n_months=24, start_date='2023-01-01', discount_rate=0.10):
        self.n_months = n_months
        self.start_date = pd.to_datetime(start_date)
        self.discount_rate = discount_rate
        self.df = None

    def weibull_retention(self, months_since_signup, shape, scale):
        return np.exp(-(months_since_signup / scale) ** shape)

    def generate(self):
        rows = []
        for cohort_idx in range(self.n_months):
            cohort_date = self.start_date + pd.Timedelta(days=30 * cohort_idx)
            cohort_label = cohort_date.strftime('%Y-%m')
            for channel_name, params in self.CHANNELS.items():
                cohort_size = params['monthly_signups']
                cac = params['cac']
                arpu = params['arpu']
                shape = params['weibull_shape']
                scale = params['weibull_scale']
                for months_since in range(self.n_months - cohort_idx):
                    retention_rate = self.weibull_retention(months_since, shape, scale)
                    active_users = int(cohort_size * retention_rate)
                    monthly_revenue = active_users * arpu
                    rows.append({
                        'cohort_month': cohort_label, 'cohort_date': cohort_date, 'channel': channel_name,
                        'cac': float(cac), 'arpu': float(arpu), 'weibull_shape': shape, 'weibull_scale': scale,
                        'months_since_signup': months_since, 'cohort_size': cohort_size,
                        'active_users': active_users, 'retention_rate': retention_rate,
                        'monthly_revenue': float(monthly_revenue)
                    })
        self.df = pd.DataFrame(rows)
        return self.df


# ============================================================
# MODULE 2: COHORT ANALYZER
# ============================================================

class CohortAnalyzer:
    def __init__(self, cohort_df, discount_rate=0.10):
        self.df = cohort_df.copy()
        self.discount_rate = discount_rate
        self.monthly_discount_rate = (1 + discount_rate) ** (1 / 12) - 1
        self._calculate_retention_rates()
        self._calculate_discounted_revenue()

    def _calculate_retention_rates(self):
        self.df['retention_rate'] = (self.df['active_users'] / self.df['cohort_size']).fillna(0)

    def _calculate_discounted_revenue(self):
        self.df['discounted_revenue'] = (
            self.df['monthly_revenue'] / ((1 + self.monthly_discount_rate) ** self.df['months_since_signup'])
        )

    def retention_matrix(self, channel=None, metric='retention_rate'):
        if channel:
            df = self.df[self.df['channel'] == channel].copy()
        else:
            df = self.df.groupby(['cohort_month', 'months_since_signup']).agg({
                'active_users': 'sum', 'cohort_size': 'sum', 'retention_rate': 'mean'
            }).reset_index()

        value_col = 'retention_rate' if metric == 'retention_rate' else 'active_users'
        pivot = df.pivot_table(index='cohort_month', columns='months_since_signup', values=value_col)
        return pivot

    def calculate_ltv_by_cohort_channel(self):
        ltv_results = []
        for (cohort, channel), group in self.df.groupby(['cohort_month', 'channel']):
            group = group.sort_values('months_since_signup').reset_index(drop=True)
            cac = group['cac'].iloc[0]
            arpu = group['arpu'].iloc[0]
            ltv = group['discounted_revenue'].sum()

            cumulative = 0
            payback_month = None
            for idx, row in group.iterrows():
                cumulative += row['monthly_revenue']
                if cumulative >= cac and payback_month is None:
                    payback_month = row['months_since_signup']

            cum_12m = group[group['months_since_signup'] <= 11]['monthly_revenue'].sum()
            ltv_ratio = round(ltv / cac, 2) if cac > 0 else 0

            ltv_results.append({
                'cohort_month': cohort, 'channel': channel, 'cac': cac, 'arpu': arpu,
                'ltv': round(ltv, 2), 'ltv_ratio': ltv_ratio, 'payback_month': payback_month,
                'cumulative_revenue_12m': round(cum_12m, 2)
            })
        return pd.DataFrame(ltv_results)

    def channel_summary(self):
        ltv_cohorts = self.calculate_ltv_by_cohort_channel()
        summary = ltv_cohorts.groupby('channel').agg({
            'cac': 'first', 'ltv': ['mean', 'std'], 'ltv_ratio': ['mean', 'std'],
            'payback_month': ['mean', 'min', 'max'], 'cohort_month': 'count'
        }).round(2)
        summary.columns = ['_'.join(col).strip('_') for col in summary.columns]
        summary = summary.rename(columns={'cohort_month_count': 'cohort_count'})
        return summary.reset_index()

    def payback_analysis(self):
        ltv_cohorts = self.calculate_ltv_by_cohort_channel()
        payback = ltv_cohorts.dropna(subset=['payback_month']).groupby('channel').agg({
            'payback_month': ['min', 'mean', 'max', 'std']
        }).round(1)
        payback.columns = ['_'.join(col).strip('_') for col in payback.columns]
        return payback.reset_index()

    def monthly_retention_curve_by_channel(self, channel):
        channel_data = self.df[self.df['channel'] == channel]
        curve = channel_data.groupby('months_since_signup')['retention_rate'].mean()
        return curve.sort_index()

    def all_retention_curves(self):
        curves = {}
        for channel in self.df['channel'].unique():
            curves[channel] = self.monthly_retention_curve_by_channel(channel)
        return pd.DataFrame(curves)

    def profitability_matrix(self):
        ltv_cohorts = self.calculate_ltv_by_cohort_channel()
        return ltv_cohorts.pivot(index='cohort_month', columns='channel', values='ltv_ratio')


# ============================================================
# MODULE 3: VISUALIZER
# ============================================================

class CohortVisualizer:
    def __init__(self, analyzer, figsize_default=(9, 5.5)):
        self.analyzer = analyzer
        self.figsize_default = figsize_default

    def retention_heatmap(self, channel=None):
        retention = self.analyzer.retention_matrix(channel=channel)
        fig, ax = plt.subplots(figsize=self.figsize_default)
        sns.heatmap(retention * 100, cmap='RdYlGn', annot=False, fmt='.0f',
                    cbar_kws={'label': 'Retention Rate (%)'}, ax=ax,
                    linewidths=0.5, vmin=0, vmax=100)
        ax.set_title(f'Cohort Retention Heatmap - {channel.title() if channel else "All Channels"}',
                     fontsize=13, fontweight='bold')
        ax.set_xlabel('Months Since Signup')
        ax.set_ylabel('Cohort (Signup Month)')
        fig.tight_layout()
        return fig

    def retention_curves_by_channel(self):
        fig, ax = plt.subplots(figsize=self.figsize_default)
        curves = self.analyzer.all_retention_curves()
        for channel in curves.columns:
            ax.plot(curves.index, curves[channel] * 100, marker='o', linewidth=2,
                    markersize=4, label=channel.title())
        ax.set_title('Retention Curves by Acquisition Channel', fontsize=13, fontweight='bold')
        ax.set_xlabel('Months Since Signup')
        ax.set_ylabel('Retention Rate (%)')
        ax.set_ylim(0, 105)
        ax.grid(True, alpha=0.3)
        ax.legend(loc='upper right', fontsize=9)
        fig.tight_layout()
        return fig

    def ltv_by_channel_boxplot(self):
        ltv_data = self.analyzer.calculate_ltv_by_cohort_channel()
        fig, ax = plt.subplots(figsize=self.figsize_default)
        channels = sorted(ltv_data['channel'].unique())
        ltv_by_channel = [ltv_data[ltv_data['channel'] == ch]['ltv'].values for ch in channels]
        bp = ax.boxplot(ltv_by_channel, labels=[ch.title() for ch in channels],
                         patch_artist=True, widths=0.6)
        colors = sns.color_palette("husl", len(channels))
        for patch, color in zip(bp['boxes'], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)
        ax.set_title('Lifetime Value Distribution by Channel', fontsize=13, fontweight='bold')
        ax.set_ylabel('LTV ($)')
        ax.set_xlabel('Acquisition Channel')
        ax.grid(True, alpha=0.3, axis='y')
        plt.setp(ax.get_xticklabels(), rotation=15, ha='right')
        fig.tight_layout()
        return fig

    def ltv_ratio_heatmap(self):
        profitability = self.analyzer.profitability_matrix()
        fig, ax = plt.subplots(figsize=self.figsize_default)
        sns.heatmap(profitability, cmap='YlOrRd', annot=True, fmt='.0f',
                    cbar_kws={'label': 'LTV / CAC Ratio'}, ax=ax, linewidths=0.5)
        ax.set_title('Profitability Heatmap (LTV/CAC Ratio)', fontsize=13, fontweight='bold')
        ax.set_xlabel('Acquisition Channel')
        ax.set_ylabel('Cohort (Signup Month)')
        plt.setp(ax.get_xticklabels(), rotation=15, ha='right')
        fig.tight_layout()
        return fig

    def channel_comparison_dashboard(self):
        fig = plt.figure(figsize=(11, 8))
        gs = GridSpec(2, 2, figure=fig, hspace=0.4, wspace=0.3)

        ax1 = fig.add_subplot(gs[0, 0])
        curves = self.analyzer.all_retention_curves()
        for channel in curves.columns:
            ax1.plot(curves.index, curves[channel] * 100, marker='o', linewidth=1.5, label=channel.title())
        ax1.set_title('Retention Curves', fontsize=11, fontweight='bold')
        ax1.set_xlabel('Months')
        ax1.set_ylabel('Retention (%)')
        ax1.legend(fontsize=7)
        ax1.grid(True, alpha=0.3)

        ax2 = fig.add_subplot(gs[0, 1])
        ltv_data = self.analyzer.calculate_ltv_by_cohort_channel()
        channels = sorted(ltv_data['channel'].unique())
        ltv_by_channel = [ltv_data[ltv_data['channel'] == ch]['ltv'].values for ch in channels]
        bp = ax2.boxplot(ltv_by_channel, labels=[ch.title() for ch in channels], patch_artist=True)
        for patch in bp['boxes']:
            patch.set_facecolor('lightblue')
        ax2.set_title('LTV Distribution', fontsize=11, fontweight='bold')
        ax2.set_ylabel('LTV ($)')
        ax2.grid(True, alpha=0.3, axis='y')
        plt.setp(ax2.get_xticklabels(), rotation=15, ha='right', fontsize=8)

        ax3 = fig.add_subplot(gs[1, 0])
        payback = self.analyzer.payback_analysis()
        if not payback.empty:
            ax3.barh(payback['channel'], payback['payback_month_mean'], color='coral', alpha=0.7)
        ax3.set_xlabel('Months to CAC Payback')
        ax3.set_title('Avg Payback Period', fontsize=11, fontweight='bold')
        ax3.grid(True, alpha=0.3, axis='x')

        ax4 = fig.add_subplot(gs[1, 1])
        channel_summary = self.analyzer.channel_summary()
        ax4.bar(channel_summary['channel'], channel_summary['ltv_ratio_mean'],
                color='lightgreen', alpha=0.7, edgecolor='black')
        ax4.set_title('LTV/CAC Ratio by Channel', fontsize=11, fontweight='bold')
        plt.setp(ax4.get_xticklabels(), rotation=15, ha='right', fontsize=8)
        ax4.grid(True, alpha=0.3, axis='y')

        fig.suptitle('Subscription Cohort Analysis Dashboard', fontsize=14, fontweight='bold')
        return fig


# ============================================================
# MODULE 4: SCENARIO SIMULATOR
# ============================================================

class ScenarioSimulator:
    def __init__(self, original_df, discount_rate=0.10):
        self.original_df = original_df.copy()
        self.discount_rate = discount_rate
        self.scenarios = {}

    def create_scenario(self, name, arpu_multiplier=1.0, retention_improvement=0.0,
                         cac_reduction=0.0, channel_specific=None):
        modified_df = self.original_df.copy()
        modified_df = modified_df.astype({'arpu': 'float64', 'cac': 'float64', 'monthly_revenue': 'float64'})

        if arpu_multiplier != 1.0:
            modified_df['arpu'] = modified_df['arpu'] * arpu_multiplier
            modified_df['monthly_revenue'] = modified_df['monthly_revenue'] * arpu_multiplier

        if cac_reduction != 0.0:
            modified_df['cac'] = modified_df['cac'] * (1 - cac_reduction)

        if retention_improvement != 0.0:
            modified_df['weibull_scale'] = modified_df['weibull_scale'] * (1 + retention_improvement)
            for idx, row in modified_df.iterrows():
                shape = row['weibull_shape']
                scale = row['weibull_scale']
                month = row['months_since_signup']
                new_retention = np.exp(-(month / scale) ** shape)
                modified_df.at[idx, 'retention_rate'] = new_retention
                modified_df.at[idx, 'active_users'] = int(row['cohort_size'] * new_retention)
                modified_df.at[idx, 'monthly_revenue'] = row['arpu'] * modified_df.at[idx, 'active_users']

        if channel_specific:
            for channel, params in channel_specific.items():
                mask = modified_df['channel'] == channel

                if 'arpu_multiplier' in params:
                    mult = params['arpu_multiplier']
                    modified_df.loc[mask, 'arpu'] = modified_df.loc[mask, 'arpu'] * mult
                    modified_df.loc[mask, 'monthly_revenue'] = (
                        modified_df.loc[mask, 'active_users'] * modified_df.loc[mask, 'arpu']
                    )

                if 'cac_reduction' in params:
                    reduction = params['cac_reduction']
                    modified_df.loc[mask, 'cac'] = modified_df.loc[mask, 'cac'] * (1 - reduction)

                if 'retention_improvement' in params:
                    improvement = params['retention_improvement']
                    modified_df.loc[mask, 'weibull_scale'] = (
                        modified_df.loc[mask, 'weibull_scale'] * (1 + improvement)
                    )
                    for idx in modified_df[mask].index:
                        row = modified_df.loc[idx]
                        shape = row['weibull_shape']
                        scale = row['weibull_scale']
                        month = row['months_since_signup']
                        new_retention = np.exp(-(month / scale) ** shape)
                        modified_df.at[idx, 'retention_rate'] = new_retention
                        modified_df.at[idx, 'active_users'] = int(row['cohort_size'] * new_retention)
                        modified_df.at[idx, 'monthly_revenue'] = row['arpu'] * modified_df.at[idx, 'active_users']

        self.scenarios[name] = {'df': modified_df}
        return self

    def get_scenario_ltv(self, scenario_name):
        df = self.scenarios[scenario_name]['df']
        analyzer = CohortAnalyzer(df, discount_rate=self.discount_rate)
        return analyzer.calculate_ltv_by_cohort_channel()

    def compare_scenarios(self, scenarios=None):
        if scenarios is None:
            scenarios = list(self.scenarios.keys())

        original_analyzer = CohortAnalyzer(self.original_df, discount_rate=self.discount_rate)
        original_ltv = original_analyzer.calculate_ltv_by_cohort_channel()

        comparison = []
        for _, row in original_ltv.groupby('channel').agg({
            'cac': 'first', 'ltv': 'mean', 'ltv_ratio': 'mean', 'payback_month': 'mean'
        }).reset_index().iterrows():
            comparison.append({
                'scenario': 'BASELINE (Original)', 'channel': row['channel'],
                'avg_cac': row['cac'], 'avg_ltv': round(row['ltv'], 2),
                'avg_ltv_ratio': round(row['ltv_ratio'], 2),
                'avg_payback_months': round(row['payback_month'], 1) if not pd.isna(row['payback_month']) else None
            })

        for scenario_name in scenarios:
            if scenario_name not in self.scenarios:
                continue
            ltv_df = self.get_scenario_ltv(scenario_name)
            for _, row in ltv_df.groupby('channel').agg({
                'cac': 'first', 'ltv': 'mean', 'ltv_ratio': 'mean', 'payback_month': 'mean'
            }).reset_index().iterrows():
                comparison.append({
                    'scenario': scenario_name, 'channel': row['channel'],
                    'avg_cac': row['cac'], 'avg_ltv': round(row['ltv'], 2),
                    'avg_ltv_ratio': round(row['ltv_ratio'], 2),
                    'avg_payback_months': round(row['payback_month'], 1) if not pd.isna(row['payback_month']) else None
                })

        return pd.DataFrame(comparison)

    def impact_analysis(self, scenario_name):
        original_analyzer = CohortAnalyzer(self.original_df, discount_rate=self.discount_rate)
        original_ltv = original_analyzer.calculate_ltv_by_cohort_channel()
        scenario_ltv = self.get_scenario_ltv(scenario_name)

        impact = []
        for channel in original_ltv['channel'].unique():
            orig = original_ltv[original_ltv['channel'] == channel]
            scen = scenario_ltv[scenario_ltv['channel'] == channel]
            if len(orig) == 0 or len(scen) == 0:
                continue

            orig_avg_ltv = orig['ltv'].mean()
            scen_avg_ltv = scen['ltv'].mean()
            orig_avg_ratio = orig['ltv_ratio'].mean()
            scen_avg_ratio = scen['ltv_ratio'].mean()
            orig_payback = orig['payback_month'].mean()
            scen_payback = scen['payback_month'].mean()

            ltv_change_pct = ((scen_avg_ltv - orig_avg_ltv) / orig_avg_ltv * 100) if orig_avg_ltv > 0 else 0
            ratio_change_pct = ((scen_avg_ratio - orig_avg_ratio) / orig_avg_ratio * 100) if orig_avg_ratio > 0 else 0
            payback_change_months = (scen_payback - orig_payback) if not np.isnan(orig_payback) else 0

            impact.append({
                'channel': channel,
                'baseline_ltv': round(orig_avg_ltv, 2),
                'scenario_ltv': round(scen_avg_ltv, 2),
                'ltv_change_%': round(ltv_change_pct, 1),
                'baseline_ltv_ratio': round(orig_avg_ratio, 2),
                'scenario_ltv_ratio': round(scen_avg_ratio, 2),
                'ratio_change_%': round(ratio_change_pct, 1),
                'baseline_payback_months': round(orig_payback, 1) if not pd.isna(orig_payback) else None,
                'scenario_payback_months': round(scen_payback, 1) if not pd.isna(scen_payback) else None,
                'payback_change_months': round(payback_change_months, 1)
            })
        return pd.DataFrame(impact)


# ============================================================
# STREAMLIT APP
# ============================================================

st.set_page_config(page_title="Cohort Retention Simulator", layout="wide")

st.title("📊 Subscription Cohort Retention Simulator")
st.caption(
    "Predicting time-to-profitability by acquisition channel — LTV, CAC payback, "
    "and scenario modeling for subscription businesses."
)

# ---- Session state init ----
if 'df' not in st.session_state:
    st.session_state.df = None
if 'simulator' not in st.session_state:
    st.session_state.simulator = None
if 'discount_rate' not in st.session_state:
    st.session_state.discount_rate = 0.10

# ---- Sidebar: data generation controls ----
with st.sidebar:
    st.header("1. Generate Data")
    n_months = st.slider("Number of cohort months", min_value=3, max_value=36, value=24)
    start_date = st.text_input("Start date", value="2023-01-01")
    discount_rate = st.slider("Annual discount rate", min_value=0.0, max_value=0.30, value=0.10, step=0.01)

    if st.button("🔄 Generate Data", type="primary", use_container_width=True):
        gen = SyntheticCohortGenerator(n_months=n_months, start_date=start_date, discount_rate=discount_rate)
        df = gen.generate()
        st.session_state.df = df
        st.session_state.simulator = ScenarioSimulator(df, discount_rate=discount_rate)
        st.session_state.discount_rate = discount_rate
        st.success(f"Generated {len(df):,} rows")

    uploaded = st.file_uploader("...or upload a CSV", type="csv")
    if uploaded is not None:
        df = pd.read_csv(uploaded)
        st.session_state.df = df
        st.session_state.simulator = ScenarioSimulator(df, discount_rate=discount_rate)
        st.session_state.discount_rate = discount_rate
        st.success(f"Loaded {len(df):,} rows")

# ---- Main content ----
if st.session_state.df is None:
    st.info("👈 Click **Generate Data** in the sidebar to get started.")
    st.stop()

df = st.session_state.df
analyzer = CohortAnalyzer(df, discount_rate=st.session_state.discount_rate)
visualizer = CohortVisualizer(analyzer)

tab1, tab2, tab3, tab4 = st.tabs(["📈 Overview", "📋 Cohort Analysis", "🎨 Visualizations", "🧪 Scenario Simulator"])

# ---- TAB 1: OVERVIEW ----
with tab1:
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Rows", f"{len(df):,}")
    col2.metric("Cohorts", df['cohort_month'].nunique())
    col3.metric("Channels", df['channel'].nunique())

    ltv_all = analyzer.calculate_ltv_by_cohort_channel()
    best_channel = ltv_all.groupby('channel')['ltv_ratio'].mean().idxmax()
    col4.metric("Best Channel (LTV/CAC)", best_channel.title())

    st.subheader("Channel Summary")
    st.dataframe(analyzer.channel_summary(), use_container_width=True)

    st.subheader("Sample Data")
    st.dataframe(df.head(50), use_container_width=True)

# ---- TAB 2: COHORT ANALYSIS ----
with tab2:
    st.subheader("Channel Summary")
    st.dataframe(analyzer.channel_summary(), use_container_width=True)

    st.subheader("Payback Analysis")
    payback = analyzer.payback_analysis()
    if payback.empty:
        st.warning("No cohorts reached payback within the modeled window.")
    else:
        st.dataframe(payback, use_container_width=True)

    st.subheader("LTV by Cohort × Channel")
    st.dataframe(analyzer.calculate_ltv_by_cohort_channel(), use_container_width=True)

# ---- TAB 3: VISUALIZATIONS ----
with tab3:
    viz_choice = st.selectbox(
        "Choose a visualization",
        ["Retention Heatmap", "Retention Curves", "LTV Boxplot", "Profitability Heatmap", "Full Dashboard"]
    )

    if viz_choice == "Retention Heatmap":
        channel_options = ["All Channels"] + sorted(df['channel'].unique().tolist())
        channel_pick = st.selectbox("Channel", channel_options)
        channel = None if channel_pick == "All Channels" else channel_pick
        fig = visualizer.retention_heatmap(channel=channel)
    elif viz_choice == "Retention Curves":
        fig = visualizer.retention_curves_by_channel()
    elif viz_choice == "LTV Boxplot":
        fig = visualizer.ltv_by_channel_boxplot()
    elif viz_choice == "Profitability Heatmap":
        fig = visualizer.ltv_ratio_heatmap()
    else:
        fig = visualizer.channel_comparison_dashboard()

    st.pyplot(fig)

# ---- TAB 4: SCENARIO SIMULATOR ----
with tab4:
    st.subheader("Create a Scenario")
    with st.form("scenario_form"):
        name = st.text_input("Scenario name", value="My Scenario")
        c1, c2, c3 = st.columns(3)
        arpu_mult = c1.number_input("ARPU multiplier", min_value=0.1, max_value=5.0, value=1.0, step=0.05)
        retention_imp = c2.number_input("Retention improvement", min_value=-0.9, max_value=2.0, value=0.0, step=0.05)
        cac_reduction = c3.number_input("CAC reduction", min_value=-2.0, max_value=0.9, value=0.0, step=0.05)
        submitted = st.form_submit_button("➕ Create Scenario", type="primary")

        if submitted:
            if name.strip() == "":
                st.error("Enter a scenario name.")
            else:
                st.session_state.simulator.create_scenario(
                    name, arpu_multiplier=arpu_mult,
                    retention_improvement=retention_imp, cac_reduction=cac_reduction
                )
                st.success(f"Scenario '{name}' created.")

    simulator = st.session_state.simulator
    if simulator.scenarios:
        st.subheader("Compare All Scenarios")
        st.dataframe(simulator.compare_scenarios(), use_container_width=True)

        st.subheader("Impact Analysis")
        scenario_pick = st.selectbox("Pick a scenario", list(simulator.scenarios.keys()))
        if scenario_pick:
            st.dataframe(simulator.impact_analysis(scenario_pick), use_container_width=True)
    else:
        st.info("Create a scenario above to see comparisons.")

st.divider()
st.caption("Built with Streamlit · pandas · NumPy · Matplotlib · Seaborn")