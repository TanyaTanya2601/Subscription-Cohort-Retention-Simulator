#!/usr/bin/env python
# coding: utf-8

# In[34]:


import pandas as pd
import numpy as np

df = pd.read_csv('synthetic_cohorts.csv')

print(f"Loaded {len(df)} rows from systhetic_cohorts.csv")
print(f"\nData shape: {df.shape}")
print(f"Columns : {df.columns.tolist()}")
print(f"\nSample Data:")
print(df.head())


# In[63]:


import numpy as np
import pandas as pd
from scipy.stats import weibull_min

class CohortAnalyzer:
    def __init__(self, cohort_df, discount_rate= 0.10):
        self.df = cohort_df.copy()
        self.discount_rate = discount_rate
        self.monthly_discount_rate = (1+ discount_rate) ** (1/12) - 1
        self._calculate_retention_rates()
        self._calculate_discounted_revenue()

    def _calculate_retention_rates(self):
        self.df['retention_rate'] = (
            self.df['active_users'] / self.df['cohort_size']
        ).fillna(0)

    def _calculate_discounted_revenue(self):
        self.df['discounted_revenue'] = (
            self.df['monthly_revenue'] / ((1+ self.monthly_discount_rate) ** self.df['months_since_signup']))

    def retention_matrix(self, channel = None, metric = 'retention_rate'):
        if channel:
            df= self.df[self.df['channel'] == channel].copy()
        else:
            df = self.df.groupby(['cohort_month', 'months_since_signup']).agg({
                'active_users' : 'sum',
                'cohort_size' : 'sum',
                'retention_rate' : 'mean'
            }).reset_index()

        if metric == 'retention_rate':
            pivot = df.pivot_table(
                index = 'cohort_month',
                columns = 'months_since_signup',
                values = 'active_users'
            )

        else:
            pivot = df.pivot_table(
                index = 'cohort_month',
                columns = 'months_since_signup',
                values = 'active_users'
            )

        return pivot

    def calculate_ltv_by_cohort_channel(self):
        ltv_results= []

        for (cohort, channel), group in self.df.groupby(['cohort_month', 'channel']):
            group = group.sort_values('months_since_signup').reset_index(drop = True)

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
                'cohort_month' : cohort,
                'channel' : channel,
                'cac' : cac,
                'arpu' : arpu,
                'ltv' : round(ltv, 2),
                'ltv_ratio' : ltv_ratio,
                'payback_month' : payback_month,
                'cumulative_revenue' : round(cum_12m, 2)
            })
    
        return pd.DataFrame(ltv_results)

    def channel_summary(self):

        ltv_cohorts = self.calculate_ltv_by_cohort_channel()

        summary = ltv_cohorts.groupby('channel').agg({
            'cac' : 'first',
            'ltv' : ['mean', 'std'],
            'ltv_ratio' : ['mean', 'std'],
            'payback_month' : ['mean','min','max'],
            'cohort_month' : 'count'
        }).round(2)

        summary.columns = ['_'.join(col).strip('_') for col in summary.columns]
        summary = summary.rename(columns = {'cohort_month_count' : 'cohort_count'})

        return summary.reset_index()


    def payback_analysis(self):
        ltv_cohorts = self.calculate_ltv_by_cohort_channel()

        payback = ltv_cohorts.dropna(subset=['payback_month']).groupby('channel').agg({
            'payback_month' : ['min', 'mean', 'max', 'std']
        }).round(1)

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

    def time_to_positive_ltv_distribution(self):
        ltv_cohorts = self.calculate_ltv_by_cohort_channel()
        payback_dist = ltv_cohorts.dropna(subset = ['payback_month'])['payback_month'].value_counts().sort_index()

        return payback_dist

    def profitability_matrix(self):
        ltv_cohorts = self.calculate_ltv_by_cohort_channel()
        matrix = ltv_cohorts.pivot(
            index= 'cohort_month',
            columns = 'channel',
            values = 'ltv_ratio'
        )
        return matrix

    def print_summary_report(self):
        print("COHORT RETENTION & PROFITABILITY ANALYSIS")
        
        print("\n1. CHANNEL SUMMARY (Average LTV by Channel)")
        print("-" * 80)
        channel_summary = self.channel_summary()
        print(channel_summary.to_string(index= False))

        print("\n2. PAYBACK PERIOD ANALYSIS (Months to cover CAC)")
        print("-" * 80)
        payback = self.payback_analysis()
        print(payback.to_string(index = False))

        print("\n3. PAYBACK PERIOD DISTRIBUTION")
        print("-" * 80)
        payback_dist = self.time_to_positive_ltv_distribution()
        for month, count in payback_dist.items():
            print(f" Month {int(month):2d}: {int(count):3d} cohorts")

        print("\n4. COHORT EXTREMES (LTV/CAC Ratio)")
        print("-" * 80)
        ltv_cohorts = self.calculate_ltv_by_cohort_channel()
        best = ltv_cohorts.nlargest(5, 'ltv_ratio')[['cohort_month', 'channel', 'ltv_ratio', 'payback_month']]
        worst = ltv_cohorts.nsmallest(5, 'ltv_ratio')[['cohort_month', 'channel', 'ltv_ratio', 'payback_month']]
        
        print("\nBest Performing Cohorts:")
        print(best.to_string(index= False))
        print("\nWorst Performing Cohorts:")
        print(worst.to_string(index = False))

        print("\n" + "=" * 80 + "\n")                


# In[64]:


analyzer = CohortAnalyzer(df, discount_rate = 0.10)

print("CHANNEL SUMMARY\n")
print(analyzer.channel_summary())


# In[65]:


print("PAYBACK ANALYSIS\n")
print(analyzer.payback_analysis())


# In[66]:


print("LTV BY COHORT-CHANNEL- first 20 rows\n")
ltv = analyzer.calculate_ltv_by_cohort_channel()
print(ltv.head(20))


# In[67]:


print("RETENTION CURVES BY CHANNEL")
print(analyzer.all_retention_curves())


# In[68]:


print("PROFITABILITY MATRIX (LTV/CAC Ratio)\n")
print(analyzer.profitability_matrix())


# In[69]:


analyzer.print_summary_report()


# In[ ]:




