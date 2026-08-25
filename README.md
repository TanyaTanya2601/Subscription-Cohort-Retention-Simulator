# Subscription Cohort Retention Simulator

A desktop tool that models subscription cohort retention and unit economics across acquisition channels — predicting when a cohort becomes profitable, not just whether it churns.

### The problem it solves:

Most churn dashboards answer "how many users left this month." This tool answers a different, more business-relevant question: which acquisition channels are actually worth the spend, and how long until they pay back?

It models five channels (organic, referral, paid search, paid social, direct) with distinct retention curves (via Weibull survival functions), then calculates LTV, LTV/CAC ratio, and payback period per cohort-channel pair — plus a scenario simulator to test "what if we raise prices 10% and cut CAC 15%?" before making the call.

### Features
Data generation — synthetic cohort data with configurable channel parameters (CAC, ARPU, retention shape/scale) and time window
Cohort analysis — retention matrices, LTV/CAC ratios, payback period by channel and cohort
Visualisations — retention heatmaps, retention curves, LTV distribution, profitability heatmap, full comparison dashboard
Scenario simulator — model pricing, retention, and CAC changes (globally or per-channel) and compare against baseline

### Screenshots

<img width="1911" height="1006" alt="image" src="https://github.com/user-attachments/assets/2c6fee0c-7fa5-484b-895e-8437b467af1e" />
<img width="1917" height="408" alt="image" src="https://github.com/user-attachments/assets/1fec66f4-4ce5-4245-b392-fb4b329b8d30" />
<img width="1910" height="965" alt="image" src="https://github.com/user-attachments/assets/98baa39a-8ca6-46e5-96f8-8a9a7142c4be" />
<img width="1915" height="1011" alt="image" src="https://github.com/user-attachments/assets/b1ed4eac-e7ec-4743-8c64-6fa3f6ef6802" />
<img width="1911" height="1015" alt="image" src="https://github.com/user-attachments/assets/3c3ea8e1-02b8-41e5-adae-8be48873c1e9" />


### Key result

Across the modelled channels, organic acquisition shows roughly a 13x higher LTV/CAC ratio than paid social ($20 CAC vs. $70 CAC, with organic users also retaining substantially longer). The scenario simulator lets you test how much of that gap closes under different pricing or retention assumptions.

### Tech stack

Python, pandas, NumPy, Matplotlib, Seaborn, Tkinter

### Project structure
app_gui.py --> Entire app: data generator, analyzer, visualizer, scenario simulator, and Tkinter GUI in one file
sample_data.csv --> Pre-generated sample dataset

Built by Tanya as a portfolio project demonstrating unit-economics analysis and business-facing data science, not just model accuracy.
