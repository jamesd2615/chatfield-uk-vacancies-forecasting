#week 1 Chatfield Project

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#Loading the ONS data:
ons_path = "/Users/riakamat/Documents/Lancaster/Chatfield/week 1/data sets/UK Vacancies (thousands) time series - Total-241026.csv"
data1 = pd.read_csv(ons_path, header=None, names=["label", "value"], dtype=str)
monthly = data1[data1["label"].str.match(r"^\d{4} [A-Z]{3}$")].copy()
monthly["period"] = pd.to_datetime(monthly["label"], format="%Y %b").dt.to_period("M")
ons = monthly.set_index("period")["value"].astype(float).rename("vacancies_k")

print(ons.head())
print("ONS first:", ons.index.min(), "| last:", ons.index.max(), "| months:", len(ons))


#Loading the Indeed data:
indeed_path = "https://raw.githubusercontent.com/hiring-lab/job_postings_tracker/master/GB/aggregate_job_postings_GB.csv"
indeed = pd.read_csv(indeed_path, parse_dates=["date"])
indeed.to_csv("/Users/riakamat/Documents/Lancaster/Chatfield/week 1/data sets/indeed_gb.csv", index=False)   # save a copy for GitHub
print(indeed.head())                  # first 5 rows
print(indeed["variable"].unique())    # shows the two types: 'total postings' and 'new postings'

# --- Split it into the two series we care about ---
# Rows for total postings
total_rows = indeed[indeed["variable"] == "total postings"]
# seasonally adjusted column for the date
total = total_rows.set_index("date")["indeed_job_postings_index_SA"]

# Doing the same for new postings
new_rows = indeed[indeed["variable"] == "new postings"]
new = new_rows.set_index("date")["indeed_job_postings_index_SA"]

print("Indeed first:", total.index.min().date(), "| last:", total.index.max().date(), "| days:", len(total))


#Checking for missing values and gaps in the data
print("\n--- Missing values and gaps ---")
print("ONS missing values:", ons.isna().sum())
all_months = pd.period_range(ons.index.min(), ons.index.max(), freq="M")
print("ONS months skipped:", len(all_months) - len(ons))

print("Indeed missing values (total):", total.isna().sum())
print("Indeed missing values (new):", new.isna().sum())
all_days = pd.date_range(total.index.min(), total.index.max(), freq="D")
print("Indeed days skipped:", len(all_days) - len(total))

print("Indeed value on 1 Feb 2020 (should be 100):", total.loc["2020-02-01"])

print("\nBiggest monthly changes in ONS (thousands):")
print(ons.diff().abs().nlargest(5))   # biggest jumps: are they real events or errors?


#plotting each series:
ons_ts = ons.copy()
ons_ts.index = ons.index.to_timestamp()

fig, ax = plt.subplots(figsize=(10, 4))
ax.plot(ons_ts.index, ons_ts.values)
ax.axvspan(pd.Timestamp("2020-03-01"), pd.Timestamp("2021-06-30"), color="grey", alpha=0.2)
ax.set_title("UK vacancies, all industries (ONS AP2Y, SA, 3-month average)")
ax.set_ylabel("Thousands")
fig.savefig("/Users/riakamat/Documents/Lancaster/Chatfield/week 1/fig1_ons.png", dpi=200, bbox_inches="tight")

fig, ax = plt.subplots(figsize=(10, 4))
ax.plot(total.index, total, label="Total postings (SA)")
ax.plot(new.index, new, label="New postings (SA)")
ax.axhline(100, ls=":", color="grey")
ax.axvspan(pd.Timestamp("2020-03-01"), pd.Timestamp("2021-06-30"), color="grey", alpha=0.2)
ax.set_title("Indeed UK job-postings index (1 Feb 2020 = 100)")
ax.set_ylabel("Index")
ax.legend()
fig.savefig("/Users/riakamat/Documents/Lancaster/Chatfield/week 1/fig2_indeed.png", dpi=200, bbox_inches="tight")


#Descriptive statistics:
print("\n--- ONS, whole period ---")
print(ons.describe().round(1))
print("\n--- ONS, before COVID (to Feb 2020) ---")
print(ons.loc[:"2020-02"].describe().round(1))
print("\n--- ONS, after COVID (Jul 2021 onwards) ---")
print(ons.loc["2021-07":].describe().round(1))
print("\n--- ONS month-on-month % change ---")
print(ons.pct_change().mul(100).describe().round(2))
print("\n--- Indeed total postings ---")
print(total.describe().round(2))
print("\n--- Indeed new postings ---")
print(new.describe().round(2))

print("\nDone. Figures saved in the week 1 folder.")

# TASK 4 - TIME-SERIES PROPERTIES
# using ons data with the real dates
# Non-seasonal adjusted Indeed total postings:
total_nsa = total_rows.set_index("date")["indeed_job_postings_index_NSA"]
 
# Trend: average level in each year
print("\n Task 4: Trend - ONS yearly average (thousands)")
print(ons_ts.resample("YS").mean().round(0))
print("\n Task 4: Trend - Indeed total postings yearly average ")
print(total.resample("YS").mean().round(1))
 
# Seasonality
# ONS data is only seasonally adjusted, so use Indeed's two versions:
# non-SA / SAdjusted above 1 = that month is normally busier than average.
print("\n Task 4: Seasonality - Indeed unadjusted / adjusted by calendar month")
ratio = total_nsa / total
print(ratio.groupby(ratio.index.month).mean().round(3))
 
# Volatility: how big are the month-to-month moves?
pct = ons_ts.pct_change() * 100
print("\n Task 4: Volatility - standard deviation of ONS monthly % change")
print("Before COVID:", round(pct.loc[:"2020-02-01"].std(), 2))
print("COVID (Mar 2020 - Jun 2021):", round(pct.loc["2020-03-01":"2021-06-01"].std(), 2))
print("After COVID:", round(pct.loc["2021-07-01":].std(), 2))
 
fig, ax = plt.subplots(figsize=(10, 4))
ax.bar(pct.index, pct, width=25, label="Month-on-month % change")
ax.plot(pct.rolling(12).std(), color="orange", label="12-month rolling standard deviation")
ax.axhline(0, color="black", linewidth=0.8)
ax.set_title("ONS vacancies: monthly % change and volatility")
ax.set_ylabel("%")
ax.legend()
fig.savefig("/Users/riakamat/Documents/Lancaster/Chatfield/week 1/fig3_volatility.png", dpi=200, bbox_inches="tight")
 
# Stationarity
from statsmodels.graphics.tsaplots import plot_acf
from statsmodels.tsa.stattools import adfuller
 
fig, axes = plt.subplots(1, 2, figsize=(11, 3.5))
plot_acf(ons_ts.loc[:"2020-02-01"], lags=36, ax=axes[0], title="ONS level (pre-COVID)")
plot_acf(ons_ts.loc[:"2020-02-01"].diff().dropna(), lags=36, ax=axes[1], title="ONS monthly change (pre-COVID)")
fig.savefig("/Users/riakamat/Documents/Lancaster/Chatfield/week 1/fig4_acf.png", dpi=200, bbox_inches="tight")

# Volatility and stationarity for Indeed data
# Indeed only starts in Feb 2020, so there is no period before COVID
# Using monthly averages so it is comparable with ONS.
indeed_monthly = total.resample("MS").mean()
indeed_pct = indeed_monthly.pct_change() * 100
print("\n--- TASK 4: Volatility - standard deviation of Indeed monthly % change ---")
print("COVID (Mar 2020 - Jun 2021):", round(indeed_pct.loc["2020-03-01":"2021-06-01"].std(), 2))
print("After COVID:", round(indeed_pct.loc["2021-07-01":].std(), 2))
 
fig, ax = plt.subplots(figsize=(10, 4))
ax.bar(indeed_pct.index, indeed_pct, width=25, label="Month-on-month % change")
ax.plot(indeed_pct.rolling(12).std(), color="orange", label="12-month rolling standard deviation")
ax.axhline(0, color="black", linewidth=0.8)
ax.set_title("Indeed total postings: monthly % change and volatility")
ax.set_ylabel("%")
ax.legend()
fig.savefig("/Users/riakamat/Documents/Lancaster/Chatfield/week 1/fig3b_indeed_volatility.png", dpi=200, bbox_inches="tight")
 
fig, axes = plt.subplots(1, 2, figsize=(11, 3.5))
plot_acf(indeed_monthly, lags=24, ax=axes[0], title="Indeed level (monthly)")
plot_acf(indeed_monthly.diff().dropna(), lags=24, ax=axes[1], title="Indeed monthly change")
fig.savefig("/Users/riakamat/Documents/Lancaster/Chatfield/week 1/fig4b_indeed_acf.png", dpi=200, bbox_inches="tight")
 
print("\n--- TASK 4: Stationarity - Indeed (ADF test p-values) ---")
print("Indeed level:", round(adfuller(indeed_monthly)[1], 3))
print("Indeed monthly change:", round(adfuller(indeed_monthly.diff().dropna())[1], 4))
 
 
# COVID period
print("\n Task 4: COVID")
print("ONS months inside Mar 2020 - Jun 2021:", len(ons_ts.loc["2020-03-01":"2021-06-01"]))
print("ONS lowest:", ons_ts.min(), "in", ons_ts.idxmin().strftime("%b %Y"))
print("ONS highest:", ons_ts.max(), "in", ons_ts.idxmax().strftime("%b %Y"))
print("Indeed lowest:", total.min(), "on", total.idxmin().date())
 
 
# Task 5 - Are the datasets suitable?

#Make Indeed monthly: average the days in each month
total_monthly = total.resample("MS").mean()
days_in_month = total.resample("MS").count()
total_monthly = total_monthly[days_in_month >= 28]     # not usinig the unfinished latest month
 
# Make Indeed a 3-month average like ONS 
total_3m = total_monthly.rolling(3).mean()
 
# ONS labels each window using its middle month (Eg: May 2001 = Apr-Jun 2001)
# Add one month so ONS is also labelled by the last month of the window
ons_end = ons_ts.copy()
ons_end.index = ons_ts.index + pd.DateOffset(months=1)
 
# Keep only the months in both series = the overlap
both = pd.concat([ons_end, total_3m], axis=1, join="inner").dropna()
both.columns = ["ons", "indeed"]
 
print("\n TASK 5: Overlap and sample size")
print("Overlap: windows ending", both.index.min().strftime("%b %Y"), "to", both.index.max().strftime("%b %Y"))
print("Number of overlapping 3-month windows:", len(both))
print("Of which after COVID (ending Jul 2021 or later):", len(both.loc["2021-07-01":]))
print("Left to fit the model if the last 24 are kept for testing:", len(both) - 24)
print("ONS months before Indeed starts (only ARIMA can use these):", len(ons_ts.loc[:"2020-01-01"]))
 
# Chart: both series on the same scale (first common window = 100)
rebased = both / both.iloc[0] * 100
fig, ax = plt.subplots(figsize=(10, 4))
ax.plot(rebased.index, rebased["ons"], label="ONS vacancies")
ax.plot(rebased.index, rebased["indeed"], label="Indeed total postings (3-month average)")
ax.axhline(100, ls=":", color="grey")
ax.axvspan(pd.Timestamp("2020-03-01"), pd.Timestamp("2021-06-30"), color="grey", alpha=0.2)
ax.set_title("Overlap period: both series rebased (first common window = 100)")
ax.set_ylabel("Index")
ax.legend()
fig.savefig("/Users/riakamat/Documents/Lancaster/Chatfield/week 1/fig5_overlap.png", dpi=200, bbox_inches="tight")
 
# Do they move together? Comparing the changes, not levels 
changes = np.log(both).diff().dropna()
print("\n Task 5: Do the series move together?")
print("Correlation of levels:", round(both["ons"].corr(both["indeed"]), 3))
print("Correlation of changes:", round(changes["ons"].corr(changes["indeed"]), 3))
 
# Shift Indeed by k months. Highest at k = 0 means the label alignment is right.
print("Correlation of changes with Indeed shifted by k months:")
for k in [-2, -1, 0, 1, 2, 3]:
    print("  k =", k, ":", round(changes["ons"].corr(changes["indeed"].shift(k)), 3))
 
# Real-time availability 
print("\n TASK 5: Was Indeed available when each forecast was made?")
print("Latest ONS window ends:", ons_end.index.max().strftime("%b %Y"), "(published about 3 weeks after it ends)")
print("Latest Indeed day:", total.index.max().date(), "(about 1 week behind)")
 
