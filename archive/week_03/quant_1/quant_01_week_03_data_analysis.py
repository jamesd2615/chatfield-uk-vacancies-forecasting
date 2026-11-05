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
 


 
# WEEK 2 - PREPARING THE DATASETS FOR THE FORECASTING MODELS
# Uses data1, indeed, total and new from the Week 1 code above.
# Outputs: ons_monthly.csv and ons_indeed_aligned.csv

week2_folder = "/Users/riakamat/Documents/Lancaster/Chatfield/week 2/"

# COVID period used in Week 1
covid_start = pd.Period("2020-03", freq="M")
covid_end = pd.Period("2021-06", freq="M")


# TASK 1 - FINALISE THE ONS DATASET

data1["label"] = data1["label"].str.strip()

# The rows at the top of the ONS file are information about the series (title, CDID, release date)
info = data1[~data1["label"].str.match(r"^\d{4}")].set_index("label")["value"]
print("\n Week 2 - Task 1: ONS dataset")
print("ONS series:", info.get("Title"), "| CDID:", info.get("CDID"))
print("ONS release date:", info.get("Release date"), "| next release:", info.get("Next release"))

# Keep only the monthly rows (e.g. "2001 MAY"); the file also has yearly and quarterly rows
monthly = data1[data1["label"].str.match(r"^\d{4} [A-Z]{3}$")].copy()
monthly["period"] = pd.to_datetime(monthly["label"], format="%Y %b").dt.to_period("M")
monthly["vacancies_k"] = pd.to_numeric(monthly["value"], errors="coerce")
ons = monthly.set_index("period")["vacancies_k"].sort_index()

# Checks: no missing values, no duplicate months, no skipped months
all_months = pd.period_range(ons.index.min(), ons.index.max(), freq="M")
print("Missing values:", ons.isna().sum())
print("Duplicate months:", ons.index.duplicated().sum())
print("Months skipped:", len(all_months) - len(ons))
ons = ons[~ons.index.duplicated()].reindex(all_months)   # any skipped month would show up as a blank row

# Build the cleaned table.
# ONS labels each 3-month window by its middle month (May 2001 = Apr-Jun 2001)
ons_monthly = pd.DataFrame(index=ons.index)
ons_monthly["ons_label_month"] = ons.index.strftime("%Y-%m")
ons_monthly["window_start"] = (ons.index - 1).strftime("%Y-%m")
ons_monthly["window_end"] = (ons.index + 1).strftime("%Y-%m")

def window_name(mid):
    start, end = mid - 1, mid + 1
    if start.year == end.year:
        return start.strftime("%b") + "-" + end.strftime("%b %Y")
    return start.strftime("%b %Y") + "-" + end.strftime("%b %Y")

ons_monthly["window"] = [window_name(p) for p in ons.index]
ons_monthly["vacancies_k"] = ons.values

# Monthly change in logs (x100 = roughly % change). Week 1 found the models should use changes.
ons_monthly["dlog_vacancies"] = (np.log(ons).diff() * 100).round(3).values

# Flag windows that include any month from Mar 2020 - Jun 2021
start_p, end_p = ons.index - 1, ons.index + 1
ons_monthly["covid_window"] = ((end_p >= covid_start) & (start_p <= covid_end)).astype(int)

# ONS revises the latest 3 months in each release, so flag them
ons_monthly["provisional"] = 0
ons_monthly.iloc[-3:, ons_monthly.columns.get_loc("provisional")] = 1

# Approximate first publication: ONS publishes in the Labour market overview,
# in the month after the window ends (usually within six weeks of the survey date)
ons_monthly["approx_first_published"] = (ons.index + 2).strftime("%Y-%m")

ons_monthly = ons_monthly.reset_index(drop=True)
ons_monthly.to_csv(week2_folder + "ons_monthly.csv", index=False)

print("ONS rows:", len(ons_monthly), "| first window:", ons_monthly["window"].iloc[0],
      "| last window:", ons_monthly["window"].iloc[-1])
print(ons_monthly.head())


# TASK 2 - PREPARE THE ALIGNED ONS-INDEED DATASET

# Save a dated copy of the Indeed file used (Indeed revises its history)
download_date = pd.Timestamp.today().strftime("%Y-%m-%d")
indeed.to_csv(week2_folder + "indeed_gb_downloaded_" + download_date + ".csv", index=False)

total = total.sort_index()
new = new.sort_index()

# Same rule as Week 1: monthly average of the daily values, only months with at least 28 days
days_in_month = total.resample("MS").count()
total_monthly = total.resample("MS").mean()[days_in_month >= 28]
new_monthly = new.resample("MS").mean()[days_in_month >= 28]
print("\n Week 2 - Task 2: Aligned dataset")
print("Dropped unfinished months:", list(days_in_month[days_in_month < 28].index.strftime("%Y-%m")))

# 3-month average so it covers the same months as the ONS window, labelled by the last month
total_3m = total_monthly.rolling(3).mean()
new_3m = new_monthly.rolling(3).mean()

# Alternative version: ONS counts vacancies on one date near the start of each month,
# so also take the Indeed value on the first Friday of each month and average 3 of them
fridays = total[total.index.dayofweek == 4]
first_friday = fridays.groupby(fridays.index.to_period("M")).first()
first_friday.index = first_friday.index.to_timestamp()
first_friday = first_friday[first_friday.index.isin(total_monthly.index)]
total_3m_survey = first_friday.rolling(3).mean()

# Put the ONS data on the window end month (same +1 month shift as Week 1)
ons_end = ons.copy()
ons_end.index = (ons.index + 1).to_timestamp()

indeed_3m = pd.DataFrame({"indeed_total_sa_3m": total_3m,
                          "indeed_new_sa_3m": new_3m,
                          "indeed_total_sa_3m_survey_dates": total_3m_survey})

aligned = pd.concat([ons_end.rename("ons_vacancies_k"), indeed_3m], axis=1, join="inner")
aligned = aligned.dropna(subset=["ons_vacancies_k", "indeed_total_sa_3m"])

# Add the ONS window information back in, matched on window end
info_cols = ons_monthly.set_index("window_end")[["window", "window_start", "ons_label_month",
                                                 "covid_window", "provisional", "approx_first_published"]]
aligned.index = aligned.index.strftime("%Y-%m")
aligned.index.name = "window_end"
aligned = aligned.join(info_cols)

# Monthly changes in logs for both series
aligned["dlog_ons"] = (np.log(aligned["ons_vacancies_k"]).diff() * 100).round(3)
aligned["dlog_indeed_total"] = (np.log(aligned["indeed_total_sa_3m"]).diff() * 100).round(3)

for c in ["indeed_total_sa_3m", "indeed_new_sa_3m", "indeed_total_sa_3m_survey_dates"]:
    aligned[c] = aligned[c].round(3)

aligned = aligned.reset_index()[["window_end", "window_start", "window", "ons_label_month",
                                 "ons_vacancies_k", "indeed_total_sa_3m", "indeed_new_sa_3m",
                                 "indeed_total_sa_3m_survey_dates", "dlog_ons", "dlog_indeed_total",
                                 "covid_window", "provisional", "approx_first_published"]]
aligned.to_csv(week2_folder + "ons_indeed_aligned.csv", index=False)


# CHECKS ON THE ALIGNED DATASET (should match the Week 1 results)
print("Windows:", aligned["window"].iloc[0], "to", aligned["window"].iloc[-1], "| rows:", len(aligned))
print("Of which not COVID windows:", (aligned["covid_window"] == 0).sum())
print("Missing values in each column:")
print(aligned.isna().sum())

ch = aligned[["dlog_ons", "dlog_indeed_total"]].dropna()
print("Correlation of changes (calendar-month averages):", round(ch["dlog_ons"].corr(ch["dlog_indeed_total"]), 3))
survey_ch = np.log(aligned["indeed_total_sa_3m_survey_dates"]).diff() * 100
print("Correlation of changes (first-Friday version):", round(aligned["dlog_ons"].corr(survey_ch), 3))
print("Correlation of changes with Indeed shifted by k months:")
for k in [-2, -1, 0, 1, 2]:
    print("  k =", k, ":", round(aligned["dlog_ons"].corr(aligned["dlog_indeed_total"].shift(k)), 3))

# Real-time availability: Indeed data for a window is complete about a week after it ends,
# ONS for the same window is published in the following month
print("\nLatest ONS window:", aligned["window"].iloc[-1], "| approx. published:", aligned["approx_first_published"].iloc[-1])
print("Latest Indeed day:", total.index.max().date())

print("\nDone. ons_monthly.csv and ons_indeed_aligned.csv saved in the week 2 folder.")