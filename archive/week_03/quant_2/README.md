Chatfield Week 2: UK Vacancy Forecasting

A Python workflow extending the Week 1 prototype to compare ARIMA and naive one-step-ahead forecasts using real ONS vacancy data.

1. Requirements

Python 3.11.3 with pandas, NumPy, Matplotlib, statsmodels and scikit-learn. Exact versions from the Week 2 environment are recorded in requirements.txt.

2. ONS dataset

The input is ONS AP2Y, UK total vacancies in thousands, seasonally adjusted and reported as overlapping three-month averages. The saved release of 20 October 2026 contains 304 monthly observations, covering April–June 2001 to July–September 2026.

Source: [ONS AP2Y time series](https://www.ons.gov.uk/employmentandlabourmarket/peopleinwork/employmentandemployeetypes/timeseries/ap2y/lms), also published in [VACS01](https://www.ons.gov.uk/employmentandlabourmarket/peoplenotinwork/unemployment/datasets/vacanciesandunemploymentvacs01).

Place the unmodified AP2Y time-series CSV at data/raw/ons_ap2y_raw.csv. The loader retains monthly rows, sorts them and rejects duplicate dates, missing months and invalid values. Native middle-month labels are shifted forward once to identify the window-ending month: 2001 MAY becomes 2001-06-01, representing April–June 2001. These dates are labels, not publication dates.

3. How to run

From the Week 2 project folder, on macOS or Linux:

    python3 -m venv .venv
    source .venv/bin/activate
    python -m pip install -r requirements.txt
    python forecast_vacancies.py

The script requires the saved raw CSV and makes no downloads. It writes the cleaned series to data/processed/ons_vacancies.csv and saves forecasts.csv, accuracy.csv and forecasts_comparison.png in outputs/, replacing previous outputs.

4. Model specification

The provisional ARIMA(1,1,1) specification uses statsmodels with trend="n", without drift. Vacancy levels are supplied directly; the model handles first differencing. ARIMA_ORDER, ARIMA_TREND and TEST_SIZE are configurable near the top of the script. Model selection remains separate from this implementation.

The naive forecast is the latest actual vacancy observation available at each origin.

5. Training/testing split

The first 292 observations, labelled June 2001 to September 2025, form the initial training sample. The final 12, October 2025 to September 2026, form the test sample.

Both models forecast one observation ahead using the same expanding history. ARIMA is refitted at each origin. Each actual test observation enters the history only after its forecast is produced. In forecasts.csv, date identifies the target window and origin_label identifies the last available observation. The figure displays only the final 36 training observations for readability; estimation uses the full available history.

6. Accuracy and results

MAE = mean(abs(actual - forecast))
RMSE = sqrt(mean((actual - forecast) ** 2))

Both metrics use the same 12 test observations and are measured in thousands of vacancies. Lower is better; RMSE gives larger errors more weight.

|     Model    |   MAE  |  RMSE  |
|      ---     |   ---: |   ---: |
|     Naive    | 5.0833 | 6.2517 |
| ARIMA(1,1,1) | 4.5801 | 6.4405 |

ARIMA has lower MAE, while naive has lower RMSE. Neither model performs better on both measures.

7. Limitations and next steps

Results use one short holdout and a saved data vintage, not historical release-by-release information. ONS revisions and overlapping windows limit interpretation. COVID observations remain in the estimation history. The ARIMA specification is provisional; Indeed predictors and ARIMAX are reserved for the next stage.