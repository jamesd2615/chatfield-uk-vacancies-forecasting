Chatfield Week 1: Forecasting Prototype

A basic Python workflow comparing ARIMA and naive forecasting using synthetic monthly data.

1. Requirements

Python 3.11.3 with pandas, NumPy, Matplotlib, statsmodels and scikit-learn. Exact package versions are recorded in requirements.txt.

2. Example dataset

The dataset contains 120 monthly observations, January 2015 to December 2024, labelled at month start. Starting from 100, with an initial change of zero:

    change[t] = 0.6 * change[t-1] + shock[t]
    value[t] = value[t-1] + change[t]

Shocks are independent standard normal draws using NumPy's default_rng with seed 42. Values are artificial index units.

3. How to run

From the project folder, on macOS or Linux:

    python3 -m venv .venv
    source .venv/bin/activate
    python -m pip install -r requirements.txt
    python prototype_arima.py
    
The script generates data/example_series.csv if absent; otherwise it loads the existing file. It saves forecasts.csv, accuracy.csv and forecasts_comparison.png in outputs/, replacing previous outputs.

4. Model Specification

ARIMA(1, 1, 1), fitted using statsmodels with trend="n", includes one autoregressive lag, first differencing and one moving-average lag, without drift. This is a simple initial specification, without parameter optimisation.

The naive benchmark repeats the final training value across the forecast horizon.


5. Training/testing split

The first 108 observations, January 2015 to December 2023, form the training sample. The final 12 months, January to December 2024, form the test sample.
Both models forecast all 12 months from the same fixed origin. Neither uses actual test observations to update its forecasts.


6. Accuracy and results

MAE = mean(abs(actual - forecast))
RMSE = sqrt(mean((actual - forecast) **2))

Both metrics use identical test observations. Lower is better; RMSE gives larger errors more weight.

|     Model    |   MAE  |   RMSE   |
|      ---     |   ---  |    ---   |
|     Naive    | 4.8680 |  5.5446  |
| ARIMA(1,1,1) | 3.2728 |  4.0241  |

ARIMA performs better on this synthetic holdout. These are illustrative results only.

7. Limitations and adaptation

The prototype uses one synthetic series and one split, with no rolling backtesting, seasonal modelling or external predictors. Input validation and convergence checks remain to be strengthened.

For actual vacancy data, replace the input CSV, disable synthetic generation and agree date conventions and model settings. The separate functions allow forecasting, evaluation and plotting to be reused.