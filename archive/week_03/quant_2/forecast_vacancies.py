from pathlib import Path

import matplotlib.pyplot as plt 
import numpy as np 
import pandas as pd 

from sklearn.metrics import mean_absolute_error, root_mean_squared_error
from statsmodels.tsa.arima.model import ARIMA

BASE_DIR = Path(__file__).resolve().parent

DATA_DIR = BASE_DIR / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
OUTPUT_DIR = BASE_DIR / "outputs"

DATA_FILE = RAW_DIR / "ons_ap2y_raw.csv"

TEST_SIZE = 12
ARIMA_ORDER = (1, 1, 1)
ARIMA_TREND = "n"


def load_series(file_path):
    #Load ONS monthly vacancies using window-ending labels#

    if not file_path.is_file():
        raise FileNotFoundError(
            f"ONS input file not found: {file_path}"
        )
    
    data = pd.read_csv(
        file_path,
        header=None,
        names=["label", "value"],
        dtype=str,
    )

    data["label"] = data["label"].str.strip()

    monthly = data.loc[
        data["label"].str.fullmatch(
        r"\d{4} [A-Z]{3}",
        na=False,
        )
    ].copy()

    if monthly.empty:
        raise ValueError(
            "No monthly observations found. "
            "Check that the input is the AP2Y time-series CSV."
        )
    
    middle_month = pd.to_datetime(
        monthly["label"],
        format="%Y %b",
        errors="raise",
    )

    monthly["date"] = (
        middle_month.dt.to_period("M") + 1
    ).dt.to_timestamp()

    monthly["value"] = pd.to_numeric(
        monthly["value"],
        errors="raise",
    )

    series = (
        monthly.set_index("date")["value"]
        .sort_index()
        .astype(float)
        .rename("vacancies_k")
    )

    if series.index.has_duplicates:
        raise ValueError("Duplicate monthly observations found.")

    series = series.asfreq("MS")

    if not np.isfinite(series.to_numpy()).all():
        raise ValueError(
            "Missing months or invalid vacancy observations found."
        )

    if (series < 0).any():
        raise ValueError("Negative vacancy observations found.")

    series.index.name = "date"

    return series

    

def split_series(series, test_size):
    #reserve the final observations for chronological evaluations#
    
    train = series.iloc[:-test_size]
    test = series.iloc[-test_size:]

    return train, test

def forecast_arima(history, order, trend):
    #fit ARIMA to available history and forecast one observation#

    fitted = ARIMA(
        history,
        order=order,
        trend=trend,
    ).fit()

    if not fitted.mle_retvals.get("converged", True):
        raise RuntimeError(
            "ARIMA did not converge using history through "
            f"{history.index[-1]:%Y-%m}."
        )
    
    return float(fitted.forecast(steps=1).iloc[0])

def forecast_naive(history):
    #forecast the next observation using the latest actual value#

    return float(history.iloc[-1])

def backtest_forecasts(series, test_size, order, trend):
    #generate expanding window one-step-ahead forecasts#

    train, test = split_series(series, test_size)
    rows = []

    for position in range(len(train), len(series)):
        history = series.iloc[:position]
        target_date = series.index[position]

        arima_forecast = forecast_arima(
            history,
            order=order,
            trend=trend,
        )

        naive_forecast = forecast_naive(history)

        rows.append({
            "date": target_date,
            "origin_label": history.index[-1],
            "actual": float(series.iloc[position]),
            "arima": arima_forecast,
            "naive": naive_forecast,
        })

    comparison = pd.DataFrame(rows).set_index("date")

    values = comparison[
        ["actual", "arima", "naive"]
    ].to_numpy()

    if not np.isfinite(values).all():
        raise ValueError("Forecast comparison contains invalid values.")
    
    if not comparison.index.equals(test.index):
        raise ValueError("Forecast dates do not match the test sample")
    
    return comparison



def evaluate_forecasts(comparison, order):
    #calculate MAE and RMSE for identical test observations#

    rows = []

    for column, model_name in [
        ("naive", "Naive"),
        ("arima", f"ARIMA{order}"),
    ]:
        rows.append({
            "model": model_name,
            "MAE": mean_absolute_error(
                comparison["actual"],
                comparison[column],
            ),
            "RMSE": root_mean_squared_error(
                comparison["actual"],
                comparison[column],
            ),
        })

    return pd.DataFrame(rows).set_index("model")

def plot_forecasts(train, comparison, file_path, order):
    #visual comparison of observations and forecasts#
    
    recent_train = train.tail(36)


    fig, ax = plt.subplots(figsize=(11, 5))

    ax.plot(
        recent_train.index, recent_train,
        label="Training observations",
        color="tab:blue",
    )

    ax.plot(
        comparison.index, comparison["actual"],
        label="Actual test observation",
        color="black",
    )

    ax.plot(
        comparison.index, comparison["arima"],
        label=f"ARIMA{order}",
        color="tab:orange",
        linestyle="--",
    )

    ax.plot(
        comparison.index, comparison["naive"],
        label="Naive benchmark",
        color="tab:green",
        linestyle="--",
    )

    ax.axvline(
        comparison.index[0],
        color="grey",
        linestyle="--",
        label="Start of test period",
    )

    ax.set_title("UK vacancies: Expanding-window one-step-ahead forecasts")
    ax.set_xlabel("Three-month window ending")
    ax.set_ylabel("Vacancies (thousands)")

    ax.legend()
    ax.grid(alpha=0.25)

    fig.tight_layout()
    fig.savefig(file_path, dpi=300)
    plt.close(fig)


def main():
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    series = load_series(DATA_FILE)
    train, test = split_series(series, TEST_SIZE)

    comparison = backtest_forecasts(
        series,
        test_size=TEST_SIZE,
        order=ARIMA_ORDER,
        trend=ARIMA_TREND,
    )

    results = evaluate_forecasts(
        comparison,
        order=ARIMA_ORDER,
    )

    expected_naive = series.shift(1).loc[test.index]

    np.testing.assert_allclose(
        comparison["naive"].to_numpy(),
        expected_naive.to_numpy(),
        rtol=0,
        atol=0,
    )

    series.to_csv(
        PROCESSED_DIR / "ons_vacancies.csv",
        date_format="%Y-%m-%d",
    )

    comparison.to_csv(
        OUTPUT_DIR / "forecasts.csv",
        date_format="%Y-%m-%d",
    )

    results.to_csv(
        OUTPUT_DIR / "accuracy.csv",
    )

    plot_forecasts(
        train,
        comparison,
        OUTPUT_DIR / "forecasts_comparison.png",
        order=ARIMA_ORDER,
    )

    print(f"Loaded {len(series)} ONS observations.")

    print(
        f"Initial training: {train.index[0]:%Y-%m} to "
        f"{train.index[-1]:%Y-%m} "
        f"({len(train)} observations)"
    )

    print(
        f"Testing: {test.index[0]:%Y-%m} to "
        f"{test.index[-1]:%Y-%m} "
        f"({len(test)} observations)"
    )

    print(
        f"Model: ARIMA{ARIMA_ORDER}, "
        f"trend={ARIMA_TREND!r}"
    )

    print("\nExpanding-window one-step-ahead accuracy:")
    print(results.round(4).to_string())

    print("\nDates identify three-month window-ending months.")
    print(f"Outputs saved to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()



