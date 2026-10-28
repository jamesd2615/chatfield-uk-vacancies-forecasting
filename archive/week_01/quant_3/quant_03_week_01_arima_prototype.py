from pathlib import Path

import matplotlib.pyplot as plt 
import numpy as np 
import pandas as pd 

from sklearn.metrics import mean_absolute_error, root_mean_squared_error
from statsmodels.tsa.arima.model import ARIMA

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
OUTPUT_DIR = BASE_DIR / "outputs"

DATA_FILE = DATA_DIR / "example_series.csv"

SEED = 42
N_OBSERVATION = 120
TEST_SIZE = 12
ARIMA_ORDER = (1, 1, 1)


def generate_example_data(file_path):
    #Generate synthetic monthly series#

    rng = np.random.default_rng(SEED)

    dates = pd.date_range(
        start="2015-01-01",
        periods=N_OBSERVATION,
        freq="MS",
    )

    values = np.zeros(N_OBSERVATION)
    values[0] = 100.0

    monthly_change = 0.0

    for t in range(1, N_OBSERVATION):
        shock = rng.normal(loc=0.0, scale=1.0)
        monthly_change = 0.6 * monthly_change + shock
        values[t] = values[t-1] + monthly_change

    data = pd.DataFrame({
        "date": dates,
        "value": values,
    })

    data.to_csv(file_path, index=False)


def load_series(file_path):
    #Load a complete monthly series with a date index#

    data = pd.read_csv(file_path)

    data["date"] = pd.to_datetime(data["date"])
    data["value"] = pd.to_numeric(data["value"])

    series = data.set_index("date")["value"].sort_index()

    return series.asfreq("MS")

def split_series(series, test_size):
    #reserve the final observations as the test period#
    
    train = series.iloc[:-test_size]
    test = series.iloc[-test_size:]

    return train, test

def forecast_arima(train, test_index):
    #fit once and forecast the entire test period#

    model = ARIMA(
        train,
        order=ARIMA_ORDER,
        trend="n",
    )

    fitted = model.fit()
    forecasts = fitted.forecast(steps=len(test_index))

    return forecasts

def forecast_naive(train, test_index):
    #repeat the final training value for every forcast month#

    return pd.Series(
        train.iloc[-1],
        index=test_index,
        name="naive",
    )

def evaluate_forecasts(comparison):
    #calculate MAE and RMSE for identical test observations#

    rows = []

    for column, model_name in [
        ("naive", "Naive"),
        ("arima", "ARIMA(1, 1, 1)"),
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

def plot_forecasts(train, comparison, file_path):
    #visual comparison of observations and forecasts#

    fig, ax = plt.subplots(figsize=(11, 5))

    ax.plot(
        train.index, train,
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
        label="ARIMA(1, 1, 1)",
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
        train.index[-1],
        color="grey",
        linestyle="--",
        label="Forecast origin",
    )

    ax.set_title("Synthetic monthly series: fixed origin forecast comparison")
    ax.set_xlabel("Date")
    ax.set_ylabel("Synthetic index, arbitrary units")

    ax.legend()
    ax.grid(alpha=0.25)

    fig.tight_layout()
    fig.savefig(file_path, dpi=300)
    plt.close(fig)


def main():
    DATA_DIR.mkdir(exist_ok=True)
    OUTPUT_DIR.mkdir(exist_ok=True)

    if not DATA_FILE.exists():
        generate_example_data(DATA_FILE)

    series = load_series(DATA_FILE)

    train, test = split_series(series, TEST_SIZE)

    arima_forecasts = forecast_arima(train, test.index)
    naive_forecasts = forecast_naive(train, test.index)

    comparison = pd.DataFrame({
        "actual": test,
        "arima": arima_forecasts,
        "naive": naive_forecasts,
    })

    comparison.index.name = "date"
    results = evaluate_forecasts(comparison)

    comparison.to_csv(OUTPUT_DIR / "forecasts.csv")
    results.to_csv(OUTPUT_DIR / "accuracy.csv")

    plot_forecasts(
        train,
        comparison,
        OUTPUT_DIR / "forecasts_comparison.png",
    )

    print(f"Loaded {len(series)} monthly observations.")

    print(
        f"Training: {train.index[0]:%Y-%m} to "
        f"{train.index[-1]:%Y-%m} ({len(train)} observations)"
    )

    print(
        f"Testing:  {test.index[0]:%Y-%m} to "
        f"{test.index[-1]:%Y-%m} ({len(test)} observations)"
    )

    print("\nFixed-origin forecast accuracy:")
    print(results.round(4).to_string())

    print(f"\nOutputs saved to: {OUTPUT_DIR}")
    print("Illustrative synthetic results only.")


if __name__ == "__main__":
    main()



