"""Simple climate data tools for exploring country-level annual temperatures.

Expects a CSV with one row per country per year and columns:
country_code, country_name, year, avg_temperature, temperature_anomaly.
The bundled resources/climate_data/temperature_annual.csv is built from
Our World in Data's ERA5 (Copernicus) surface temperature series, 1940 onward.
"""

import json
from pathlib import Path

import pandas as pd

# climate_analyzer/src/climate_analyzer/climate_tools.py -> project root (nat_projects)
DEFAULT_DATA_PATH = (
    Path(__file__).resolve().parents[3] / "resources" / "climate_data" / "temperature_annual.csv"
)


def load_climate_data(path: str | Path | None = None) -> pd.DataFrame:
    """Load the annual temperature CSV into a DataFrame sorted by country and year."""
    path = Path(path) if path else DEFAULT_DATA_PATH
    if not path.exists():
        raise FileNotFoundError(
            f"Climate data not found at {path}. Expected default location: {DEFAULT_DATA_PATH}"
        )

    df = pd.read_csv(path)
    df = df.dropna(subset=["avg_temperature"])
    return df.sort_values(["country_name", "year"]).reset_index(drop=True)


def list_countries(df: pd.DataFrame) -> list[str]:
    """Return the sorted list of country names present in the data."""
    return sorted(df["country_name"].unique().tolist())


def filter_by_country(
    df: pd.DataFrame,
    country: str,
    start_year: int | None = None,
    end_year: int | None = None,
) -> pd.DataFrame:
    """Return rows for one country, matched by name or ISO3 code (case-insensitive).

    Optionally restricts to an inclusive year range.
    """
    key = country.strip().lower()
    mask = (df["country_name"].str.lower() == key) | (df["country_code"].str.lower() == key)
    if not mask.any():
        raise ValueError(
            f"Country '{country}' not found. Use list_countries() to see valid names."
        )
    if start_year is not None:
        mask &= df["year"] >= start_year
    if end_year is not None:
        mask &= df["year"] <= end_year
    return df[mask].sort_values("year").reset_index(drop=True)


def calculate_statistics(df: pd.DataFrame, country: str | None = None) -> str:
    """Temperature statistics globally or for one country, as a JSON string.

    Keys: mean_temperature, min_temperature, max_temperature, std_deviation (°C),
    num_records, trend_per_decade (°C/decade), years_analyzed, and country if given.
    """
    if country:
        try:
            df = filter_by_country(df, country)
        except ValueError as e:
            return json.dumps({"error": str(e)})
    if df.empty:
        return json.dumps({"error": "No data to summarise."})

    temps = df["avg_temperature"]
    stats = {
        "mean_temperature": round(float(temps.mean()), 2),
        "min_temperature": round(float(temps.min()), 2),
        "max_temperature": round(float(temps.max()), 2),
        "std_deviation": round(float(temps.std()), 2),
        "num_records": int(len(df)),
        "years_analyzed": f"{int(df['year'].min())}-{int(df['year'].max())}",
    }

    # Linear trend over time; average across countries first so each year counts once
    by_year = df.groupby("year")["avg_temperature"].mean()
    if len(by_year) >= 2:
        x = by_year.index.to_numpy(dtype=float)
        y = by_year.to_numpy(dtype=float)
        slope = ((x - x.mean()) * (y - y.mean())).sum() / ((x - x.mean()) ** 2).sum()
        stats["trend_per_decade"] = round(float(slope * 10), 3)

    if country:
        stats["country"] = df["country_name"].iloc[0]
    return json.dumps(stats)


def find_extreme_years(df: pd.DataFrame, n: int = 5) -> dict:
    """Return the n hottest and n coldest years (averaged across countries in df)."""
    by_year = df.groupby("year")["avg_temperature"].mean().round(2)
    return {
        "hottest": [
            {"year": int(y), "temperature": float(t)}
            for y, t in by_year.nlargest(n).items()
        ],
        "coldest": [
            {"year": int(y), "temperature": float(t)}
            for y, t in by_year.nsmallest(n).items()
        ],
    }


PLOT_TYPES = ("annual_trend", "country_comparison", "anomaly")


def create_visualization(
    df: pd.DataFrame,
    plot_type: str = "annual_trend",
    save_path: str | Path = "temperature_plot.png",
    countries: list[str] | None = None,
    rolling_window: int = 10,
) -> str:
    """Create a temperature chart, save it to save_path and return a summary message.

    plot_type:
      - "annual_trend": average temperature across all countries in df per year,
        with a rolling average
      - "country_comparison": one line per country in `countries`
      - "anomaly": bars of the average temperature anomaly per year
        (red above the baseline, blue below)
    """
    if plot_type not in PLOT_TYPES:
        raise ValueError(f"Unknown plot_type '{plot_type}'. Choose one of: {', '.join(PLOT_TYPES)}")
    if plot_type == "country_comparison" and not countries:
        raise ValueError("plot_type='country_comparison' needs a list of countries.")

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(12, 6))

    if plot_type == "annual_trend":
        series = df.groupby("year")["avg_temperature"].mean()
        ax.plot(series.index, series.values, alpha=0.3, label="Yearly average")
        ax.plot(
            series.index,
            series.rolling(rolling_window, min_periods=1).mean(),
            linewidth=2,
            label=f"{rolling_window}-yr rolling average",
        )
        ax.set_ylabel("Average temperature (°C)")
        ax.set_title("Annual temperature trend")
    elif plot_type == "country_comparison":
        for country in countries:
            series = filter_by_country(df, country).set_index("year")["avg_temperature"]
            line = ax.plot(series.index, series.values, alpha=0.3)[0]
            ax.plot(
                series.index,
                series.rolling(rolling_window, min_periods=1).mean(),
                color=line.get_color(),
                linewidth=2,
                label=f"{country} ({rolling_window}-yr avg)",
            )
        ax.set_ylabel("Average temperature (°C)")
        ax.set_title("Temperature comparison by country")
    else:
        series = df.groupby("year")["temperature_anomaly"].mean()
        colors = ["#d62728" if v > 0 else "#1f77b4" for v in series.values]
        ax.bar(series.index, series.values, color=colors)
        ax.axhline(0, color="black", linewidth=0.8)
        ax.set_ylabel("Temperature anomaly (°C)")
        ax.set_title("Temperature anomaly by year")

    ax.set_xlabel("Year")
    ax.grid(alpha=0.3)
    if plot_type != "anomaly":
        ax.legend()
    fig.tight_layout()

    save_path = Path(save_path)
    fig.savefig(save_path, dpi=120)
    plt.close(fig)
    return f"Saved {plot_type} plot to {save_path}"
