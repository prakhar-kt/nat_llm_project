import json
import logging
from functools import lru_cache

import pandas as pd
from pydantic import Field

from nat.plugin_api import Builder
from nat.plugin_api import FunctionBaseConfig
from nat.plugin_api import FunctionInfo
from nat.plugin_api import LLMFrameworkEnum
from nat.plugin_api import register_function

from . import climate_tools
from .climate_tools import DEFAULT_DATA_PATH

logger = logging.getLogger(__name__)


@lru_cache(maxsize=4)
def _load(data_path: str):
    return climate_tools.load_climate_data(data_path)


def _subset(df, country: str, start_year: int, end_year: int):
    """Filter to one country if given, otherwise apply only the year range to all countries."""
    if country.strip():
        return climate_tools.filter_by_country(df, country, start_year or None, end_year or None)
    if start_year:
        df = df[df["year"] >= start_year]
    if end_year:
        df = df[df["year"] <= end_year]
    return df


class ClimateToolConfig(FunctionBaseConfig):
    data_path: str = Field(default=str(DEFAULT_DATA_PATH), description="Path to temperature_annual.csv.")


class ListCountriesConfig(ClimateToolConfig, name="list_countries"):
    """List countries available in the climate dataset."""


class CalculateStatisticsConfig(ClimateToolConfig, name="calculate_statistics"):
    """Temperature statistics for a country or globally."""


class FindExtremeYearsConfig(ClimateToolConfig, name="find_extreme_years"):
    """Hottest and coldest years for a country or globally."""


class CreateVisualizationConfig(ClimateToolConfig, name="create_visualization"):
    """Create and save a temperature chart."""
    output_dir: str = Field(default=".", description="Directory where charts are saved.")


@register_function(config_type=ListCountriesConfig, framework_wrappers=[LLMFrameworkEnum.LANGCHAIN])
async def list_countries_function(config: ListCountriesConfig, builder: Builder):

    async def _list_countries(search: str = "") -> str:
        """
        List the countries available in the climate dataset.

        Args:
            search (str): Optional case-insensitive text to filter country names, e.g. "united".
                Leave empty to list all countries.

        Returns:
            str: Comma-separated country names.
        """
        names = climate_tools.list_countries(_load(config.data_path))
        if search.strip():
            names = [n for n in names if search.strip().lower() in n.lower()]
        return f"{len(names)} countries: " + ", ".join(names) if names else f"No countries match '{search}'."

    yield FunctionInfo.from_fn(_list_countries, description=_list_countries.__doc__)


@register_function(config_type=CalculateStatisticsConfig, framework_wrappers=[LLMFrameworkEnum.LANGCHAIN])
async def calculate_statistics_function(config: CalculateStatisticsConfig, builder: Builder):

    async def _calculate_statistics(country: str = "", start_year: int = 0, end_year: int = 0) -> str:
        """
        Calculate annual temperature statistics (mean, min, max, standard deviation and
        warming trend per decade, all in °C) from 1940 onward.

        Args:
            country (str): Country name or ISO3 code, e.g. "India" or "IND". Leave empty for all countries.
            start_year (int): First year to include, or 0 for no lower bound.
            end_year (int): Last year to include, or 0 for no upper bound.

        Returns:
            str: JSON object with the statistics.
        """
        try:
            subset = _subset(_load(config.data_path), country, start_year, end_year)
        except ValueError as e:
            return str(e)
        stats = json.loads(climate_tools.calculate_statistics(subset))
        stats["scope"] = country or "all countries"
        return json.dumps(stats)

    yield FunctionInfo.from_fn(_calculate_statistics, description=_calculate_statistics.__doc__)


@register_function(config_type=FindExtremeYearsConfig, framework_wrappers=[LLMFrameworkEnum.LANGCHAIN])
async def find_extreme_years_function(config: FindExtremeYearsConfig, builder: Builder):

    async def _find_extreme_years(country: str = "", n: int = 5) -> str:
        """
        Find the hottest and coldest years by average temperature (°C).

        Args:
            country (str): Country name or ISO3 code. Leave empty to average across all countries.
            n (int): How many hottest and coldest years to return.

        Returns:
            str: JSON object with "hottest" and "coldest" lists of {year, temperature}.
        """
        try:
            subset = _subset(_load(config.data_path), country, 0, 0)
        except ValueError as e:
            return str(e)
        return json.dumps(climate_tools.find_extreme_years(subset, n))

    yield FunctionInfo.from_fn(_find_extreme_years, description=_find_extreme_years.__doc__)


@register_function(config_type=CreateVisualizationConfig, framework_wrappers=[LLMFrameworkEnum.LANGCHAIN])
async def create_visualization_function(config: CreateVisualizationConfig, builder: Builder):

    async def _create_visualization(plot_type: str = "annual_trend", countries: str = "", filename: str = "") -> str:
        """
        Create a temperature chart and save it as a PNG.

        Args:
            plot_type (str): One of "annual_trend" (global average over time),
                "country_comparison" (one line per country; requires countries) or
                "anomaly" (yearly deviation from the baseline).
            countries (str): Comma-separated country names for "country_comparison", e.g. "India, Canada".
            filename (str): Optional output filename; defaults to "<plot_type>.png".

        Returns:
            str: Where the chart was saved, or an error message.
        """
        from pathlib import Path

        names = [c.strip() for c in countries.split(",") if c.strip()] or None
        save_path = Path(config.output_dir) / (filename or f"{plot_type}.png")
        try:
            df = _load(config.data_path)
            # annual_trend/anomaly average over df, so restrict df to the requested countries
            if names and plot_type != "country_comparison":
                df = pd.concat([climate_tools.filter_by_country(df, n) for n in names])
            return climate_tools.create_visualization(df, plot_type=plot_type, save_path=save_path, countries=names)
        except ValueError as e:
            return str(e)

    yield FunctionInfo.from_fn(_create_visualization, description=_create_visualization.__doc__)
