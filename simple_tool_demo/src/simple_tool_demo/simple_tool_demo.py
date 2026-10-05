import logging

from pydantic import BaseModel
from pydantic import Field

from nat.plugin_api import Builder
from nat.plugin_api import FunctionBaseConfig
from nat.plugin_api import FunctionInfo
from nat.plugin_api import LLMFrameworkEnum
from nat.plugin_api import register_function

from .climate_tools import calculate_statistics
from .climate_tools import load_climate_data

logger = logging.getLogger(__name__)


class CalculateStatsInput(BaseModel):
    country: str = Field(
        default="",
        description="Country name to filter by (e.g., 'United States', 'France'). Leave empty for global statistics."
    )


class CalculateStatisticsConfig(FunctionBaseConfig, name="simple_calculate_statistics"):
    """Configuration for calculating climate statistics."""
    data_path: str | None = Field(default=None, description="Path to temperature_annual.csv; defaults to the project's copy.")


@register_function(config_type=CalculateStatisticsConfig, framework_wrappers=[LLMFrameworkEnum.LANGCHAIN])
async def calculate_statistics_tool(config: CalculateStatisticsConfig, builder: Builder):
    """Register tool for calculating climate statistics."""
    df = load_climate_data(config.data_path)

    async def _wrapper(country: str = "") -> str:
        return calculate_statistics(df, country or None)

    yield FunctionInfo.from_fn(
        _wrapper,
        input_schema=CalculateStatsInput,
        description=("Calculate temperature statistics globally or for a specific country. "
                     "Returns JSON with: mean_temperature (°C), min_temperature (°C), max_temperature (°C), "
                     "std_deviation (°C), num_records (count), trend_per_decade (°C/decade), "
                     "years_analyzed (e.g. '1950-2025'), and country (if specified).")
    )
