"""Application configuration and safety gate validation for ATLAS AI."""

from enum import StrEnum
from pathlib import Path
from typing import Literal

from pydantic import Field, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

from atlas_ai.core.exceptions import TradingSafetyError


class EnvironmentMode(StrEnum):
    """Supported runtime environment modes."""

    DEVELOPMENT = "development"
    TESTING = "testing"
    PRODUCTION = "production"


class ExecutionMode(StrEnum):
    """Supported trading execution modes."""

    BACKTEST = "backtest"
    DEMO = "demo"
    LIVE = "live"


class AppSettings(BaseSettings):
    """Validated application settings loaded from environment variables or .env file."""

    model_config = SettingsConfigDict(
        env_prefix="ATLAS_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        frozen=True,
    )

    env: EnvironmentMode = Field(
        default=EnvironmentMode.DEVELOPMENT,
        description="Runtime environment (development, testing, production).",
    )
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"] = Field(
        default="INFO",
        description="Application logging level.",
    )
    log_dir: Path = Field(
        default=Path("logs"),
        description="Directory where rotating log files are stored.",
    )
    log_to_file: bool = Field(
        default=True,
        description="Whether to write logs to disk in addition to console output.",
    )
    primary_symbol: str = Field(
        default="XAUUSD",
        min_length=3,
        max_length=16,
        description="Primary trading instrument symbol.",
    )
    execution_mode: ExecutionMode = Field(
        default=ExecutionMode.BACKTEST,
        description="Current execution mode (backtest, demo, live).",
    )
    live_trading_enabled: bool = Field(
        default=False,
        description="Explicit safety flag required before live execution mode can be selected.",
    )

    @field_validator("primary_symbol")
    @classmethod
    def normalize_symbol(cls, value: str) -> str:
        """Normalize trading symbol to uppercase and strip whitespace."""
        cleaned = value.strip().upper()
        if not cleaned.isalnum():
            raise ValueError(f"Primary symbol must be alphanumeric, got '{value}'.")
        return cleaned

    @model_validator(mode="after")
    def enforce_trading_safety_gates(self) -> "AppSettings":
        """Ensure live trading is never active unless explicitly authorized."""
        if self.execution_mode == ExecutionMode.LIVE and not self.live_trading_enabled:
            raise TradingSafetyError(
                "SAFETY VIOLATION: ATLAS_EXECUTION_MODE='live' is blocked because "
                "ATLAS_LIVE_TRADING_ENABLED is False. Live trading must remain disabled "
                "unless explicitly authorized."
            )
        if self.live_trading_enabled and self.execution_mode == ExecutionMode.LIVE:
            # In Phase 0, live trading infrastructure does not exist yet; block live mode altogether.
            raise TradingSafetyError(
                "SAFETY VIOLATION: Live trading is not permitted in Phase 0 (Project Foundation). "
                "Start with 'backtest' or 'demo' mode only."
            )
        return self


def load_settings(**overrides: object) -> AppSettings:
    """Load and validate application settings, optionally applying keyword overrides."""
    return AppSettings(**overrides)  # type: ignore[arg-type]
