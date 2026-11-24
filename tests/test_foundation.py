"""Phase 0 tests for configuration, logging, safety gates, and application bootstrap."""

from pathlib import Path

import pytest
from pydantic import ValidationError

from atlas_ai import __version__
from atlas_ai.core.config import (
    AppSettings,
    EnvironmentMode,
    ExecutionMode,
    load_settings,
)
from atlas_ai.core.exceptions import TradingSafetyError
from atlas_ai.core.logger import get_logger, setup_logging
from atlas_ai.main import main, run_foundation_check


def test_default_settings_are_safe() -> None:
    """Default configuration must target XAUUSD, backtest mode, and disable live trading."""
    settings = load_settings(log_to_file=False)
    assert settings.env == EnvironmentMode.DEVELOPMENT
    assert settings.primary_symbol == "XAUUSD"
    assert settings.execution_mode == ExecutionMode.BACKTEST
    assert settings.live_trading_enabled is False


def test_symbol_normalization_and_validation() -> None:
    """Primary symbol must be normalized to uppercase and reject invalid characters."""
    settings = load_settings(primary_symbol=" xauusd ", log_to_file=False)
    assert settings.primary_symbol == "XAUUSD"

    with pytest.raises(ValidationError):
        load_settings(primary_symbol="XAU/USD!", log_to_file=False)


def test_live_trading_safety_gate_blocks_unauthorized_live_mode() -> None:
    """Selecting live execution mode must raise TradingSafetyError."""
    with pytest.raises(TradingSafetyError, match="SAFETY VIOLATION"):
        load_settings(
            execution_mode=ExecutionMode.LIVE,
            live_trading_enabled=False,
            log_to_file=False,
        )

    # Even if live_trading_enabled=True is passed, Phase 0 blocks live mode completely.
    with pytest.raises(TradingSafetyError, match="not permitted in Phase 0"):
        load_settings(
            execution_mode=ExecutionMode.LIVE,
            live_trading_enabled=True,
            log_to_file=False,
        )


def test_logging_setup_writes_to_file(tmp_path: Path) -> None:
    """Logger must create the log directory and write formatted entries to atlas_ai.log."""
    settings = AppSettings(
        env=EnvironmentMode.TESTING,
        log_level="DEBUG",
        log_dir=tmp_path / "test_logs",
        log_to_file=True,
    )
    logger = setup_logging(settings)
    child_logger = get_logger("test_module")
    child_logger.info("Phase 0 test log message for %s", settings.primary_symbol)

    for handler in logger.handlers:
        handler.flush()

    log_file = tmp_path / "test_logs" / "atlas_ai.log"
    assert log_file.exists()
    content = log_file.read_text(encoding="utf-8")
    assert "Phase 0 test log message for XAUUSD" in content
    assert "atlas_ai.test_module" in content

    # Clean up file handlers on Windows
    for handler in list(logger.handlers):
        handler.close()
        logger.removeHandler(handler)


def test_run_foundation_check_and_cli_entry(tmp_path: Path) -> None:
    """Foundation check and CLI main() must succeed under valid settings."""
    settings = AppSettings(
        env=EnvironmentMode.TESTING,
        log_dir=tmp_path / "cli_logs",
        log_to_file=False,
    )
    assert run_foundation_check(settings) is True
    assert __version__ == "0.1.0"
    assert main() == 0
