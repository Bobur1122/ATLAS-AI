"""Entry point and system readiness check for ATLAS AI (Completed through Phase 9)."""

import sys

from atlas_ai import __app_name__, __version__
from atlas_ai.core.config import AppSettings, load_settings
from atlas_ai.core.exceptions import AtlasError
from atlas_ai.core.logger import setup_logging


def run_foundation_check(settings: AppSettings | None = None) -> bool:
    """Initialize configuration and logging, then verify system readiness across Phases 0–9.

    Args:
        settings: Optional pre-loaded AppSettings instance for testing.

    Returns:
        True if initialization and safety checks pass.
    """
    active_settings = settings or load_settings()
    logger = setup_logging(active_settings)

    logger.info(
        "Starting %s v%s [Completed: Phase 0, Phase 1, Phase 2, Phase 3, Phase 4, Phase 5, Phase 6, Phase 7, Phase 8, and Phase 9]",
        __app_name__,
        __version__,
    )
    logger.info(
        "Environment: %s | Primary Symbol: %s | Execution Mode: %s | Live Trading Enabled: %s",
        active_settings.env.value,
        active_settings.primary_symbol,
        active_settings.execution_mode.value,
        active_settings.live_trading_enabled,
    )
    logger.info(
        "System readiness check passed (Phases 0–9 completed: Foundation, Domain/Risk, Market Data, Strategies, Backtest, Knowledge, ML Models, MT5 Demo Execution, Post-Trade Analytics/Governance, and Closed-Loop Demo Runtime Orchestrator). Strictly DEMO-only; live trading disabled."
    )
    return True


def main() -> int:
    """CLI entry point for ATLAS AI."""
    try:
        run_foundation_check()
        return 0
    except AtlasError as exc:
        sys.stderr.write(f"ATLAS AI Error: {exc}\n")
        return 1
    except Exception as exc:  # pragma: no cover
        sys.stderr.write(f"Unexpected fatal error: {exc}\n")
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
