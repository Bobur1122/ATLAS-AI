"""Exceptions for the ATLAS AI Phase 8 Post-Trade Analytics & Continuous Feedback Loop package."""

from atlas_ai.core.exceptions import AtlasError, TradingSafetyError


class AnalyticsError(AtlasError):
    """Base exception for all Phase 8 post-trade analytics, journal, and feedback errors."""


class TradeReconstructionError(TradingSafetyError, AnalyticsError):
    """Raised when closed or partially closed trade reconstruction fails closed due to
    DEAL_ENTRY_INOUT / DEAL_ENTRY_OUT_BY deals, volume mismatches, conflicting duplicate tickets,
    inline + separate commission double-counting, or missing broker deal history.
    """


class FeedbackLeakageError(TradingSafetyError, AnalyticsError):
    """Raised when a post-trade feedback dataset or retraining split violates chronological
    ordering or purge separation (`purge_bars >= horizon_bars`).
    """


class GovernancePromotionRejectedError(TradingSafetyError, AnalyticsError):
    """Raised when a candidate model or parameter recommendation fails out-of-sample
    walk-forward promotion gates or attempts to relax any safety/risk parameter.
    """
