"""Exceptions for the ATLAS AI Phase 7 MetaTrader 5 Integration & Demo Execution Engine."""

from atlas_ai.core.exceptions import AtlasError, TradingSafetyError


class ExecutionError(AtlasError):
    """Base exception for all Phase 7 MT5 execution, connection, and persistence errors."""


class MT5BackendNotAvailableError(ExecutionError):
    """Raised when RealMetaTrader5Backend is invoked in an environment without the MetaTrader5 package."""


class MT5ConnectionError(ExecutionError):
    """Raised when terminal initialization, connection health check, or reconnection fails."""


class MT5AuthenticationError(TradingSafetyError, ExecutionError):
    """Raised when MT5 account login or server does not match expected configured credentials."""


class LiveTradingBlockedError(TradingSafetyError, ExecutionError):
    """Raised immediately whenever a real/live or contest trading account or LIVE execution mode is detected."""


class AccountStateReconstructionError(TradingSafetyError, ExecutionError):
    """Raised when daily starting balance, deal history, or commission accounting cannot be reliably verified."""


class StaleMarketDataError(ExecutionError):
    """Raised when MT5 tick or candle data is missing, malformed, or older than max_tick_age_seconds."""


class OrderValidationError(TradingSafetyError, ExecutionError):
    """Raised when an order request fails pre-trade safety, spread, stop-level, filling-mode, or RR checks."""


class DuplicateOrderError(TradingSafetyError, ExecutionError):
    """Raised when an order for the same signal_id, client_order_id, or active setup has already been submitted."""


class AmbiguousExecutionStateError(TradingSafetyError, ExecutionError):
    """Raised when new order submissions are blocked due to an unresolved SUBMITTING or UNKNOWN_PENDING_RECONCILIATION entry."""


class OrderExecutionError(ExecutionError):
    """Raised when order submission or broker reconciliation encounters an execution error."""


class PersistenceError(ExecutionError):
    """Base exception for crash-safe persistence and file-locking errors."""


class PersistenceLockTimeoutError(PersistenceError):
    """Raised when an exclusive cross-process file lock cannot be acquired within the configured timeout."""


class PersistenceCorruptionError(TradingSafetyError, PersistenceError):
    """Raised when a persisted state or ledger file fails JSON parsing or SHA-256 checksum verification."""


class PersistenceRollbackDetectedError(TradingSafetyError, PersistenceError):
    """Raised when a persisted file has a lower monotonic generation than its backup or prior known state."""


class LedgerRecoveryLockError(PersistenceCorruptionError):
    """Raised when persistence or the execution ledger is locked due to an interrupted write or missing primary file awaiting broker reconciliation."""


class PersistenceAccountMismatchError(
    AccountStateReconstructionError,
    PersistenceCorruptionError,
    MT5AuthenticationError,
):
    """Raised when a persisted state file is bound to a different account_login or server than the active MT5 account."""

