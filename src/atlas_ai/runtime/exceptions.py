"""Exceptions for the ATLAS AI Phase 9 Runtime Orchestration & Emergency Circuit Breaker package."""

from atlas_ai.core.exceptions import AtlasError, TradingSafetyError


class RuntimeOrchestrationError(AtlasError):
    """Base exception for all Phase 9 runtime orchestration and circuit-breaker errors."""


class EmergencyStopError(TradingSafetyError, RuntimeOrchestrationError):
    """Base safety exception for emergency stop and circuit-breaker violations."""


class CircuitBreakerLatchedError(EmergencyStopError):
    """Raised when trading authorization is requested while the persistent emergency circuit breaker is latched."""


class EmergencyStopResetError(EmergencyStopError):
    """Raised when an operator reset of the emergency circuit breaker is invalid (e.g., empty rationale)."""
