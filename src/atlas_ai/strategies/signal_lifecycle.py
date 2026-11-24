"""Auditable Signal Lifecycle Manager for ATLAS AI."""

from datetime import datetime, timedelta
from decimal import Decimal
import uuid

from atlas_ai.domain.risk import RiskDecision
from atlas_ai.domain.signals import SignalDirection, TradeSignal
from atlas_ai.strategies.config import SignalLifecycleConfig
from atlas_ai.strategies.exceptions import SignalLifecycleError
from atlas_ai.strategies.models import (
    LifecycleEvent,
    ManagedSignalRecord,
    RiskEvaluationStatus,
    SignalLifecycleState,
)

# Valid state transitions enforcing that a signal NEVER becomes ACTIVE without passing RISK_APPROVED
ALLOWED_STATE_TRANSITIONS: dict[SignalLifecycleState, frozenset[SignalLifecycleState]] = {
    SignalLifecycleState.GENERATED: frozenset(
        {
            SignalLifecycleState.PENDING_CONFIRMATION,
            SignalLifecycleState.CONFIRMED,
            SignalLifecycleState.RISK_APPROVED,
            SignalLifecycleState.RISK_REJECTED,
            SignalLifecycleState.CANCELLED,
            SignalLifecycleState.EXPIRED,
        }
    ),
    SignalLifecycleState.PENDING_CONFIRMATION: frozenset(
        {
            SignalLifecycleState.CONFIRMED,
            SignalLifecycleState.CANCELLED,
            SignalLifecycleState.EXPIRED,
        }
    ),
    SignalLifecycleState.CONFIRMED: frozenset(
        {
            SignalLifecycleState.RISK_APPROVED,
            SignalLifecycleState.RISK_REJECTED,
            SignalLifecycleState.CANCELLED,
            SignalLifecycleState.EXPIRED,
        }
    ),
    SignalLifecycleState.RISK_APPROVED: frozenset(
        {
            SignalLifecycleState.ACTIVE,
            SignalLifecycleState.CANCELLED,
            SignalLifecycleState.EXPIRED,
        }
    ),
    SignalLifecycleState.ACTIVE: frozenset(
        {
            SignalLifecycleState.TARGET_HIT,
            SignalLifecycleState.STOP_LOSS_HIT,
            SignalLifecycleState.CANCELLED,
            SignalLifecycleState.EXPIRED,
        }
    ),
    SignalLifecycleState.RISK_REJECTED: frozenset(),
    SignalLifecycleState.CANCELLED: frozenset(),
    SignalLifecycleState.EXPIRED: frozenset(),
    SignalLifecycleState.TARGET_HIT: frozenset(),
    SignalLifecycleState.STOP_LOSS_HIT: frozenset(),
}

NON_TERMINAL_STATES: frozenset[SignalLifecycleState] = frozenset(
    {
        SignalLifecycleState.GENERATED,
        SignalLifecycleState.PENDING_CONFIRMATION,
        SignalLifecycleState.CONFIRMED,
        SignalLifecycleState.RISK_APPROVED,
        SignalLifecycleState.ACTIVE,
    }
)


class SignalLifecycleManager:
    """Tracks trading signal states, prevents duplicate active setups, and enforces risk-approval gates.

    Key Safety Invariants:
    1. A signal starts in `GENERATED` and can NEVER transition directly to `ACTIVE` without first
       passing through `RISK_APPROVED` via an explicit `RiskDecision(approved=True)`.
    2. Duplicate active signals for the same symbol, timeframe, direction, and entry zone (within
       `entry_zone_dedup_tolerance`) are rejected with `SignalLifecycleError`.
    3. Every state change appends an immutable `LifecycleEvent` to `record.history`.
    """

    def __init__(self, config: SignalLifecycleConfig | None = None) -> None:
        self._config = config or SignalLifecycleConfig()
        self._records: dict[str, ManagedSignalRecord] = {}

    @property
    def config(self) -> SignalLifecycleConfig:
        return self._config

    def get_record(self, signal_id: str) -> ManagedSignalRecord:
        """Retrieve a ManagedSignalRecord by `signal_id`."""
        if signal_id not in self._records:
            raise SignalLifecycleError(f"Unknown signal_id: '{signal_id}'.")
        return self._records[signal_id]

    def list_records(self, state: SignalLifecycleState | None = None) -> list[ManagedSignalRecord]:
        """List all tracked signal records, optionally filtered by state."""
        if state is None:
            return list(self._records.values())
        return [r for r in self._records.values() if r.state == state]

    def _is_duplicate_active_setup(self, signal: TradeSignal) -> ManagedSignalRecord | None:
        """Return an existing non-terminal record if `signal` duplicates an open setup."""
        if not signal.is_executable or signal.entry_price is None:
            return None

        tol = self._config.entry_zone_dedup_tolerance
        for existing in self._records.values():
            if existing.state not in NON_TERMINAL_STATES:
                continue
            ex_sig = existing.signal
            if (
                ex_sig.symbol == signal.symbol
                and ex_sig.timeframe == signal.timeframe
                and ex_sig.direction == signal.direction
                and ex_sig.entry_price is not None
                and abs(ex_sig.entry_price - signal.entry_price) <= tol
            ):
                return existing
        return None

    def register_signal(
        self,
        signal: TradeSignal,
        *,
        supporting_evidence: dict[str, object] | None = None,
        expiration_minutes: int | None = None,
        signal_id: str | None = None,
    ) -> ManagedSignalRecord:
        """Register a newly generated executable `TradeSignal` in `GENERATED` state."""
        if not signal.is_executable:
            raise SignalLifecycleError("WAIT signals are non-executable and cannot enter SignalLifecycleManager.")

        dup = self._is_duplicate_active_setup(signal)
        if dup is not None:
            raise SignalLifecycleError(
                f"Duplicate active signal setup detected for {signal.symbol} {signal.timeframe.value} "
                f"{signal.direction.value} near {signal.entry_price} (existing signal_id='{dup.signal_id}')."
            )

        sig_id = signal_id or f"sig-{uuid.uuid4().hex[:12]}"
        exp_mins = expiration_minutes or self._config.default_expiration_minutes
        expires_at = signal.timestamp + timedelta(minutes=exp_mins)
        setup_key = f"{signal.symbol}:{signal.timeframe.value}:{signal.direction.value}:{signal.entry_price}"

        initial_event = LifecycleEvent(
            from_state=None,
            to_state=SignalLifecycleState.GENERATED,
            timestamp=signal.timestamp,
            reason=f"Signal generated by strategy '{signal.strategy_id}'.",
        )
        record = ManagedSignalRecord(
            signal_id=sig_id,
            setup_key=setup_key,
            created_at=signal.timestamp,
            expires_at=expires_at,
            signal=signal,
            state=SignalLifecycleState.GENERATED,
            risk_status=RiskEvaluationStatus.NOT_EVALUATED,
            risk_decision=None,
            invalidation_price=signal.stop_loss,
            supporting_evidence=supporting_evidence or {},
            history=(initial_event,),
        )
        self._records[sig_id] = record
        return record

    def transition(
        self,
        signal_id: str,
        to_state: SignalLifecycleState,
        timestamp: datetime,
        reason: str,
    ) -> ManagedSignalRecord:
        """Transition a signal to a new state after validating transition rules."""
        record = self.get_record(signal_id)
        allowed = ALLOWED_STATE_TRANSITIONS.get(record.state, frozenset())
        if to_state not in allowed:
            raise SignalLifecycleError(
                f"Invalid signal state transition from {record.state.value} to {to_state.value} "
                f"(allowed: {[s.value for s in allowed]})."
            )

        if to_state == SignalLifecycleState.ACTIVE and record.risk_status != RiskEvaluationStatus.RISK_APPROVED:
            raise SignalLifecycleError(
                f"Signal '{signal_id}' cannot become ACTIVE because risk_status is {record.risk_status.value}."
            )

        new_event = LifecycleEvent(
            from_state=record.state,
            to_state=to_state,
            timestamp=timestamp,
            reason=reason,
        )
        updated = record.model_copy(
            update={
                "state": to_state,
                "history": (*record.history, new_event),
            }
        )
        self._records[signal_id] = updated
        return updated

    def apply_risk_decision(
        self,
        signal_id: str,
        decision: RiskDecision,
        timestamp: datetime,
    ) -> ManagedSignalRecord:
        """Record a `RiskDecision` from `RiskEngine` and transition to `RISK_APPROVED` or `RISK_REJECTED`."""
        record = self.get_record(signal_id)
        if record.state not in (SignalLifecycleState.GENERATED, SignalLifecycleState.CONFIRMED):
            raise SignalLifecycleError(
                f"Cannot apply risk decision to signal '{signal_id}' in state {record.state.value}."
            )

        target_state = (
            SignalLifecycleState.RISK_APPROVED if decision.approved else SignalLifecycleState.RISK_REJECTED
        )
        risk_status = (
            RiskEvaluationStatus.RISK_APPROVED if decision.approved else RiskEvaluationStatus.RISK_REJECTED
        )
        reason = (
            f"RiskEngine approved: size={decision.calculated_position_size} lots, risk={decision.calculated_risk_amount}."
            if decision.approved
            else f"RiskEngine rejected: {decision.rejection_reason}"
        )
        new_event = LifecycleEvent(
            from_state=record.state,
            to_state=target_state,
            timestamp=timestamp,
            reason=reason,
        )
        updated = record.model_copy(
            update={
                "state": target_state,
                "risk_status": risk_status,
                "risk_decision": decision,
                "history": (*record.history, new_event),
            }
        )
        self._records[signal_id] = updated
        return updated

    def evaluate_market_update(
        self,
        signal_id: str,
        *,
        current_timestamp: datetime,
        high_price: Decimal,
        low_price: Decimal,
    ) -> ManagedSignalRecord:
        """Update signal state for expiration, pre-activation invalidation, or active SL/TP hits."""
        record = self.get_record(signal_id)
        if record.state not in NON_TERMINAL_STATES:
            return record

        # 1. Check expiration for non-ACTIVE signals
        if (
            record.state != SignalLifecycleState.ACTIVE
            and record.expires_at is not None
            and current_timestamp >= record.expires_at
        ):
            return self.transition(
                signal_id,
                SignalLifecycleState.EXPIRED,
                current_timestamp,
                f"Signal expired at {record.expires_at.isoformat()}.",
            )

        sig = record.signal
        assert sig.stop_loss is not None and sig.take_profit is not None

        # 2. If ACTIVE, check stop-loss or take-profit hit (conservative: check stop-loss first)
        if record.state == SignalLifecycleState.ACTIVE:
            if sig.direction == SignalDirection.BUY:
                if low_price <= sig.stop_loss:
                    return self.transition(
                        signal_id,
                        SignalLifecycleState.STOP_LOSS_HIT,
                        current_timestamp,
                        f"BUY stop-loss ({sig.stop_loss}) hit by bar low ({low_price}).",
                    )
                if high_price >= sig.take_profit:
                    return self.transition(
                        signal_id,
                        SignalLifecycleState.TARGET_HIT,
                        current_timestamp,
                        f"BUY take-profit ({sig.take_profit}) hit by bar high ({high_price}).",
                    )
            elif sig.direction == SignalDirection.SELL:
                if high_price >= sig.stop_loss:
                    return self.transition(
                        signal_id,
                        SignalLifecycleState.STOP_LOSS_HIT,
                        current_timestamp,
                        f"SELL stop-loss ({sig.stop_loss}) hit by bar high ({high_price}).",
                    )
                if low_price <= sig.take_profit:
                    return self.transition(
                        signal_id,
                        SignalLifecycleState.TARGET_HIT,
                        current_timestamp,
                        f"SELL take-profit ({sig.take_profit}) hit by bar low ({low_price}).",
                    )
        else:
            # Pre-activation invalidation if price crosses invalidation_price (stop_loss)
            if sig.direction == SignalDirection.BUY and low_price <= sig.stop_loss:
                return self.transition(
                    signal_id,
                    SignalLifecycleState.CANCELLED,
                    current_timestamp,
                    f"Pre-entry setup invalidated: bar low ({low_price}) crossed stop-loss ({sig.stop_loss}).",
                )
            if sig.direction == SignalDirection.SELL and high_price >= sig.stop_loss:
                return self.transition(
                    signal_id,
                    SignalLifecycleState.CANCELLED,
                    current_timestamp,
                    f"Pre-entry setup invalidated: bar high ({high_price}) crossed stop-loss ({sig.stop_loss}).",
                )

        return record
