"""Crash-Safe Write-Ahead Order Execution Ledger & Broker Reconciliation for ATLAS AI Phase 7.4."""

from datetime import datetime, timedelta, timezone
from decimal import Decimal
import threading
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from atlas_ai.core.logger import get_logger
from atlas_ai.domain.market import Timeframe
from atlas_ai.domain.signals import SignalDirection
from atlas_ai.execution.config import ExecutionSafetyConfig
from atlas_ai.execution.connection import MT5ConnectionAdapter
from atlas_ai.execution.exceptions import (
    AmbiguousExecutionStateError,
    DuplicateOrderError,
    LedgerRecoveryLockError,
    OrderExecutionError,
    PersistenceAccountMismatchError,
    PersistenceCorruptionError,
    PersistenceRollbackDetectedError,
)
from atlas_ai.execution.models import (
    ExecutionLedgerEntry,
    LedgerHealthStatus,
    LedgerTransitionRecord,
    MT5DealEntry,
    MT5DealRecord,
    MT5DealType,
    MT5OrderFillingMode,
    MT5OrderRecord,
    MT5PositionRecord,
    OrderLedgerStatus,
    ReconciliationStatus,
)
from atlas_ai.execution.persistence import CrashSafeJsonPersistence

logger = get_logger("execution.ledger")

EPOCH_UTC = datetime(1970, 1, 1, 0, 0, 0, tzinfo=timezone.utc)

UNRESOLVED_LEDGER_STATUSES: frozenset[OrderLedgerStatus] = frozenset(
    {
        OrderLedgerStatus.SUBMITTING,
        OrderLedgerStatus.SUBMITTED,
        OrderLedgerStatus.UNKNOWN,
        OrderLedgerStatus.RECONCILIATION_REQUIRED,
        OrderLedgerStatus.UNKNOWN_PENDING_RECONCILIATION,
    }
)

PRE_SUBMISSION_LEDGER_STATUSES: frozenset[OrderLedgerStatus] = frozenset(
    {
        OrderLedgerStatus.CREATED,
        OrderLedgerStatus.VALIDATED,
    }
)

ACTIVE_OR_EXECUTED_LEDGER_STATUSES: frozenset[OrderLedgerStatus] = frozenset(
    {
        OrderLedgerStatus.ACTIVE,
        OrderLedgerStatus.PARTIALLY_FILLED,
        OrderLedgerStatus.FILLED,
    }
)

TERMINAL_NON_EXECUTED_LEDGER_STATUSES: frozenset[OrderLedgerStatus] = frozenset(
    {
        OrderLedgerStatus.REJECTED,
        OrderLedgerStatus.CANCELLED,
        OrderLedgerStatus.FAILED,
    }
)

ALLOWED_LEDGER_TRANSITIONS: dict[OrderLedgerStatus, frozenset[OrderLedgerStatus]] = {
    OrderLedgerStatus.CREATED: frozenset(
        {
            OrderLedgerStatus.VALIDATED,
            OrderLedgerStatus.REJECTED,
            OrderLedgerStatus.CANCELLED,
            OrderLedgerStatus.FAILED,
        }
    ),
    OrderLedgerStatus.VALIDATED: frozenset(
        {
            OrderLedgerStatus.SUBMITTING,
            OrderLedgerStatus.REJECTED,
            OrderLedgerStatus.CANCELLED,
            OrderLedgerStatus.FAILED,
        }
    ),
    OrderLedgerStatus.SUBMITTING: frozenset(
        {
            OrderLedgerStatus.SUBMITTED,
            OrderLedgerStatus.ACTIVE,
            OrderLedgerStatus.PARTIALLY_FILLED,
            OrderLedgerStatus.FILLED,
            OrderLedgerStatus.REJECTED,
            OrderLedgerStatus.CANCELLED,
            OrderLedgerStatus.UNKNOWN,
            OrderLedgerStatus.RECONCILIATION_REQUIRED,
            OrderLedgerStatus.UNKNOWN_PENDING_RECONCILIATION,
            OrderLedgerStatus.FAILED,
        }
    ),
    OrderLedgerStatus.SUBMITTED: frozenset(
        {
            OrderLedgerStatus.ACTIVE,
            OrderLedgerStatus.PARTIALLY_FILLED,
            OrderLedgerStatus.FILLED,
            OrderLedgerStatus.REJECTED,
            OrderLedgerStatus.CANCELLED,
            OrderLedgerStatus.UNKNOWN,
            OrderLedgerStatus.RECONCILIATION_REQUIRED,
            OrderLedgerStatus.UNKNOWN_PENDING_RECONCILIATION,
        }
    ),
    OrderLedgerStatus.UNKNOWN: frozenset(
        {
            OrderLedgerStatus.RECONCILIATION_REQUIRED,
            OrderLedgerStatus.UNKNOWN_PENDING_RECONCILIATION,
            OrderLedgerStatus.ACTIVE,
            OrderLedgerStatus.PARTIALLY_FILLED,
            OrderLedgerStatus.FILLED,
            OrderLedgerStatus.REJECTED,
            OrderLedgerStatus.CANCELLED,
        }
    ),
    OrderLedgerStatus.RECONCILIATION_REQUIRED: frozenset(
        {
            OrderLedgerStatus.ACTIVE,
            OrderLedgerStatus.PARTIALLY_FILLED,
            OrderLedgerStatus.FILLED,
            OrderLedgerStatus.REJECTED,
            OrderLedgerStatus.CANCELLED,
        }
    ),
    OrderLedgerStatus.UNKNOWN_PENDING_RECONCILIATION: frozenset(
        {
            OrderLedgerStatus.RECONCILIATION_REQUIRED,
            OrderLedgerStatus.ACTIVE,
            OrderLedgerStatus.PARTIALLY_FILLED,
            OrderLedgerStatus.FILLED,
            OrderLedgerStatus.REJECTED,
            OrderLedgerStatus.CANCELLED,
        }
    ),
    OrderLedgerStatus.PARTIALLY_FILLED: frozenset(
        {
            OrderLedgerStatus.ACTIVE,
            OrderLedgerStatus.FILLED,
            OrderLedgerStatus.CANCELLED,
            OrderLedgerStatus.RECONCILIATION_REQUIRED,
            OrderLedgerStatus.UNKNOWN_PENDING_RECONCILIATION,
        }
    ),
    OrderLedgerStatus.FILLED: frozenset(
        {
            OrderLedgerStatus.ACTIVE,
            OrderLedgerStatus.RECONCILIATION_REQUIRED,
        }
    ),
    OrderLedgerStatus.ACTIVE: frozenset(
        {
            OrderLedgerStatus.FILLED,
            OrderLedgerStatus.CANCELLED,
            OrderLedgerStatus.RECONCILIATION_REQUIRED,
        }
    ),
    OrderLedgerStatus.REJECTED: frozenset(),
    OrderLedgerStatus.CANCELLED: frozenset(),
    OrderLedgerStatus.FAILED: frozenset(),
}


def extract_comment_identifiers(comment: str) -> frozenset[str]:
    """Extract exact order/signal identifier tokens from an MT5 broker comment without substring matching.

    Strips optional trailing bracketed broker annotations (e.g., 'ord-1[sl 2640.20]' -> 'ord-1')
    and splits only on whitespace, '|', or ':' delimiter boundaries.
    Never matches prefixes such as 'ord-1' against 'ord-10'.
    """
    cleaned = comment.strip()
    if not cleaned:
        return frozenset()
    tokens: set[str] = {cleaned}
    bracket_idx = cleaned.find("[")
    if bracket_idx > 0:
        base = cleaned[:bracket_idx].strip()
        if base:
            tokens.add(base)
    for sep in ("|", ":", " "):
        if sep in cleaned:
            for part in cleaned.split(sep):
                stripped = part.strip()
                if stripped:
                    b_idx = stripped.find("[")
                    if b_idx > 0:
                        stripped = stripped[:b_idx].strip()
                    if stripped:
                        tokens.add(stripped)
    return frozenset(tokens)


def extract_primary_comment_id(comment: str) -> str:
    """Return the primary order/signal identifier from a broker comment (stripping any '[sl/tp]' suffix)."""
    cleaned = comment.strip()
    if not cleaned:
        return ""
    bracket_idx = cleaned.find("[")
    if bracket_idx > 0:
        cleaned = cleaned[:bracket_idx].strip()
    return cleaned


class LedgerReconciliationReport(BaseModel):
    """Structured report of reconciling the execution ledger against broker positions, orders, and deals."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    health_status: LedgerHealthStatus
    reconciled_executed_ids: tuple[str, ...] = Field(default=())
    reconciled_not_executed_ids: tuple[str, ...] = Field(default=())
    still_working_ids: tuple[str, ...] = Field(default=())
    discovered_orphan_ids: tuple[str, ...] = Field(default=())
    envelope_generation: int = Field(..., ge=1)


class PersistentExecutionLedger:
    """Durable write-ahead order execution ledger with idempotency, state transition validation,
    restart recovery locking, and broker-side reconciliation.

    Safety Invariants:
    1. Every order intent is persisted to disk (`CREATED` -> `VALIDATED` -> `SUBMITTING`) before
       `order_send()` is invoked.
    2. An unresolved intent (`SUBMITTING`, `SUBMITTED`, `UNKNOWN`, `RECONCILIATION_REQUIRED`,
       `UNKNOWN_PENDING_RECONCILIATION`) blocks all new order submissions (`AmbiguousExecutionStateError`)
       and can never be dismissed to `REJECTED`/`CANCELLED` without explicit broker reconciliation proof.
    3. A missing or corrupted primary ledger file or interrupted `.tmp.*` write places the ledger in
       `LedgerHealthStatus.RECOVERY_REQUIRED_LOCKED`, blocking all new orders (`LedgerRecoveryLockError`)
       until `reconcile_with_broker()` verifies against live broker positions, working orders, and deals.
    """

    def __init__(
        self,
        *,
        safety_config: ExecutionSafetyConfig | None = None,
        persistence: CrashSafeJsonPersistence | None = None,
    ) -> None:
        self._safety_config = safety_config or ExecutionSafetyConfig()
        self._persistence = persistence or CrashSafeJsonPersistence(
            self._safety_config.ledger_path,
            lock_timeout_seconds=self._safety_config.lock_timeout_seconds,
        )
        self._mutex = threading.RLock()
        self._entries: dict[str, ExecutionLedgerEntry] = {}
        self._health_status: LedgerHealthStatus = LedgerHealthStatus.HEALTHY
        self._loaded_account: tuple[int, str] | None = None
        self._unrecoverable_corruption: bool = False
        self._corruption_reason: str | None = None
        self._last_generation: int = 0

    @property
    def safety_config(self) -> ExecutionSafetyConfig:
        return self._safety_config

    @property
    def persistence(self) -> CrashSafeJsonPersistence:
        return self._persistence

    @property
    def health_status(self) -> LedgerHealthStatus:
        return self._health_status

    @property
    def last_generation(self) -> int:
        return self._last_generation

    def get_entry(self, client_order_id: str) -> ExecutionLedgerEntry | None:
        with self._mutex:
            return self._entries.get(client_order_id)

    def list_entries(self, status: OrderLedgerStatus | None = None) -> tuple[ExecutionLedgerEntry, ...]:
        with self._mutex:
            all_entries = tuple(self._entries.values())
            if status is None:
                return all_entries
            return tuple(e for e in all_entries if e.status == status)

    def get_unresolved_entries(self) -> tuple[ExecutionLedgerEntry, ...]:
        """Return all entries currently in an unresolved or in-flight submission state."""
        with self._mutex:
            return tuple(
                e for e in self._entries.values() if e.status in UNRESOLVED_LEDGER_STATUSES
            )

    def has_unresolved_entries(self) -> bool:
        with self._mutex:
            return any(e.status in UNRESOLVED_LEDGER_STATUSES for e in self._entries.values())

    def _serialize_payload(self) -> dict[str, Any]:
        return {
            "entries": [
                entry.model_dump(mode="json")
                for entry in sorted(
                    self._entries.values(),
                    key=lambda item: (item.created_at_utc, item.client_order_id),
                )
            ]
        }

    @staticmethod
    def _parse_entries_from_payload(payload: dict[str, Any]) -> dict[str, ExecutionLedgerEntry]:
        raw_list = payload.get("entries")
        if not isinstance(raw_list, list):
            raise PersistenceCorruptionError("Execution ledger payload must contain an 'entries' list.")
        parsed: dict[str, ExecutionLedgerEntry] = {}
        for idx, item in enumerate(raw_list):
            if not isinstance(item, dict):
                raise PersistenceCorruptionError(f"Ledger entry at index {idx} must be a JSON object.")
            try:
                entry = ExecutionLedgerEntry.model_validate(item)
            except ValidationError as exc:
                raise PersistenceCorruptionError(
                    f"Ledger entry at index {idx} failed schema validation: {exc}"
                ) from exc
            if entry.client_order_id in parsed:
                raise PersistenceCorruptionError(
                    f"Duplicate client_order_id '{entry.client_order_id}' in persisted execution ledger."
                )
            parsed[entry.client_order_id] = entry
        return parsed

    def load_or_initialize(
        self,
        *,
        account_login: int,
        server: str,
        reference_now_utc: datetime | None = None,
    ) -> LedgerHealthStatus:
        """Load the execution ledger from disk or lock it in `RECOVERY_REQUIRED_LOCKED` if
        interrupted writes, missing primary file, or corruption are detected.
        """
        now_utc = (
            reference_now_utc.astimezone(timezone.utc)
            if (reference_now_utc is not None and reference_now_utc.tzinfo is not None)
            else (
                reference_now_utc.replace(tzinfo=timezone.utc)
                if reference_now_utc is not None
                else datetime.now(timezone.utc)
            )
        )
        cleaned_server = server.strip()

        with self._mutex:
            try:
                inspection = self._persistence.inspect_recovery_state(
                    expected_login=account_login,
                    expected_server=cleaned_server,
                )
            except (PersistenceAccountMismatchError, PersistenceRollbackDetectedError):
                self._health_status = LedgerHealthStatus.RECOVERY_REQUIRED_LOCKED
                raise
            except PersistenceCorruptionError as exc:
                self._health_status = LedgerHealthStatus.RECOVERY_REQUIRED_LOCKED
                self._unrecoverable_corruption = True
                self._corruption_reason = str(exc)
                raise

            self._loaded_account = (account_login, cleaned_server)

            if inspection.recovery_required:
                self._health_status = LedgerHealthStatus.RECOVERY_REQUIRED_LOCKED
                if inspection.highest_valid_envelope is None:
                    self._unrecoverable_corruption = True
                    self._corruption_reason = (
                        inspection.primary_error
                        or "All primary, backup, and temporary ledger files are corrupted."
                    )
                    return self._health_status

                try:
                    baseline_entries = self._parse_entries_from_payload(
                        inspection.highest_valid_envelope.payload
                    )
                except PersistenceCorruptionError as exc:
                    self._unrecoverable_corruption = True
                    self._corruption_reason = str(exc)
                    return self._health_status

                self._unrecoverable_corruption = False
                self._corruption_reason = inspection.primary_error
                self._last_generation = inspection.highest_valid_envelope.generation

                # Promote any in-flight, pre-submission, or unactivated FILLED entries in the baseline to RECONCILIATION_REQUIRED
                recovered_map: dict[str, ExecutionLedgerEntry] = {}
                for cid, entry in baseline_entries.items():
                    visited_active = any(
                        h.to_status == OrderLedgerStatus.ACTIVE for h in entry.history
                    )
                    is_unactivated_fill = (
                        entry.status == OrderLedgerStatus.FILLED
                        and not visited_active
                        and "closed" not in entry.status_reason.lower()
                        and "orphan" not in entry.status_reason.lower()
                    )
                    if (
                        entry.status in (UNRESOLVED_LEDGER_STATUSES | PRE_SUBMISSION_LEDGER_STATUSES)
                        or is_unactivated_fill
                    ):
                        ev = LedgerTransitionRecord(
                            from_status=entry.status,
                            to_status=OrderLedgerStatus.RECONCILIATION_REQUIRED,
                            timestamp_utc=now_utc,
                            reason="Persistence recovery lock: promoting in-flight entry to RECONCILIATION_REQUIRED.",
                            retcode=entry.retcode,
                        )
                        recovered_map[cid] = entry.model_copy(
                            update={
                                "status": OrderLedgerStatus.RECONCILIATION_REQUIRED,
                                "reconciliation_status": ReconciliationStatus.PENDING,
                                "updated_at_utc": now_utc,
                                "status_reason": ev.reason,
                                "history": (*entry.history, ev),
                            }
                        )
                    else:
                        recovered_map[cid] = entry
                self._entries = recovered_map
                return self._health_status

            # Clean persistence state (no interrupted .tmp.*, no missing/corrupted primary)
            self._unrecoverable_corruption = False
            self._corruption_reason = None

            if inspection.envelope is None:
                self._entries = {}
                self._health_status = LedgerHealthStatus.HEALTHY
                self._last_generation = 0
                return self._health_status

            loaded_entries = self._parse_entries_from_payload(inspection.envelope.payload)
            self._last_generation = inspection.envelope.generation

            # Check for any entry left in CREATED/VALIDATED, SUBMITTING/SUBMITTED, or unactivated FILLED across a restart
            needs_save = False
            normalized_entries: dict[str, ExecutionLedgerEntry] = {}
            for cid, entry in loaded_entries.items():
                visited_active = any(
                    h.to_status == OrderLedgerStatus.ACTIVE for h in entry.history
                )
                is_unactivated_fill = (
                    entry.status == OrderLedgerStatus.FILLED
                    and not visited_active
                    and "closed" not in entry.status_reason.lower()
                    and "orphan" not in entry.status_reason.lower()
                )
                if (
                    entry.status in (OrderLedgerStatus.SUBMITTING, OrderLedgerStatus.SUBMITTED)
                    or is_unactivated_fill
                ):
                    ev = LedgerTransitionRecord(
                        from_status=entry.status,
                        to_status=OrderLedgerStatus.RECONCILIATION_REQUIRED,
                        timestamp_utc=now_utc,
                        reason=(
                            f"Application restarted while order '{cid}' was in {entry.status.value}; "
                            "broker reconciliation required before any new submissions."
                        ),
                        retcode=entry.retcode,
                    )
                    normalized_entries[cid] = entry.model_copy(
                        update={
                            "status": OrderLedgerStatus.RECONCILIATION_REQUIRED,
                            "reconciliation_status": ReconciliationStatus.PENDING,
                            "updated_at_utc": now_utc,
                            "status_reason": ev.reason,
                            "history": (*entry.history, ev),
                        }
                    )
                    needs_save = True
                elif entry.status in PRE_SUBMISSION_LEDGER_STATUSES:
                    ev = LedgerTransitionRecord(
                        from_status=entry.status,
                        to_status=OrderLedgerStatus.CANCELLED,
                        timestamp_utc=now_utc,
                        reason=(
                            f"Pre-submission intent '{cid}' in state {entry.status.value} never reached "
                            "SUBMITTING before restart; cancelled safely."
                        ),
                    )
                    normalized_entries[cid] = entry.model_copy(
                        update={
                            "status": OrderLedgerStatus.CANCELLED,
                            "reconciliation_status": ReconciliationStatus.NOT_REQUIRED,
                            "updated_at_utc": now_utc,
                            "status_reason": ev.reason,
                            "history": (*entry.history, ev),
                        }
                    )
                    needs_save = True
                else:
                    normalized_entries[cid] = entry

            self._entries = normalized_entries
            self._health_status = LedgerHealthStatus.HEALTHY

            if needs_save:
                saved_env = self._persistence.save(
                    self._serialize_payload(),
                    account_login=account_login,
                    server=cleaned_server,
                    now_utc=now_utc,
                )
                self._last_generation = saved_env.generation

            return self._health_status

    def ensure_ready_for_new_submission(
        self,
        *,
        account_login: int,
        server: str,
        client_order_id: str | None = None,
        signal_id: str | None = None,
        symbol: str | None = None,
        direction: SignalDirection | None = None,
    ) -> None:
        """Verify that the ledger is healthy, has no unresolved submissions, and has no duplicate
        or conflicting active orders for `client_order_id` or `signal_id`.
        """
        with self._mutex:
            if self._loaded_account != (account_login, server.strip()):
                self.load_or_initialize(account_login=account_login, server=server)

            if self._unrecoverable_corruption:
                raise PersistenceCorruptionError(
                    f"Execution ledger is corrupted and cannot be recovered: {self._corruption_reason}"
                )

            if self._health_status == LedgerHealthStatus.RECOVERY_REQUIRED_LOCKED:
                raise LedgerRecoveryLockError(
                    "Execution ledger is in RECOVERY_REQUIRED_LOCKED state due to an interrupted write "
                    "or missing/corrupted primary ledger file. Broker reconciliation is required before "
                    "any new order submission."
                )

            unresolved = self.get_unresolved_entries()
            if unresolved:
                unresolved_desc = ", ".join(
                    f"{e.client_order_id}({e.status.value})" for e in unresolved
                )
                raise AmbiguousExecutionStateError(
                    f"New order submissions are blocked while {len(unresolved)} execution ledger intent(s) "
                    f"remain unresolved: [{unresolved_desc}]. Broker reconciliation is required."
                )

            if client_order_id is not None and client_order_id in self._entries:
                existing = self._entries[client_order_id]
                raise DuplicateOrderError(
                    f"Duplicate client_order_id '{client_order_id}' already exists in execution ledger "
                    f"(status={existing.status.value}, signal_id='{existing.signal_id}')."
                )

            if signal_id is not None:
                for entry in self._entries.values():
                    if entry.signal_id == signal_id and entry.status in (
                        UNRESOLVED_LEDGER_STATUSES
                        | PRE_SUBMISSION_LEDGER_STATUSES
                        | ACTIVE_OR_EXECUTED_LEDGER_STATUSES
                    ):
                        raise DuplicateOrderError(
                            f"Duplicate order submission blocked for signal_id='{signal_id}': "
                            f"existing ledger entry '{entry.client_order_id}' is in status {entry.status.value}."
                        )

            if symbol is not None and direction is not None:
                cleaned_sym = symbol.strip().upper()
                for entry in self._entries.values():
                    if (
                        entry.symbol == cleaned_sym
                        and entry.status
                        in (
                            OrderLedgerStatus.ACTIVE,
                            OrderLedgerStatus.PARTIALLY_FILLED,
                        )
                        and entry.direction != direction
                    ):
                        raise DuplicateOrderError(
                            f"Conflicting active/partially-filled order in execution ledger for symbol '{cleaned_sym}': "
                            f"existing entry '{entry.client_order_id}' is {entry.status.value} in opposite direction "
                            f"({entry.direction.value} vs {direction.value})."
                        )

    def create_intent(
        self,
        *,
        account_login: int,
        server: str,
        client_order_id: str,
        signal_id: str,
        symbol: str,
        timeframe: Timeframe,
        direction: SignalDirection,
        magic: int,
        requested_volume: Decimal,
        requested_price: Decimal,
        stop_loss: Decimal,
        take_profit: Decimal,
        filling_mode: MT5OrderFillingMode,
        now_utc: datetime | None = None,
        reason: str = "Order intent created.",
    ) -> ExecutionLedgerEntry:
        """Create and durably persist a new order intent in `OrderLedgerStatus.CREATED`."""
        ts = (
            now_utc.astimezone(timezone.utc)
            if (now_utc is not None and now_utc.tzinfo is not None)
            else (
                now_utc.replace(tzinfo=timezone.utc)
                if now_utc is not None
                else datetime.now(timezone.utc)
            )
        )
        with self._mutex:
            self.ensure_ready_for_new_submission(
                account_login=account_login,
                server=server,
                client_order_id=client_order_id,
                signal_id=signal_id,
                symbol=symbol,
                direction=direction,
            )

            init_event = LedgerTransitionRecord(
                from_status=None,
                to_status=OrderLedgerStatus.CREATED,
                timestamp_utc=ts,
                reason=reason,
            )
            entry = ExecutionLedgerEntry(
                client_order_id=client_order_id,
                signal_id=signal_id,
                symbol=symbol.strip().upper(),
                timeframe=timeframe,
                direction=direction,
                magic=magic,
                requested_volume=requested_volume,
                executed_volume=Decimal("0"),
                remaining_volume=requested_volume,
                requested_price=requested_price,
                fill_price=None,
                stop_loss=stop_loss,
                take_profit=take_profit,
                filling_mode=filling_mode,
                status=OrderLedgerStatus.CREATED,
                reconciliation_status=ReconciliationStatus.NOT_REQUIRED,
                created_at_utc=ts,
                updated_at_utc=ts,
                status_reason=reason,
                attempt_count=1,
                history=(init_event,),
            )
            self._entries[client_order_id] = entry
            saved_env = self._persistence.save(
                self._serialize_payload(),
                account_login=account_login,
                server=server,
                now_utc=ts,
            )
            self._last_generation = saved_env.generation
            return entry

    def transition_entry(
        self,
        client_order_id: str,
        to_status: OrderLedgerStatus,
        *,
        account_login: int,
        server: str,
        reason: str,
        now_utc: datetime | None = None,
        executed_volume: Decimal | None = None,
        remaining_volume: Decimal | None = None,
        fill_price: Decimal | None = None,
        order_ticket: int | None = None,
        deal_ticket: int | None = None,
        position_ticket: int | None = None,
        retcode: int | None = None,
        reconciliation_status: ReconciliationStatus | None = None,
        reconciliation_verified: bool = False,
        synchronous_broker_rejection: bool = False,
    ) -> ExecutionLedgerEntry:
        """Atomically transition an existing ledger entry to `to_status` and persist to disk.

        Enforces:
        - `ALLOWED_LEDGER_TRANSITIONS` state machine.
        - An unresolved intent (`UNKNOWN`, `RECONCILIATION_REQUIRED`, `UNKNOWN_PENDING_RECONCILIATION`,
          or post-timeout `SUBMITTING`/`SUBMITTED`) cannot be transitioned to `REJECTED`, `CANCELLED`,
          or `FAILED` unless `reconciliation_verified=True` or `synchronous_broker_rejection=True` (from `SUBMITTING`).
        """
        ts = (
            now_utc.astimezone(timezone.utc)
            if (now_utc is not None and now_utc.tzinfo is not None)
            else (
                now_utc.replace(tzinfo=timezone.utc)
                if now_utc is not None
                else datetime.now(timezone.utc)
            )
        )
        with self._mutex:
            if self._health_status == LedgerHealthStatus.RECOVERY_REQUIRED_LOCKED:
                raise LedgerRecoveryLockError(
                    "Cannot transition single ledger entry while ledger is in RECOVERY_REQUIRED_LOCKED state."
                )

            if client_order_id not in self._entries:
                raise OrderExecutionError(
                    f"Unknown client_order_id '{client_order_id}' in execution ledger."
                )

            current = self._entries[client_order_id]
            allowed = ALLOWED_LEDGER_TRANSITIONS.get(current.status, frozenset())
            if to_status not in allowed:
                raise OrderExecutionError(
                    f"Invalid execution ledger state transition for '{client_order_id}': "
                    f"{current.status.value} -> {to_status.value} (allowed: {[s.value for s in allowed]})."
                )

            # Block silent dismissal of ambiguous/unresolved states without broker reconciliation proof
            if current.status in (
                OrderLedgerStatus.UNKNOWN,
                OrderLedgerStatus.RECONCILIATION_REQUIRED,
                OrderLedgerStatus.UNKNOWN_PENDING_RECONCILIATION,
                OrderLedgerStatus.SUBMITTED,
            ) and to_status in TERMINAL_NON_EXECUTED_LEDGER_STATUSES:
                if not reconciliation_verified:
                    raise AmbiguousExecutionStateError(
                        f"Cannot transition unresolved intent '{client_order_id}' from {current.status.value} "
                        f"to {to_status.value} without verified broker reconciliation proof."
                    )

            if (
                current.status == OrderLedgerStatus.SUBMITTING
                and to_status in TERMINAL_NON_EXECUTED_LEDGER_STATUSES
                and not (synchronous_broker_rejection or reconciliation_verified)
            ):
                raise AmbiguousExecutionStateError(
                    f"Cannot mark SUBMITTING intent '{client_order_id}' as {to_status.value} "
                    "without either a synchronous broker rejection retcode or broker reconciliation proof."
                )

            if to_status in (
                OrderLedgerStatus.ACTIVE,
                OrderLedgerStatus.PARTIALLY_FILLED,
                OrderLedgerStatus.FILLED,
            ):
                eff_exec_vol = (
                    executed_volume if executed_volume is not None else current.executed_volume
                )
                if eff_exec_vol <= Decimal("0"):
                    raise OrderExecutionError(
                        f"Cannot transition '{client_order_id}' to {to_status.value} with executed_volume={eff_exec_vol} <= 0."
                    )

            next_recon_status = (
                reconciliation_status
                if reconciliation_status is not None
                else (
                    ReconciliationStatus.PENDING
                    if to_status in UNRESOLVED_LEDGER_STATUSES
                    else current.reconciliation_status
                )
            )

            ev = LedgerTransitionRecord(
                from_status=current.status,
                to_status=to_status,
                timestamp_utc=ts,
                reason=reason,
                retcode=retcode if retcode is not None else current.retcode,
            )
            updated = current.model_copy(
                update={
                    "status": to_status,
                    "reconciliation_status": next_recon_status,
                    "executed_volume": (
                        executed_volume if executed_volume is not None else current.executed_volume
                    ),
                    "remaining_volume": (
                        remaining_volume
                        if remaining_volume is not None
                        else current.remaining_volume
                    ),
                    "fill_price": fill_price if fill_price is not None else current.fill_price,
                    "order_ticket": (
                        order_ticket if order_ticket is not None else current.order_ticket
                    ),
                    "deal_ticket": deal_ticket if deal_ticket is not None else current.deal_ticket,
                    "position_ticket": (
                        position_ticket
                        if position_ticket is not None
                        else current.position_ticket
                    ),
                    "retcode": retcode if retcode is not None else current.retcode,
                    "updated_at_utc": ts,
                    "status_reason": reason,
                    "history": (*current.history, ev),
                }
            )
            self._entries[client_order_id] = updated
            saved_env = self._persistence.save(
                self._serialize_payload(),
                account_login=account_login,
                server=server,
                now_utc=ts,
            )
            self._last_generation = saved_env.generation
            return updated

    def transition_filled_to_active_atomic(
        self,
        client_order_id: str,
        *,
        account_login: int,
        server: str,
        now_utc: datetime | None = None,
        executed_volume: Decimal,
        remaining_volume: Decimal = Decimal("0"),
        fill_price: Decimal,
        order_ticket: int | None = None,
        deal_ticket: int | None = None,
        position_ticket: int | None = None,
        retcode: int | None = None,
        filled_reason: str,
        active_reason: str,
    ) -> ExecutionLedgerEntry:
        """Atomically record both `FILLED` and `ACTIVE` transitions in a single disk write,
        eliminating any two-write crash window while preserving complete transition history.
        """
        ts = (
            now_utc.astimezone(timezone.utc)
            if (now_utc is not None and now_utc.tzinfo is not None)
            else (
                now_utc.replace(tzinfo=timezone.utc)
                if now_utc is not None
                else datetime.now(timezone.utc)
            )
        )
        with self._mutex:
            if self._health_status == LedgerHealthStatus.RECOVERY_REQUIRED_LOCKED:
                raise LedgerRecoveryLockError(
                    "Cannot transition single ledger entry while ledger is in RECOVERY_REQUIRED_LOCKED state."
                )
            if client_order_id not in self._entries:
                raise OrderExecutionError(
                    f"Unknown client_order_id '{client_order_id}' in execution ledger."
                )
            current = self._entries[client_order_id]
            allowed_to_filled = ALLOWED_LEDGER_TRANSITIONS.get(current.status, frozenset())
            if OrderLedgerStatus.FILLED not in allowed_to_filled:
                raise OrderExecutionError(
                    f"Invalid execution ledger state transition for '{client_order_id}': "
                    f"{current.status.value} -> FILLED."
                )
            if executed_volume <= Decimal("0"):
                raise OrderExecutionError(
                    f"Cannot transition '{client_order_id}' to FILLED/ACTIVE with executed_volume={executed_volume} <= 0."
                )

            eff_retcode = retcode if retcode is not None else current.retcode
            ev_filled = LedgerTransitionRecord(
                from_status=current.status,
                to_status=OrderLedgerStatus.FILLED,
                timestamp_utc=ts,
                reason=filled_reason,
                retcode=eff_retcode,
            )
            ev_active = LedgerTransitionRecord(
                from_status=OrderLedgerStatus.FILLED,
                to_status=OrderLedgerStatus.ACTIVE,
                timestamp_utc=ts,
                reason=active_reason,
                retcode=eff_retcode,
            )
            updated = current.model_copy(
                update={
                    "status": OrderLedgerStatus.ACTIVE,
                    "reconciliation_status": ReconciliationStatus.RECONCILED_EXECUTED,
                    "executed_volume": executed_volume,
                    "remaining_volume": remaining_volume,
                    "fill_price": fill_price,
                    "order_ticket": (
                        order_ticket if order_ticket is not None else current.order_ticket
                    ),
                    "deal_ticket": deal_ticket if deal_ticket is not None else current.deal_ticket,
                    "position_ticket": (
                        position_ticket
                        if position_ticket is not None
                        else current.position_ticket
                    ),
                    "retcode": eff_retcode,
                    "updated_at_utc": ts,
                    "status_reason": active_reason,
                    "history": (*current.history, ev_filled, ev_active),
                }
            )
            self._entries[client_order_id] = updated
            saved_env = self._persistence.save(
                self._serialize_payload(),
                account_login=account_login,
                server=server,
                now_utc=ts,
            )
            self._last_generation = saved_env.generation
            return updated

    @staticmethod
    def _matches_entry_on_broker(
        entry: ExecutionLedgerEntry,
        *,
        comment: str,
        magic: int,
        symbol: str = "",
        direction: SignalDirection | None = None,
        order_ticket: int = 0,
        deal_ticket: int = 0,
        position_ticket: int = 0,
        other_entry_ids: frozenset[str] = frozenset(),
    ) -> bool:
        if magic != entry.magic:
            return False
        if symbol and symbol.strip().upper() != entry.symbol:
            return False
        if direction is not None and direction != entry.direction:
            return False

        comment_tokens = extract_comment_identifiers(comment)
        # If the comment explicitly names another distinct ledger entry (and not this entry), never cross-match
        if (
            comment_tokens
            and entry.client_order_id not in comment_tokens
            and entry.signal_id not in comment_tokens
            and (comment_tokens & other_entry_ids)
        ):
            return False

        if entry.order_ticket is not None and entry.order_ticket > 0 and order_ticket == entry.order_ticket:
            return True
        if entry.deal_ticket is not None and entry.deal_ticket > 0 and deal_ticket == entry.deal_ticket:
            return True
        if (
            entry.position_ticket is not None
            and entry.position_ticket > 0
            and position_ticket == entry.position_ticket
        ):
            return True

        if not comment_tokens:
            return False
        return (
            entry.client_order_id in comment_tokens
            or entry.signal_id in comment_tokens
        )

    def reconcile_with_broker(
        self,
        connection: MT5ConnectionAdapter,
        *,
        reference_now_utc: datetime | None = None,
    ) -> LedgerReconciliationReport:
        """Reconcile all unresolved or recovery-locked ledger entries against live broker
        positions, active orders, and historical execution deals.

        Fails closed (`LedgerRecoveryLockError` / `AmbiguousExecutionStateError` / `PersistenceCorruptionError`)
        if broker queries return `None` or if existing persistence files are unrecoverably corrupted.
        """
        account = connection.ensure_connected()
        now_utc = (
            reference_now_utc.astimezone(timezone.utc)
            if (reference_now_utc is not None and reference_now_utc.tzinfo is not None)
            else (
                reference_now_utc.replace(tzinfo=timezone.utc)
                if reference_now_utc is not None
                else datetime.now(timezone.utc)
            )
        )

        with self._mutex:
            if self._loaded_account != (account.login, account.server.strip()):
                self.load_or_initialize(
                    account_login=account.login,
                    server=account.server,
                    reference_now_utc=now_utc,
                )

            if self._unrecoverable_corruption:
                raise PersistenceCorruptionError(
                    f"Cannot reconcile execution ledger: {self._corruption_reason}"
                )

            positions = connection.backend.positions_get(None)
            orders = connection.backend.orders_get(None)
            future_horizon_utc = max(
                now_utc + timedelta(days=3650),
                datetime(2100, 1, 1, 0, 0, 0, tzinfo=timezone.utc),
            )
            deals = connection.backend.history_deals_get(EPOCH_UTC, future_horizon_utc)

            if positions is None or orders is None or deals is None:
                raise LedgerRecoveryLockError(
                    "Broker reconciliation failed closed: positions_get(), orders_get(), or "
                    "history_deals_get() returned None. Unresolved intents remain locked."
                )

            has_unresolved_or_locked = (
                self._health_status == LedgerHealthStatus.RECOVERY_REQUIRED_LOCKED
                or any(
                    e.status in (UNRESOLVED_LEDGER_STATUSES | PRE_SUBMISSION_LEDGER_STATUSES)
                    for e in self._entries.values()
                )
            )
            if has_unresolved_or_locked and len(deals) == 0:
                raise LedgerRecoveryLockError(
                    "Broker reconciliation failed closed: history_deals_get() returned empty deal history "
                    "(len=0). An unloaded or truncated MT5 history cache cannot prove that unresolved "
                    "intents did not execute."
                )

            exec_deals = [
                d
                for d in deals
                if d.deal_type in (int(MT5DealType.DEAL_TYPE_BUY), int(MT5DealType.DEAL_TYPE_SELL))
                and d.entry == int(MT5DealEntry.DEAL_ENTRY_IN)
            ]

            reconciled_executed: list[str] = []
            reconciled_not_executed: list[str] = []
            still_working: list[str] = []
            matched_pos_tickets: set[int] = set()
            matched_deal_tickets: set[int] = set()
            matched_order_tickets: set[int] = set()

            all_known_ids: frozenset[str] = frozenset(
                token
                for e in self._entries.values()
                for token in (e.client_order_id, e.signal_id)
            )

            # Pre-index already-known tickets from resolved ledger entries only
            for existing_entry in self._entries.values():
                if existing_entry.status in (
                    UNRESOLVED_LEDGER_STATUSES | PRE_SUBMISSION_LEDGER_STATUSES
                ):
                    continue
                if existing_entry.position_ticket:
                    matched_pos_tickets.add(existing_entry.position_ticket)
                if existing_entry.deal_ticket:
                    matched_deal_tickets.add(existing_entry.deal_ticket)
                if existing_entry.order_ticket:
                    matched_order_tickets.add(existing_entry.order_ticket)

            for cid, entry in list(self._entries.items()):
                other_ids = all_known_ids - {entry.client_order_id, entry.signal_id}

                # If an entry is in FILLED state (e.g. from an interrupted two-write state) and has a live open position, promote to ACTIVE
                if entry.status == OrderLedgerStatus.FILLED:
                    open_pos_for_filled = [
                        p
                        for p in positions
                        if self._matches_entry_on_broker(
                            entry,
                            comment=p.comment,
                            magic=p.magic,
                            symbol=p.symbol,
                            direction=(
                                SignalDirection.BUY if p.position_type == 0 else SignalDirection.SELL
                            ),
                            position_ticket=p.ticket,
                            other_entry_ids=other_ids,
                        )
                    ]
                    if open_pos_for_filled:
                        pos_tkt = open_pos_for_filled[-1].ticket
                        matched_pos_tickets.add(pos_tkt)
                        reason_msg = (
                            f"Broker reconciliation promoted FILLED entry '{cid}' with open broker "
                            f"position (ticket={pos_tkt}) to ACTIVE."
                        )
                        ev = LedgerTransitionRecord(
                            from_status=entry.status,
                            to_status=OrderLedgerStatus.ACTIVE,
                            timestamp_utc=now_utc,
                            reason=reason_msg,
                            retcode=entry.retcode,
                        )
                        self._entries[cid] = entry.model_copy(
                            update={
                                "status": OrderLedgerStatus.ACTIVE,
                                "reconciliation_status": ReconciliationStatus.RECONCILED_EXECUTED,
                                "position_ticket": pos_tkt,
                                "updated_at_utc": now_utc,
                                "status_reason": reason_msg,
                                "history": (*entry.history, ev),
                            }
                        )
                        reconciled_executed.append(cid)
                    continue

                if entry.status not in (UNRESOLVED_LEDGER_STATUSES | PRE_SUBMISSION_LEDGER_STATUSES):
                    continue

                # Find matching entry deals, open positions, and active working orders not already claimed
                entry_deals: list[MT5DealRecord] = [
                    d
                    for d in exec_deals
                    if d.ticket not in matched_deal_tickets
                    and (d.order == 0 or d.order not in matched_order_tickets or d.order == entry.order_ticket)
                    and self._matches_entry_on_broker(
                        entry,
                        comment=d.comment,
                        magic=d.magic,
                        symbol=d.symbol,
                        direction=(
                            SignalDirection.BUY
                            if d.deal_type == int(MT5DealType.DEAL_TYPE_BUY)
                            else SignalDirection.SELL
                        ),
                        order_ticket=d.order,
                        deal_ticket=d.ticket,
                        position_ticket=d.position_id,
                        other_entry_ids=other_ids,
                    )
                ]
                entry_positions: list[MT5PositionRecord] = [
                    p
                    for p in positions
                    if p.ticket not in matched_pos_tickets
                    and self._matches_entry_on_broker(
                        entry,
                        comment=p.comment,
                        magic=p.magic,
                        symbol=p.symbol,
                        direction=(
                            SignalDirection.BUY if p.position_type == 0 else SignalDirection.SELL
                        ),
                        position_ticket=p.ticket,
                        other_entry_ids=other_ids,
                    )
                ]
                entry_orders: list[MT5OrderRecord] = [
                    o
                    for o in orders
                    if o.ticket not in matched_order_tickets
                    and self._matches_entry_on_broker(
                        entry,
                        comment=o.comment,
                        magic=o.magic,
                        symbol=o.symbol,
                        direction=(
                            SignalDirection.BUY if (o.order_type % 2 == 0) else SignalDirection.SELL
                        ),
                        order_ticket=o.ticket,
                        position_ticket=o.position_id,
                        other_entry_ids=other_ids,
                    )
                ]

                for d in entry_deals:
                    matched_deal_tickets.add(d.ticket)
                    if d.order > 0:
                        matched_order_tickets.add(d.order)
                    if d.position_id > 0:
                        matched_pos_tickets.add(d.position_id)
                for p in entry_positions:
                    matched_pos_tickets.add(p.ticket)
                for o in entry_orders:
                    matched_order_tickets.add(o.ticket)

                if entry_deals or entry_positions:
                    deal_vol = sum((d.volume for d in entry_deals), Decimal("0"))
                    pos_vol = sum((p.volume for p in entry_positions), Decimal("0"))
                    exec_vol = deal_vol if deal_vol > Decimal("0") else pos_vol
                    rem_vol = max(Decimal("0"), entry.requested_volume - exec_vol)

                    fill_px = (
                        entry_deals[-1].price
                        if entry_deals
                        else (entry_positions[-1].price_open if entry_positions else entry.requested_price)
                    )
                    ord_tkt = (
                        entry_deals[-1].order
                        if (entry_deals and entry_deals[-1].order > 0)
                        else (entry_orders[-1].ticket if entry_orders else entry.order_ticket)
                    )
                    dl_tkt = entry_deals[-1].ticket if entry_deals else entry.deal_ticket
                    pos_tkt = (
                        entry_positions[-1].ticket
                        if entry_positions
                        else (
                            entry_deals[-1].position_id
                            if (entry_deals and entry_deals[-1].position_id > 0)
                            else entry.position_ticket
                        )
                    )

                    if entry_orders:
                        target_status = OrderLedgerStatus.PARTIALLY_FILLED
                        recon_st = ReconciliationStatus.PENDING
                        still_working.append(cid)
                    elif exec_vol < entry.requested_volume:
                        target_status = OrderLedgerStatus.PARTIALLY_FILLED
                        recon_st = ReconciliationStatus.RECONCILED_EXECUTED
                    elif entry_positions:
                        target_status = OrderLedgerStatus.ACTIVE
                        recon_st = ReconciliationStatus.RECONCILED_EXECUTED
                    else:
                        target_status = OrderLedgerStatus.FILLED
                        recon_st = ReconciliationStatus.RECONCILED_EXECUTED

                    reason_msg = (
                        f"Broker reconciliation confirmed execution ({target_status.value}): "
                        f"executed_volume={exec_vol}, fill_price={fill_px}, "
                        f"order_ticket={ord_tkt}, deal_ticket={dl_tkt}, position_ticket={pos_tkt}."
                    )
                    ev = LedgerTransitionRecord(
                        from_status=entry.status,
                        to_status=target_status,
                        timestamp_utc=now_utc,
                        reason=reason_msg,
                        retcode=entry.retcode,
                    )
                    self._entries[cid] = entry.model_copy(
                        update={
                            "status": target_status,
                            "reconciliation_status": recon_st,
                            "executed_volume": exec_vol,
                            "remaining_volume": rem_vol,
                            "fill_price": fill_px,
                            "order_ticket": ord_tkt,
                            "deal_ticket": dl_tkt,
                            "position_ticket": pos_tkt,
                            "updated_at_utc": now_utc,
                            "status_reason": reason_msg,
                            "history": (*entry.history, ev),
                        }
                    )
                    reconciled_executed.append(cid)

                elif entry_orders:
                    # Order is still live/working on the broker server without a deal/position yet
                    work_order = entry_orders[-1]
                    reason_msg = (
                        f"Broker reconciliation found order '{cid}' still working on server "
                        f"(order_ticket={work_order.ticket}, volume_current={work_order.volume_current}); "
                        "remaining in RECONCILIATION_REQUIRED."
                    )
                    ev = LedgerTransitionRecord(
                        from_status=entry.status,
                        to_status=OrderLedgerStatus.RECONCILIATION_REQUIRED,
                        timestamp_utc=now_utc,
                        reason=reason_msg,
                        retcode=entry.retcode,
                    )
                    self._entries[cid] = entry.model_copy(
                        update={
                            "status": OrderLedgerStatus.RECONCILIATION_REQUIRED,
                            "reconciliation_status": ReconciliationStatus.PENDING,
                            "order_ticket": work_order.ticket,
                            "updated_at_utc": now_utc,
                            "status_reason": reason_msg,
                            "history": (*entry.history, ev),
                        }
                    )
                    still_working.append(cid)

                else:
                    # Proven absent across positions, working orders, and deal history
                    reason_msg = (
                        f"Broker reconciliation proved order '{cid}' did not execute on server "
                        "(absent from positions, active orders, and deal history)."
                    )
                    ev = LedgerTransitionRecord(
                        from_status=entry.status,
                        to_status=OrderLedgerStatus.CANCELLED,
                        timestamp_utc=now_utc,
                        reason=reason_msg,
                        retcode=entry.retcode,
                    )
                    self._entries[cid] = entry.model_copy(
                        update={
                            "status": OrderLedgerStatus.CANCELLED,
                            "reconciliation_status": ReconciliationStatus.RECONCILED_NOT_EXECUTED,
                            "executed_volume": Decimal("0"),
                            "remaining_volume": entry.requested_volume,
                            "updated_at_utc": now_utc,
                            "status_reason": reason_msg,
                            "history": (*entry.history, ev),
                        }
                    )
                    reconciled_not_executed.append(cid)

            # Discover any orphan broker positions, working orders, or closed execution deals carrying our magic number
            discovered_orphans: list[str] = []
            magic = self._safety_config.magic_number

            # 1. Orphan open positions
            for p in positions:
                if p.magic != magic or p.ticket in matched_pos_tickets:
                    continue
                pos_dir = SignalDirection.BUY if p.position_type == 0 else SignalDirection.SELL
                already_matched = any(
                    self._matches_entry_on_broker(
                        e,
                        comment=p.comment,
                        magic=p.magic,
                        symbol=p.symbol,
                        direction=pos_dir,
                        position_ticket=p.ticket,
                    )
                    for e in self._entries.values()
                )
                if already_matched:
                    matched_pos_tickets.add(p.ticket)
                    continue

                # Also mark any entry deal for this orphan position as matched so it isn't double-added as an orphan deal
                linked_deals = [
                    d
                    for d in exec_deals
                    if d.magic == magic
                    and (
                        d.position_id == p.ticket
                        or (
                            extract_primary_comment_id(p.comment)
                            and extract_primary_comment_id(d.comment)
                            == extract_primary_comment_id(p.comment)
                        )
                    )
                ]
                for ld in linked_deals:
                    matched_deal_tickets.add(ld.ticket)
                    if ld.order > 0:
                        matched_order_tickets.add(ld.order)

                primary_cid = extract_primary_comment_id(p.comment)
                orphan_cid = primary_cid or f"orphan-pos-{p.ticket}"
                if orphan_cid in self._entries:
                    orphan_cid = f"{orphan_cid}-pos-{p.ticket}"

                sl_val = p.sl if p.sl > Decimal("0") else (
                    p.price_open * Decimal("0.99")
                    if pos_dir == SignalDirection.BUY
                    else p.price_open * Decimal("1.01")
                )
                tp_val = p.tp if p.tp > Decimal("0") else (
                    p.price_open * Decimal("1.02")
                    if pos_dir == SignalDirection.BUY
                    else p.price_open * Decimal("0.98")
                )
                reason_msg = (
                    f"Reconstructed orphan broker position (ticket={p.ticket}, comment='{p.comment}') "
                    "during crash recovery reconciliation."
                )
                ev = LedgerTransitionRecord(
                    from_status=None,
                    to_status=OrderLedgerStatus.ACTIVE,
                    timestamp_utc=now_utc,
                    reason=reason_msg,
                )
                orphan_entry = ExecutionLedgerEntry(
                    client_order_id=orphan_cid,
                    signal_id=primary_cid or orphan_cid,
                    symbol=p.symbol,
                    timeframe=Timeframe.H1,
                    direction=pos_dir,
                    magic=p.magic,
                    requested_volume=p.volume,
                    executed_volume=p.volume,
                    remaining_volume=Decimal("0"),
                    requested_price=p.price_open,
                    fill_price=p.price_open,
                    stop_loss=sl_val,
                    take_profit=tp_val,
                    filling_mode=MT5OrderFillingMode.ORDER_FILLING_FOK,
                    order_ticket=linked_deals[-1].order if (linked_deals and linked_deals[-1].order > 0) else None,
                    deal_ticket=linked_deals[-1].ticket if linked_deals else None,
                    position_ticket=p.ticket,
                    status=OrderLedgerStatus.ACTIVE,
                    reconciliation_status=ReconciliationStatus.RECONCILED_EXECUTED,
                    created_at_utc=p.time_utc,
                    updated_at_utc=now_utc,
                    status_reason=reason_msg,
                    history=(ev,),
                )
                self._entries[orphan_cid] = orphan_entry
                matched_pos_tickets.add(p.ticket)
                discovered_orphans.append(orphan_cid)

            # 2. Orphan active working orders (orders_get)
            for o in orders:
                if o.magic != magic or o.ticket in matched_order_tickets:
                    continue
                ord_dir = SignalDirection.BUY if (o.order_type % 2 == 0) else SignalDirection.SELL
                already_matched_ord = any(
                    self._matches_entry_on_broker(
                        e,
                        comment=o.comment,
                        magic=o.magic,
                        symbol=o.symbol,
                        direction=ord_dir,
                        order_ticket=o.ticket,
                        position_ticket=o.position_id,
                    )
                    for e in self._entries.values()
                )
                if already_matched_ord:
                    matched_order_tickets.add(o.ticket)
                    continue

                primary_cid = extract_primary_comment_id(o.comment)
                orphan_cid = primary_cid or f"orphan-ord-{o.ticket}"
                if orphan_cid in self._entries:
                    orphan_cid = f"{orphan_cid}-ord-{o.ticket}"

                sl_val = o.sl if o.sl > Decimal("0") else (
                    o.price_open * Decimal("0.99")
                    if ord_dir == SignalDirection.BUY
                    else o.price_open * Decimal("1.01")
                )
                tp_val = o.tp if o.tp > Decimal("0") else (
                    o.price_open * Decimal("1.02")
                    if ord_dir == SignalDirection.BUY
                    else o.price_open * Decimal("0.98")
                )
                rem_vol = o.volume_current if o.volume_current > Decimal("0") else o.volume_initial
                reason_msg = (
                    f"Discovered orphan working order on broker (ticket={o.ticket}, comment='{o.comment}') "
                    "during crash recovery reconciliation; locked in RECONCILIATION_REQUIRED."
                )
                ev = LedgerTransitionRecord(
                    from_status=None,
                    to_status=OrderLedgerStatus.RECONCILIATION_REQUIRED,
                    timestamp_utc=now_utc,
                    reason=reason_msg,
                )
                orphan_ord_entry = ExecutionLedgerEntry(
                    client_order_id=orphan_cid,
                    signal_id=primary_cid or orphan_cid,
                    symbol=o.symbol,
                    timeframe=Timeframe.H1,
                    direction=ord_dir,
                    magic=o.magic,
                    requested_volume=o.volume_initial,
                    executed_volume=max(Decimal("0"), o.volume_initial - rem_vol),
                    remaining_volume=rem_vol,
                    requested_price=o.price_open,
                    fill_price=None,
                    stop_loss=sl_val,
                    take_profit=tp_val,
                    filling_mode=MT5OrderFillingMode.ORDER_FILLING_FOK,
                    order_ticket=o.ticket,
                    position_ticket=o.position_id if o.position_id > 0 else None,
                    status=OrderLedgerStatus.RECONCILIATION_REQUIRED,
                    reconciliation_status=ReconciliationStatus.PENDING,
                    created_at_utc=o.time_setup_utc,
                    updated_at_utc=now_utc,
                    status_reason=reason_msg,
                    history=(ev,),
                )
                self._entries[orphan_cid] = orphan_ord_entry
                matched_order_tickets.add(o.ticket)
                discovered_orphans.append(orphan_cid)
                still_working.append(orphan_cid)

            # 3. Orphan historical execution deals (already-closed positions missing from stale .bak)
            for d in exec_deals:
                if (
                    d.magic != magic
                    or d.ticket in matched_deal_tickets
                    or (d.position_id > 0 and d.position_id in matched_pos_tickets)
                    or (d.order > 0 and d.order in matched_order_tickets)
                ):
                    continue
                deal_dir = (
                    SignalDirection.BUY
                    if d.deal_type == int(MT5DealType.DEAL_TYPE_BUY)
                    else SignalDirection.SELL
                )
                already_matched_deal = any(
                    self._matches_entry_on_broker(
                        e,
                        comment=d.comment,
                        magic=d.magic,
                        symbol=d.symbol,
                        direction=deal_dir,
                        order_ticket=d.order,
                        deal_ticket=d.ticket,
                        position_ticket=d.position_id,
                    )
                    for e in self._entries.values()
                )
                if already_matched_deal:
                    matched_deal_tickets.add(d.ticket)
                    if d.position_id > 0:
                        matched_pos_tickets.add(d.position_id)
                    if d.order > 0:
                        matched_order_tickets.add(d.order)
                    continue

                primary_cid = extract_primary_comment_id(d.comment)
                orphan_cid = primary_cid or f"orphan-deal-{d.ticket}"
                if orphan_cid in self._entries:
                    orphan_cid = f"{orphan_cid}-deal-{d.ticket}"

                px_val = d.price if d.price > Decimal("0") else Decimal("1.00")
                d_vol = d.volume if d.volume > Decimal("0") else Decimal("0.01")
                sl_val = (
                    px_val * Decimal("0.99")
                    if deal_dir == SignalDirection.BUY
                    else px_val * Decimal("1.01")
                )
                tp_val = (
                    px_val * Decimal("1.02")
                    if deal_dir == SignalDirection.BUY
                    else px_val * Decimal("0.98")
                )
                reason_msg = (
                    f"Reconstructed closed orphan broker execution deal (ticket={d.ticket}, "
                    f"order={d.order}, position_id={d.position_id}, comment='{d.comment}') "
                    "during crash recovery reconciliation."
                )
                ev = LedgerTransitionRecord(
                    from_status=None,
                    to_status=OrderLedgerStatus.FILLED,
                    timestamp_utc=now_utc,
                    reason=reason_msg,
                )
                orphan_deal_entry = ExecutionLedgerEntry(
                    client_order_id=orphan_cid,
                    signal_id=primary_cid or orphan_cid,
                    symbol=d.symbol or "XAUUSD",
                    timeframe=Timeframe.H1,
                    direction=deal_dir,
                    magic=d.magic,
                    requested_volume=d_vol,
                    executed_volume=d_vol,
                    remaining_volume=Decimal("0"),
                    requested_price=px_val,
                    fill_price=px_val,
                    stop_loss=sl_val,
                    take_profit=tp_val,
                    filling_mode=MT5OrderFillingMode.ORDER_FILLING_FOK,
                    order_ticket=d.order if d.order > 0 else None,
                    deal_ticket=d.ticket,
                    position_ticket=d.position_id if d.position_id > 0 else None,
                    status=OrderLedgerStatus.FILLED,
                    reconciliation_status=ReconciliationStatus.RECONCILED_EXECUTED,
                    created_at_utc=d.time_utc,
                    updated_at_utc=now_utc,
                    status_reason=reason_msg,
                    history=(ev,),
                )
                self._entries[orphan_cid] = orphan_deal_entry
                matched_deal_tickets.add(d.ticket)
                if d.position_id > 0:
                    matched_pos_tickets.add(d.position_id)
                if d.order > 0:
                    matched_order_tickets.add(d.order)
                discovered_orphans.append(orphan_cid)

            has_remaining_unresolved = bool(still_working) or self.has_unresolved_entries()
            payload = self._serialize_payload()
            if self._health_status == LedgerHealthStatus.RECOVERY_REQUIRED_LOCKED:
                saved_env = self._persistence.commit_recovered_payload(
                    payload,
                    account_login=account.login,
                    server=account.server,
                    now_utc=now_utc,
                )
                if not has_remaining_unresolved:
                    self._health_status = LedgerHealthStatus.HEALTHY
            else:
                saved_env = self._persistence.save(
                    payload,
                    account_login=account.login,
                    server=account.server,
                    now_utc=now_utc,
                )
            self._last_generation = saved_env.generation

            return LedgerReconciliationReport(
                health_status=self._health_status,
                reconciled_executed_ids=tuple(reconciled_executed),
                reconciled_not_executed_ids=tuple(reconciled_not_executed),
                still_working_ids=tuple(still_working),
                discovered_orphan_ids=tuple(discovered_orphans),
                envelope_generation=saved_env.generation,
            )
