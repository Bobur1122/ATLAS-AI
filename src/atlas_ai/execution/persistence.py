"""Crash-safe atomic JSON persistence with cross-process OS file locking, SHA-256 checksums,
monotonic generation counters, previous-checksum hash chains, and strict account identity binding.
"""

from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import time
from typing import Any
import uuid

from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator

from atlas_ai.core.logger import get_logger
from atlas_ai.execution.exceptions import (
    LedgerRecoveryLockError,
    PersistenceAccountMismatchError,
    PersistenceCorruptionError,
    PersistenceError,
    PersistenceLockTimeoutError,
    PersistenceRollbackDetectedError,
)

logger = get_logger("execution.persistence")

GENESIS_CHECKSUM: str = "0" * 64
ENVELOPE_SCHEMA_VERSION: str = "1.0.0"


def compute_envelope_checksum(
    *,
    schema_version: str,
    generation: int,
    previous_checksum: str,
    account_login: int,
    server: str,
    payload: dict[str, Any],
) -> str:
    """Compute a deterministic SHA-256 hex digest over canonical envelope metadata and payload."""
    canonical_obj = {
        "schema_version": schema_version,
        "generation": int(generation),
        "previous_checksum": previous_checksum,
        "account_login": int(account_login),
        "server": server.strip().lower(),
        "payload": payload,
    }
    canonical_bytes = json.dumps(
        canonical_obj,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode("utf-8")
    return hashlib.sha256(canonical_bytes).hexdigest()


class PersistedEnvelope(BaseModel):
    """Immutable checksummed and monotonically versioned persistence envelope."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    schema_version: str = Field(default=ENVELOPE_SCHEMA_VERSION, min_length=1)
    generation: int = Field(..., ge=1, description="Strictly monotonic generation counter (1, 2, 3, ...).")
    previous_checksum: str = Field(
        ...,
        min_length=64,
        max_length=64,
        description="SHA-256 payload_checksum of generation (G - 1), or 64 zeros for generation 1.",
    )
    payload_checksum: str = Field(
        ...,
        min_length=64,
        max_length=64,
        description="SHA-256 checksum over canonical metadata and payload.",
    )
    account_login: int = Field(..., gt=0, description="Bound MT5 demo account login.")
    server: str = Field(..., min_length=1, description="Bound MT5 demo server name.")
    written_at_utc: datetime = Field(..., description="Timezone-aware UTC write timestamp.")
    payload: dict[str, Any] = Field(..., description="Serialized JSON payload dictionary.")

    @field_validator("written_at_utc")
    @classmethod
    def ensure_utc(cls, value: datetime) -> datetime:
        if value.tzinfo is None:
            raise ValueError("PersistedEnvelope written_at_utc must be timezone-aware.")
        return value.astimezone(timezone.utc)


class PersistenceLoadResult(BaseModel):
    """Non-silent inspection report for crash recovery and ledger reconciliation."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    envelope: PersistedEnvelope | None = Field(default=None)
    backup_envelope: PersistedEnvelope | None = Field(default=None)
    valid_tmp_envelopes: tuple[PersistedEnvelope, ...] = Field(
        default=(),
        description="Valid checksum-verified envelopes recovered from interrupted .tmp.* files.",
    )
    highest_valid_envelope: PersistedEnvelope | None = Field(
        default=None,
        description="Highest-generation valid envelope across primary, backup, and .tmp.* files.",
    )
    interrupted_tmp_files: tuple[str, ...] = Field(default=())
    primary_missing: bool = Field(default=False)
    primary_corrupted: bool = Field(default=False)
    primary_error: str | None = Field(default=None)
    recovery_required: bool = Field(
        default=False,
        description="True when primary is missing (with .bak/.tmp present), corrupted, or interrupted .tmp.* files exist.",
    )


class FileLock:
    """OS-kernel byte-range file lock (`msvcrt.locking` on Windows, `fcntl.flock` on POSIX).

    Key Safety Properties:
    1. Locks byte 0 (length 1) of a persistent sidecar `<target>.lock` file.
    2. Never unlinks the `.lock` file on exit so concurrent processes never lock disassociated inodes.
    3. Initializes the 1-byte lock file body only AFTER acquiring the OS lock (M-2 remediation).
    4. Because the lock lives in the OS kernel file-descriptor table, if the owning process crashes
       or is killed (`TerminateProcess` / `SIGKILL`), the OS kernel automatically closes the file
       descriptor and releases the lock immediately (zero stale lock files).
    """

    def __init__(
        self,
        lock_path: Path | str,
        *,
        timeout_seconds: float = 5.0,
        poll_interval_seconds: float = 0.02,
    ) -> None:
        if timeout_seconds <= 0:
            raise ValueError(f"Lock timeout_seconds must be > 0, got {timeout_seconds}.")
        self._lock_path = Path(lock_path)
        self._timeout_seconds = float(timeout_seconds)
        self._poll_interval_seconds = float(poll_interval_seconds)
        self._fd: int | None = None

    @property
    def lock_path(self) -> Path:
        return self._lock_path

    @property
    def is_locked(self) -> bool:
        return self._fd is not None

    def acquire(self) -> None:
        """Acquire an exclusive non-blocking OS lock within `timeout_seconds`."""
        if self._fd is not None:
            raise PersistenceError(f"FileLock on '{self._lock_path}' is already acquired by this instance.")

        self._lock_path.parent.mkdir(parents=True, exist_ok=True)
        flags = os.O_CREAT | os.O_RDWR
        fd = os.open(str(self._lock_path), flags, 0o666)
        try:
            deadline = time.monotonic() + self._timeout_seconds
            while True:
                try:
                    os.lseek(fd, 0, os.SEEK_SET)
                    self._os_lock_nonblocking(fd)
                    # Only initialize the 1-byte marker AFTER acquiring the exclusive OS lock (M-2 fix)
                    try:
                        stat_info = os.fstat(fd)
                        if stat_info.st_size == 0:
                            os.lseek(fd, 0, os.SEEK_SET)
                            os.write(fd, b"0")
                            os.lseek(fd, 0, os.SEEK_SET)
                    except OSError:
                        pass
                    self._fd = fd
                    return
                except OSError as exc:
                    if time.monotonic() >= deadline:
                        raise PersistenceLockTimeoutError(
                            f"Timed out after {self._timeout_seconds:.2f}s waiting for exclusive OS file lock "
                            f"on '{self._lock_path}' ({exc})."
                        ) from exc
                    time.sleep(self._poll_interval_seconds)
        except Exception:
            try:
                os.close(fd)
            except OSError:
                pass
            raise

    def release(self) -> None:
        """Release the exclusive OS lock and close the underlying file descriptor."""
        fd = self._fd
        if fd is None:
            return
        self._fd = None
        try:
            os.lseek(fd, 0, os.SEEK_SET)
            self._os_unlock(fd)
        finally:
            os.close(fd)

    @staticmethod
    def _os_lock_nonblocking(fd: int) -> None:
        if os.name == "nt":
            import msvcrt

            msvcrt.locking(fd, msvcrt.LK_NBLCK, 1)
        else:
            import fcntl

            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)

    @staticmethod
    def _os_unlock(fd: int) -> None:
        if os.name == "nt":
            import msvcrt

            msvcrt.locking(fd, msvcrt.LK_UNLCK, 1)
        else:
            import fcntl

            fcntl.flock(fd, fcntl.LOCK_UN)

    def __enter__(self) -> "FileLock":
        self.acquire()
        return self

    def __exit__(self, exc_type: object, exc_val: object, exc_tb: object) -> None:
        self.release()


class CrashSafeJsonPersistence:
    """Crash-safe JSON store with atomic temp-file replacement, OS file locking,
    SHA-256 verification, monotonic generation tracking, previous-checksum hash chains,
    and strict account identity binding.

    Safety Rules:
    - Never silently restores a stale `.bak` file to permit new trades.
    - Detects interrupted writes across both `<file>.tmp.*` and `<file>.bak.tmp.*`.
    - Inspects and verifies valid higher-generation `.tmp.*` envelopes during recovery so a
      flushed write-ahead envelope is never discarded.
    - Fails closed (`PersistenceCorruptionError`) if existing primary/backup files are all corrupted,
      never resetting `generation` to 1.
    """

    def __init__(
        self,
        file_path: Path | str,
        *,
        lock_timeout_seconds: float = 5.0,
    ) -> None:
        self._primary_path = Path(file_path)
        self._backup_path = Path(f"{self._primary_path}.bak")
        self._lock_path = Path(f"{self._primary_path}.lock")
        self._lock_timeout_seconds = float(lock_timeout_seconds)
        self._last_known_generation: int = 0
        self._last_known_checksum: str | None = None

    @property
    def primary_path(self) -> Path:
        return self._primary_path

    @property
    def backup_path(self) -> Path:
        return self._backup_path

    @property
    def lock_path(self) -> Path:
        return self._lock_path

    @property
    def last_known_generation(self) -> int:
        return self._last_known_generation

    def find_primary_temp_files(self) -> tuple[Path, ...]:
        """Return lingering `<primary_name>.tmp.*` files left by an interrupted primary write."""
        parent = self._primary_path.parent
        if not parent.exists():
            return ()
        prefix = f"{self._primary_path.name}.tmp."
        return tuple(
            sorted(p for p in parent.iterdir() if p.is_file() and p.name.startswith(prefix))
        )

    def find_backup_temp_files(self) -> tuple[Path, ...]:
        """Return lingering `<backup_name>.tmp.*` (`<primary_name>.bak.tmp.*`) files."""
        parent = self._backup_path.parent
        if not parent.exists():
            return ()
        prefix = f"{self._backup_path.name}.tmp."
        return tuple(
            sorted(p for p in parent.iterdir() if p.is_file() and p.name.startswith(prefix))
        )

    def find_interrupted_temp_files(self) -> tuple[Path, ...]:
        """Return all lingering primary (`.tmp.*`) and backup (`.bak.tmp.*`) temporary files."""
        combined = {p: None for p in (*self.find_primary_temp_files(), *self.find_backup_temp_files())}
        return tuple(sorted(combined.keys()))

    def _read_and_verify_file(
        self,
        path: Path,
        *,
        expected_login: int | None = None,
        expected_server: str | None = None,
    ) -> PersistedEnvelope:
        """Read, parse, checksum-verify, and account-verify a persisted envelope file."""
        try:
            raw_text = path.read_text(encoding="utf-8")
        except OSError as exc:
            raise PersistenceCorruptionError(f"Failed to read persistence file '{path}': {exc}") from exc

        try:
            raw_data = json.loads(raw_text)
        except json.JSONDecodeError as exc:
            raise PersistenceCorruptionError(
                f"Corrupted JSON syntax in persistence file '{path}': {exc}"
            ) from exc

        if not isinstance(raw_data, dict):
            raise PersistenceCorruptionError(
                f"Persistence file '{path}' root must be a JSON object, got {type(raw_data).__name__}."
            )

        try:
            envelope = PersistedEnvelope.model_validate(raw_data)
        except ValidationError as exc:
            raise PersistenceCorruptionError(
                f"Envelope schema validation failed for '{path}': {exc}"
            ) from exc

        computed_checksum = compute_envelope_checksum(
            schema_version=envelope.schema_version,
            generation=envelope.generation,
            previous_checksum=envelope.previous_checksum,
            account_login=envelope.account_login,
            server=envelope.server,
            payload=envelope.payload,
        )
        if envelope.payload_checksum != computed_checksum:
            raise PersistenceCorruptionError(
                f"SHA-256 integrity checksum mismatch in '{path}': "
                f"recorded={envelope.payload_checksum}, computed={computed_checksum}."
            )

        if envelope.generation == 1 and envelope.previous_checksum != GENESIS_CHECKSUM:
            raise PersistenceCorruptionError(
                f"Generation 1 envelope in '{path}' must have previous_checksum == GENESIS ({GENESIS_CHECKSUM}), "
                f"got {envelope.previous_checksum}."
            )
        if envelope.generation > 1 and envelope.previous_checksum == GENESIS_CHECKSUM:
            raise PersistenceCorruptionError(
                f"Generation {envelope.generation} envelope in '{path}' cannot have GENESIS previous_checksum."
            )

        if expected_login is not None and envelope.account_login != expected_login:
            raise PersistenceAccountMismatchError(
                f"Persisted file '{path}' belongs to account_login={envelope.account_login}, "
                f"which does not match active account_login={expected_login}."
            )

        if (
            expected_server is not None
            and envelope.server.strip().lower() != expected_server.strip().lower()
        ):
            raise PersistenceAccountMismatchError(
                f"Persisted file '{path}' belongs to server='{envelope.server}', "
                f"which does not match active server='{expected_server}'."
            )

        return envelope

    def _validate_anti_rollback_and_chain(
        self,
        primary: PersistedEnvelope,
        backup: PersistedEnvelope | None,
    ) -> None:
        """Enforce monotonic generation ordering and previous_checksum hash-chain linkage."""
        if backup is not None:
            if (
                primary.account_login != backup.account_login
                or primary.server.strip().lower() != backup.server.strip().lower()
            ):
                raise PersistenceAccountMismatchError(
                    f"Account identity mismatch between primary ({primary.account_login}@{primary.server}) "
                    f"and backup ({backup.account_login}@{backup.server})."
                )
            if primary.generation < backup.generation:
                raise PersistenceRollbackDetectedError(
                    f"ROLLBACK DETECTED on '{self._primary_path}': primary generation ({primary.generation}) "
                    f"is older than backup generation ({backup.generation})."
                )
            if primary.generation == backup.generation:
                if primary.payload_checksum != backup.payload_checksum:
                    raise PersistenceCorruptionError(
                        f"Primary and backup files for '{self._primary_path}' share generation {primary.generation} "
                        "but have conflicting SHA-256 checksums."
                    )
            elif primary.generation == backup.generation + 1:
                if primary.previous_checksum != backup.payload_checksum:
                    raise PersistenceCorruptionError(
                        f"Broken previous_checksum chain between backup (gen={backup.generation}, "
                        f"sha256={backup.payload_checksum}) and primary (gen={primary.generation}, "
                        f"previous_checksum={primary.previous_checksum})."
                    )

        if self._last_known_generation > 0:
            if primary.generation < self._last_known_generation:
                raise PersistenceRollbackDetectedError(
                    f"ROLLBACK DETECTED on '{self._primary_path}': on-disk generation ({primary.generation}) "
                    f"is lower than in-memory last known generation ({self._last_known_generation})."
                )
            if (
                primary.generation == self._last_known_generation
                and self._last_known_checksum is not None
                and primary.payload_checksum != self._last_known_checksum
            ):
                raise PersistenceCorruptionError(
                    f"On-disk checksum ({primary.payload_checksum}) at generation {primary.generation} "
                    f"does not match in-memory checksum ({self._last_known_checksum})."
                )

    @staticmethod
    def _validate_candidate_chain(candidates: list[PersistedEnvelope]) -> PersistedEnvelope:
        """Validate account consistency and previous_checksum linkage across sorted valid envelopes,
        returning the highest-generation verified envelope.
        """
        ordered = sorted(candidates, key=lambda env: env.generation)
        first = ordered[0]
        for prev_env, curr_env in zip(ordered, ordered[1:]):
            if (
                curr_env.account_login != first.account_login
                or curr_env.server.strip().lower() != first.server.strip().lower()
            ):
                raise PersistenceAccountMismatchError(
                    f"Account identity mismatch across candidate recovery envelopes "
                    f"({first.account_login}@{first.server} vs {curr_env.account_login}@{curr_env.server})."
                )
            if curr_env.generation == prev_env.generation:
                if curr_env.payload_checksum != prev_env.payload_checksum:
                    raise PersistenceCorruptionError(
                        f"Conflicting recovery envelopes at generation {curr_env.generation} "
                        f"({prev_env.payload_checksum} != {curr_env.payload_checksum})."
                    )
            elif curr_env.generation == prev_env.generation + 1:
                if curr_env.previous_checksum != prev_env.payload_checksum:
                    raise PersistenceCorruptionError(
                        f"Broken previous_checksum chain between recovery envelopes at generations "
                        f"{prev_env.generation} -> {curr_env.generation}."
                    )
        return ordered[-1]

    def load(
        self,
        *,
        expected_login: int | None = None,
        expected_server: str | None = None,
    ) -> PersistedEnvelope | None:
        """Load and strictly verify the primary persisted envelope under an OS file lock.

        Never silently restores from `.bak`. Raises `LedgerRecoveryLockError` if an interrupted
        `.tmp.*` or `.bak.tmp.*` write is detected or if the primary file is missing while a `.bak` exists.
        """
        with FileLock(self._lock_path, timeout_seconds=self._lock_timeout_seconds):
            return self._load_unlocked(
                expected_login=expected_login,
                expected_server=expected_server,
            )

    def _load_unlocked(
        self,
        *,
        expected_login: int | None = None,
        expected_server: str | None = None,
    ) -> PersistedEnvelope | None:
        tmp_files = self.find_interrupted_temp_files()
        if tmp_files:
            names = ", ".join(p.name for p in tmp_files)
            raise LedgerRecoveryLockError(
                f"Interrupted atomic write detected for '{self._primary_path}' "
                f"(lingering temporary file(s): {names}). Recovery reconciliation required."
            )

        primary_exists = self._primary_path.exists()
        backup_exists = self._backup_path.exists()

        if not primary_exists:
            if backup_exists:
                raise LedgerRecoveryLockError(
                    f"Primary persistence file '{self._primary_path}' is missing while backup "
                    f"'{self._backup_path}' exists. Silent backup restoration is forbidden; "
                    "explicit recovery reconciliation is required."
                )
            if self._last_known_generation > 0:
                raise PersistenceRollbackDetectedError(
                    f"Primary persistence file '{self._primary_path}' was deleted after reaching "
                    f"generation {self._last_known_generation}."
                )
            return None

        primary_env = self._read_and_verify_file(
            self._primary_path,
            expected_login=expected_login,
            expected_server=expected_server,
        )

        backup_env: PersistedEnvelope | None = None
        if backup_exists:
            backup_env = self._read_and_verify_file(
                self._backup_path,
                expected_login=expected_login,
                expected_server=expected_server,
            )

        self._validate_anti_rollback_and_chain(primary_env, backup_env)
        self._last_known_generation = primary_env.generation
        self._last_known_checksum = primary_env.payload_checksum
        return primary_env

    def inspect_recovery_state(
        self,
        *,
        expected_login: int | None = None,
        expected_server: str | None = None,
    ) -> PersistenceLoadResult:
        """Inspect primary, backup, and temporary files under lock without silently restoring `.bak`.

        Also parses and verifies any valid higher-generation `.tmp.*` envelopes so write-ahead
        data flushed before a crash is preserved for reconciliation.
        """
        with FileLock(self._lock_path, timeout_seconds=self._lock_timeout_seconds):
            all_tmp_paths = self.find_interrupted_temp_files()
            tmp_files = tuple(p.name for p in all_tmp_paths)
            primary_exists = self._primary_path.exists()
            backup_exists = self._backup_path.exists()

            valid_candidates: list[PersistedEnvelope] = []

            backup_env: PersistedEnvelope | None = None
            if backup_exists:
                try:
                    backup_env = self._read_and_verify_file(
                        self._backup_path,
                        expected_login=expected_login,
                        expected_server=expected_server,
                    )
                    valid_candidates.append(backup_env)
                except PersistenceAccountMismatchError:
                    raise
                except PersistenceError:
                    backup_env = None

            valid_tmp_list: list[PersistedEnvelope] = []
            for tmp_p in self.find_primary_temp_files():
                try:
                    t_env = self._read_and_verify_file(
                        tmp_p,
                        expected_login=expected_login,
                        expected_server=expected_server,
                    )
                    valid_tmp_list.append(t_env)
                    valid_candidates.append(t_env)
                except PersistenceAccountMismatchError:
                    raise
                except PersistenceError:
                    continue

            primary_env: PersistedEnvelope | None = None
            primary_corrupted = False
            primary_error: str | None = None

            if primary_exists:
                try:
                    primary_env = self._read_and_verify_file(
                        self._primary_path,
                        expected_login=expected_login,
                        expected_server=expected_server,
                    )
                    self._validate_anti_rollback_and_chain(primary_env, backup_env)
                    valid_candidates.append(primary_env)
                except (PersistenceAccountMismatchError, PersistenceRollbackDetectedError):
                    raise
                except PersistenceError as exc:
                    primary_corrupted = True
                    primary_error = str(exc)

            highest_valid: PersistedEnvelope | None = None
            if valid_candidates:
                highest_valid = self._validate_candidate_chain(valid_candidates)

            if not primary_exists:
                recovery_needed = bool(backup_exists or tmp_files or self._last_known_generation > 0)
                return PersistenceLoadResult(
                    envelope=None,
                    backup_envelope=backup_env,
                    valid_tmp_envelopes=tuple(valid_tmp_list),
                    highest_valid_envelope=highest_valid,
                    interrupted_tmp_files=tmp_files,
                    primary_missing=True,
                    primary_corrupted=False,
                    primary_error="Primary file missing" if recovery_needed else None,
                    recovery_required=recovery_needed,
                )

            if primary_corrupted:
                return PersistenceLoadResult(
                    envelope=None,
                    backup_envelope=backup_env,
                    valid_tmp_envelopes=tuple(valid_tmp_list),
                    highest_valid_envelope=highest_valid,
                    interrupted_tmp_files=tmp_files,
                    primary_missing=False,
                    primary_corrupted=True,
                    primary_error=primary_error,
                    recovery_required=True,
                )

            if tmp_files:
                return PersistenceLoadResult(
                    envelope=primary_env,
                    backup_envelope=backup_env,
                    valid_tmp_envelopes=tuple(valid_tmp_list),
                    highest_valid_envelope=highest_valid,
                    interrupted_tmp_files=tmp_files,
                    primary_missing=False,
                    primary_corrupted=False,
                    primary_error=f"Interrupted temporary file(s) present: {', '.join(tmp_files)}",
                    recovery_required=True,
                )

            assert primary_env is not None
            self._last_known_generation = primary_env.generation
            self._last_known_checksum = primary_env.payload_checksum
            return PersistenceLoadResult(
                envelope=primary_env,
                backup_envelope=backup_env,
                valid_tmp_envelopes=(),
                highest_valid_envelope=primary_env,
                interrupted_tmp_files=(),
                primary_missing=False,
                primary_corrupted=False,
                primary_error=None,
                recovery_required=False,
            )

    def save(
        self,
        payload: dict[str, Any],
        *,
        account_login: int,
        server: str,
        now_utc: datetime | None = None,
    ) -> PersistedEnvelope:
        """Atomically persist `payload` at `generation + 1` under an exclusive OS file lock."""
        if account_login <= 0:
            raise ValueError(f"account_login must be > 0, got {account_login}.")
        cleaned_server = server.strip()
        if not cleaned_server:
            raise ValueError("server must be a non-empty string.")

        write_ts = (
            now_utc.astimezone(timezone.utc)
            if (now_utc is not None and now_utc.tzinfo is not None)
            else (now_utc.replace(tzinfo=timezone.utc) if now_utc is not None else datetime.now(timezone.utc))
        )

        with FileLock(self._lock_path, timeout_seconds=self._lock_timeout_seconds):
            current_env = self._load_unlocked(
                expected_login=account_login,
                expected_server=cleaned_server,
            )

            if current_env is not None:
                # Atomically copy current verified primary to .bak before advancing generation
                self._atomic_write_bytes(
                    self._backup_path,
                    self._primary_path.read_bytes(),
                )
                next_gen = current_env.generation + 1
                prev_checksum = current_env.payload_checksum
            else:
                next_gen = 1
                prev_checksum = GENESIS_CHECKSUM

            checksum = compute_envelope_checksum(
                schema_version=ENVELOPE_SCHEMA_VERSION,
                generation=next_gen,
                previous_checksum=prev_checksum,
                account_login=account_login,
                server=cleaned_server,
                payload=payload,
            )
            new_env = PersistedEnvelope(
                schema_version=ENVELOPE_SCHEMA_VERSION,
                generation=next_gen,
                previous_checksum=prev_checksum,
                payload_checksum=checksum,
                account_login=account_login,
                server=cleaned_server,
                written_at_utc=write_ts,
                payload=payload,
            )

            serialized = json.dumps(
                new_env.model_dump(mode="json"),
                indent=2,
                sort_keys=True,
                ensure_ascii=True,
            ).encode("utf-8")
            self._atomic_write_bytes(self._primary_path, serialized)

            self._last_known_generation = new_env.generation
            self._last_known_checksum = new_env.payload_checksum
            return new_env

    def commit_recovered_payload(
        self,
        payload: dict[str, Any],
        *,
        account_login: int,
        server: str,
        now_utc: datetime | None = None,
    ) -> PersistedEnvelope:
        """Atomically commit an explicitly reconciled recovery payload, advancing generation
        monotonically past the highest valid primary/backup/temp generation and clearing stale `.tmp.*` files.

        Fails closed (`PersistenceCorruptionError` / `PersistenceRollbackDetectedError`) if existing
        on-disk persistence files are all corrupted or violate monotonic hash-chain invariants.
        """
        if account_login <= 0:
            raise ValueError(f"account_login must be > 0, got {account_login}.")
        cleaned_server = server.strip()
        if not cleaned_server:
            raise ValueError("server must be a non-empty string.")

        write_ts = (
            now_utc.astimezone(timezone.utc)
            if (now_utc is not None and now_utc.tzinfo is not None)
            else (now_utc.replace(tzinfo=timezone.utc) if now_utc is not None else datetime.now(timezone.utc))
        )

        with FileLock(self._lock_path, timeout_seconds=self._lock_timeout_seconds):
            primary_exists = self._primary_path.exists()
            backup_exists = self._backup_path.exists()
            all_tmp_files = self.find_interrupted_temp_files()
            had_existing_artifacts = bool(
                primary_exists or backup_exists or all_tmp_files or self._last_known_generation > 0
            )

            valid_candidates: list[PersistedEnvelope] = []
            primary_env: PersistedEnvelope | None = None
            backup_env: PersistedEnvelope | None = None

            if primary_exists:
                try:
                    primary_env = self._read_and_verify_file(
                        self._primary_path,
                        expected_login=account_login,
                        expected_server=cleaned_server,
                    )
                    valid_candidates.append(primary_env)
                except PersistenceAccountMismatchError:
                    raise
                except PersistenceError:
                    primary_env = None

            if backup_exists:
                try:
                    backup_env = self._read_and_verify_file(
                        self._backup_path,
                        expected_login=account_login,
                        expected_server=cleaned_server,
                    )
                    valid_candidates.append(backup_env)
                except PersistenceAccountMismatchError:
                    raise
                except PersistenceError:
                    backup_env = None

            if primary_env is not None and backup_env is not None:
                self._validate_anti_rollback_and_chain(primary_env, backup_env)

            # Inspect every primary .tmp.* file so a valid higher-generation temporary envelope is never lost
            for tmp_path in self.find_primary_temp_files():
                try:
                    tmp_env = self._read_and_verify_file(
                        tmp_path,
                        expected_login=account_login,
                        expected_server=cleaned_server,
                    )
                    valid_candidates.append(tmp_env)
                except PersistenceAccountMismatchError:
                    raise
                except PersistenceError:
                    continue

            if not valid_candidates:
                if had_existing_artifacts:
                    raise PersistenceCorruptionError(
                        f"Cannot recover '{self._primary_path}': existing persistence files are corrupted "
                        "and no valid checksum-verified envelope exists. Refusing to reset generation to 1."
                    )
                next_gen = 1
                prev_checksum = GENESIS_CHECKSUM
            else:
                best_env = self._validate_candidate_chain(valid_candidates)
                if self._last_known_generation > 0 and best_env.generation < self._last_known_generation:
                    raise PersistenceRollbackDetectedError(
                        f"ROLLBACK DETECTED during recovery of '{self._primary_path}': highest valid on-disk "
                        f"generation ({best_env.generation}) is older than in-memory generation "
                        f"({self._last_known_generation})."
                    )
                best_bytes = json.dumps(
                    best_env.model_dump(mode="json"),
                    indent=2,
                    sort_keys=True,
                    ensure_ascii=True,
                ).encode("utf-8")
                self._atomic_write_bytes(self._backup_path, best_bytes)
                next_gen = best_env.generation + 1
                prev_checksum = best_env.payload_checksum

            checksum = compute_envelope_checksum(
                schema_version=ENVELOPE_SCHEMA_VERSION,
                generation=next_gen,
                previous_checksum=prev_checksum,
                account_login=account_login,
                server=cleaned_server,
                payload=payload,
            )
            new_env = PersistedEnvelope(
                schema_version=ENVELOPE_SCHEMA_VERSION,
                generation=next_gen,
                previous_checksum=prev_checksum,
                payload_checksum=checksum,
                account_login=account_login,
                server=cleaned_server,
                written_at_utc=write_ts,
                payload=payload,
            )
            serialized = json.dumps(
                new_env.model_dump(mode="json"),
                indent=2,
                sort_keys=True,
                ensure_ascii=True,
            ).encode("utf-8")
            self._atomic_write_bytes(self._primary_path, serialized)

            # Remove lingering interrupted temporary files (.tmp.* and .bak.tmp.*) after atomic replacement
            for tmp_file in self.find_interrupted_temp_files():
                try:
                    tmp_file.unlink()
                except OSError:
                    pass

            self._last_known_generation = new_env.generation
            self._last_known_checksum = new_env.payload_checksum
            return new_env

    @staticmethod
    def _atomic_write_bytes(target_path: Path, data: bytes) -> None:
        """Write `data` to `<target>.tmp.<pid>.<uuid>`, flush + `os.fsync`, and atomically `os.replace`."""
        target_path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = target_path.parent / f"{target_path.name}.tmp.{os.getpid()}.{uuid.uuid4().hex}"
        try:
            with open(tmp_path, "wb") as f:
                f.write(data)
                f.flush()
                os.fsync(f.fileno())
            os.replace(tmp_path, target_path)
        except Exception as exc:
            try:
                if tmp_path.exists():
                    tmp_path.unlink()
            except OSError:
                pass
            raise PersistenceError(f"Atomic file write failed for '{target_path}': {exc}") from exc
