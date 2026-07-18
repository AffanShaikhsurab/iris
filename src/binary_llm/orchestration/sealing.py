"""Read-only verification of sealed experiment inputs before compute allocation."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from enum import Enum
from pathlib import Path, PurePosixPath
from typing import Callable, TypeVar

from binary_llm.domain.errors import Retryability, SealViolation
from binary_llm.domain.identity import identify_content
from binary_llm.domain.models import ArtifactRef, SealedBaselineRef

_SHA256_LENGTH = 64
_REQUIRED_ORACLE_ROLES = ("base", "adapter", "bf16", "q4")
_ResultT = TypeVar("_ResultT")


class SealedAssetCategory(str, Enum):
    """Required classes of immutable input bytes."""

    WEIGHTS = "weights"
    TOKENIZERS = "tokenizers"
    TEMPLATES = "templates"
    CONFIGURATIONS = "configurations"
    DATASETS = "datasets"
    EVALUATORS = "evaluators"
    BASELINE_OUTPUTS = "baseline_outputs"


class SealMismatchReason(str, Enum):
    MISSING = "missing"
    NOT_A_FILE = "not_a_file"
    PATH_ESCAPE = "path_escape"
    SIZE = "size_mismatch"
    HASH = "hash_mismatch"
    READ_ERROR = "read_error"


def _sha256_file(path: Path) -> str:
    """Match the established Iris streaming file-hash contract without adapter imports."""

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _is_sha256(value: str) -> bool:
    return len(value) == _SHA256_LENGTH and all(character in "0123456789abcdef" for character in value)


@dataclass(frozen=True, slots=True)
class SealedAsset:
    """A required file bound to its sealed artifact identity and oracle roles."""

    asset: ArtifactRef
    category: SealedAssetCategory
    relative_path: str
    oracle_roles: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not isinstance(self.category, SealedAssetCategory):
            raise TypeError("category must be a SealedAssetCategory")
        if not _is_sha256(self.asset.sha256):
            raise ValueError("sealed asset sha256 must be a lowercase 64-character digest")
        path = PurePosixPath(self.relative_path)
        if not self.relative_path or path.is_absolute() or path.as_posix() in {".", ""}:
            raise ValueError("sealed asset path must be a non-empty relative POSIX path")
        if len(set(self.oracle_roles)) != len(self.oracle_roles):
            raise ValueError("sealed asset oracle roles must be unique")
        if any(role not in _REQUIRED_ORACLE_ROLES for role in self.oracle_roles):
            raise ValueError("sealed asset oracle role must be base, adapter, bf16, or q4")


@dataclass(frozen=True, slots=True)
class SealedInventory:
    """Complete pre-compute inventory and four independent regression oracles."""

    inventory_id: str
    assets: tuple[SealedAsset, ...]
    oracles: tuple[SealedBaselineRef, ...]

    def __post_init__(self) -> None:
        if not self.inventory_id:
            raise ValueError("inventory_id must not be empty")
        asset_ids = tuple(item.asset.artifact_id for item in self.assets)
        if len(set(asset_ids)) != len(asset_ids):
            raise ValueError("sealed inventory asset IDs must be unique")
        paths = tuple(PurePosixPath(item.relative_path).as_posix() for item in self.assets)
        if len(set(paths)) != len(paths):
            raise ValueError("sealed inventory paths must be unique")
        categories = {item.category for item in self.assets}
        missing_categories = set(SealedAssetCategory) - categories
        if missing_categories:
            missing = ", ".join(sorted(item.value for item in missing_categories))
            raise ValueError(f"sealed inventory lacks required asset categories: {missing}")

        roles = tuple(item.role for item in self.oracles)
        if tuple(sorted(roles)) != tuple(sorted(_REQUIRED_ORACLE_ROLES)):
            raise ValueError("sealed inventory requires exactly one base, adapter, bf16, and q4 oracle")
        baseline_ids = tuple(item.baseline_id for item in self.oracles)
        if len(set(baseline_ids)) != len(baseline_ids):
            raise ValueError("sealed oracle baseline IDs must be unique")
        identities = tuple(
            identify_content(item.to_dict(), kind="sealed-baseline").value for item in self.oracles
        )
        if len(set(identities)) != len(identities):
            raise ValueError("sealed regression oracles must have independent content identities")

        by_role = {
            role: tuple(asset for asset in self.assets if role in asset.oracle_roles)
            for role in _REQUIRED_ORACLE_ROLES
        }
        for oracle in self.oracles:
            role_assets = by_role[oracle.role]
            expected_by_category = {
                SealedAssetCategory.WEIGHTS: set(oracle.model.weight_hashes),
                SealedAssetCategory.TOKENIZERS: set(oracle.model.tokenizer_hashes),
                SealedAssetCategory.TEMPLATES: set(oracle.model.template_hashes),
                SealedAssetCategory.CONFIGURATIONS: {oracle.model.config_hash},
            }
            for category, expected_hashes in expected_by_category.items():
                available = {item.asset.sha256 for item in role_assets if item.category is category}
                if not expected_hashes <= available:
                    raise ValueError(
                        f"{oracle.role} oracle is missing {category.value} hashes from the sealed inventory"
                    )
            available_refs = {item.asset.artifact_id: item.asset for item in role_assets}
            for reference in oracle.artifact_refs:
                if available_refs.get(reference.artifact_id) != reference:
                    raise ValueError(f"{oracle.role} oracle artifact reference is not sealed exactly")
            output_refs = {
                item.asset.artifact_id: item.asset
                for item in role_assets
                if item.category is SealedAssetCategory.BASELINE_OUTPUTS
            }
            for reference in oracle.baseline_output_refs:
                if output_refs.get(reference.artifact_id) != reference:
                    raise ValueError(f"{oracle.role} baseline output reference is not sealed exactly")
            if not any(item.category is SealedAssetCategory.EVALUATORS for item in role_assets):
                raise ValueError(f"{oracle.role} oracle requires a sealed evaluator")

    @property
    def oracle_identities(self) -> tuple[tuple[str, str, str], ...]:
        """Return role, baseline ID, and immutable content identity without coalescing roles."""

        return tuple(
            (
                oracle.role,
                oracle.baseline_id,
                identify_content(oracle.to_dict(), kind="sealed-baseline").value,
            )
            for oracle in sorted(self.oracles, key=lambda item: item.role)
        )


@dataclass(frozen=True, slots=True)
class SealMismatch:
    asset_id: str
    category: SealedAssetCategory
    relative_path: str
    reasons: tuple[SealMismatchReason, ...]
    expected_sha256: str
    observed_sha256: str | None
    expected_bytes: int
    observed_bytes: int | None
    detail: str | None = None

    def to_dict(self) -> dict[str, object]:
        return {
            "asset_id": self.asset_id,
            "category": self.category.value,
            "relative_path": self.relative_path,
            "reasons": [reason.value for reason in self.reasons],
            "expected_sha256": self.expected_sha256,
            "observed_sha256": self.observed_sha256,
            "expected_bytes": self.expected_bytes,
            "observed_bytes": self.observed_bytes,
            "detail": self.detail,
        }


@dataclass(frozen=True, slots=True)
class SealVerificationReport:
    inventory_id: str
    verified_asset_ids: tuple[str, ...]
    mismatches: tuple[SealMismatch, ...]
    oracle_identities: tuple[tuple[str, str, str], ...]

    @property
    def compute_allowed(self) -> bool:
        return not self.mismatches

    def require_compute_allowed(self) -> None:
        if self.compute_allowed:
            return
        raise SealViolation(
            "sealed inventory verification failed before compute allocation",
            retryability=Retryability.AFTER_REMEDIATION,
            code="seal.inventory_mismatch",
            affected_ids={
                "artifact_ids": tuple(item.asset_id for item in self.mismatches),
                "baseline_ids": tuple(item[1] for item in self.oracle_identities),
            },
            context={
                "inventory_id": self.inventory_id,
                "mismatches": [item.to_dict() for item in self.mismatches],
                "oracle_identities": [list(item) for item in self.oracle_identities],
            },
        )


def _verify_asset(root: Path, sealed: SealedAsset) -> SealMismatch | None:
    expected = sealed.asset
    candidate = (root / Path(*PurePosixPath(sealed.relative_path).parts)).resolve()
    try:
        candidate.relative_to(root)
    except ValueError:
        return SealMismatch(
            expected.artifact_id, sealed.category, sealed.relative_path,
            (SealMismatchReason.PATH_ESCAPE,), expected.sha256, None, expected.bytes, None,
        )
    if not candidate.exists():
        return SealMismatch(
            expected.artifact_id, sealed.category, sealed.relative_path,
            (SealMismatchReason.MISSING,), expected.sha256, None, expected.bytes, None,
        )
    if not candidate.is_file():
        return SealMismatch(
            expected.artifact_id, sealed.category, sealed.relative_path,
            (SealMismatchReason.NOT_A_FILE,), expected.sha256, None, expected.bytes, None,
        )
    try:
        observed_bytes = candidate.stat().st_size
        observed_sha256 = _sha256_file(candidate)
    except OSError as error:
        return SealMismatch(
            expected.artifact_id, sealed.category, sealed.relative_path,
            (SealMismatchReason.READ_ERROR,), expected.sha256, None, expected.bytes, None,
            detail=f"{type(error).__name__}: {error}",
        )
    reasons = tuple(
        reason
        for condition, reason in (
            (observed_bytes != expected.bytes, SealMismatchReason.SIZE),
            (observed_sha256 != expected.sha256, SealMismatchReason.HASH),
        )
        if condition
    )
    if not reasons:
        return None
    return SealMismatch(
        expected.artifact_id, sealed.category, sealed.relative_path, reasons,
        expected.sha256, observed_sha256, expected.bytes, observed_bytes,
    )


def verify_sealed_inventory(inventory: SealedInventory, root: Path) -> SealVerificationReport:
    """Hash every required asset, aggregate mismatches, and never write source bytes."""

    resolved_root = root.resolve()
    mismatches: list[SealMismatch] = []
    verified: list[str] = []
    for sealed in inventory.assets:
        mismatch = _verify_asset(resolved_root, sealed)
        if mismatch is None:
            verified.append(sealed.asset.artifact_id)
        else:
            mismatches.append(mismatch)
    return SealVerificationReport(
        inventory_id=inventory.inventory_id,
        verified_asset_ids=tuple(verified),
        mismatches=tuple(mismatches),
        oracle_identities=inventory.oracle_identities,
    )


def allocate_after_seal_verification(
    inventory: SealedInventory,
    root: Path,
    allocator: Callable[[SealVerificationReport], _ResultT],
) -> _ResultT:
    """Invoke compute allocation only after all sealed hashes and sizes pass."""

    report = verify_sealed_inventory(inventory, root)
    report.require_compute_allowed()
    return allocator(report)
