from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from binary_llm.domain.errors import SealViolation
from binary_llm.domain.models import ArtifactRef, ModelIdentity, SealedBaselineRef
from binary_llm.orchestration import (
    SealMismatchReason,
    SealedAsset,
    SealedAssetCategory,
    SealedInventory,
    allocate_after_seal_verification,
    verify_sealed_inventory,
)


def _digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _write_asset(
    root: Path,
    *,
    asset_id: str,
    category: SealedAssetCategory,
    data: bytes,
    roles: tuple[str, ...] = (),
) -> SealedAsset:
    relative = f"{category.value}/{asset_id}.bin"
    path = root / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)
    reference = ArtifactRef(
        artifact_id=asset_id,
        kind=category.value,
        sha256=_digest(data),
        bytes=len(data),
        media_type="application/octet-stream",
    )
    return SealedAsset(reference, category, relative, roles)


def _sealed_fixture(root: Path) -> SealedInventory:
    assets: list[SealedAsset] = []
    oracles: list[SealedBaselineRef] = []
    for role in ("base", "adapter", "bf16", "q4"):
        role_tuple = (role,)
        weight = _write_asset(
            root, asset_id=f"{role}-weights", category=SealedAssetCategory.WEIGHTS,
            data=f"{role}:weights".encode(), roles=role_tuple,
        )
        tokenizer = _write_asset(
            root, asset_id=f"{role}-tokenizer", category=SealedAssetCategory.TOKENIZERS,
            data=f"{role}:tokenizer".encode(), roles=role_tuple,
        )
        template = _write_asset(
            root, asset_id=f"{role}-template", category=SealedAssetCategory.TEMPLATES,
            data=f"{role}:template".encode(), roles=role_tuple,
        )
        config = _write_asset(
            root, asset_id=f"{role}-config", category=SealedAssetCategory.CONFIGURATIONS,
            data=f"{role}:config".encode(), roles=role_tuple,
        )
        output = _write_asset(
            root, asset_id=f"{role}-output", category=SealedAssetCategory.BASELINE_OUTPUTS,
            data=f"{role}:output".encode(), roles=role_tuple,
        )
        evaluator = _write_asset(
            root, asset_id=f"{role}-evaluator", category=SealedAssetCategory.EVALUATORS,
            data=f"{role}:evaluator".encode(), roles=role_tuple,
        )
        assets.extend((weight, tokenizer, template, config, output, evaluator))
        oracles.append(
            SealedBaselineRef(
                baseline_id=f"iris-{role}",
                role=role,
                model=ModelIdentity(
                    model_id=f"model-{role}", revision="pinned-revision",
                    tokenizer_revision="pinned-tokenizer", architecture="test",
                    parameter_count=100, config_hash=config.asset.sha256,
                    tokenizer_hashes=(tokenizer.asset.sha256,),
                    template_hashes=(template.asset.sha256,),
                    weight_hashes=(weight.asset.sha256,), tied_weights=False,
                ),
                artifact_refs=(weight.asset,), evaluator_revision=evaluator.asset.sha256,
                baseline_output_refs=(output.asset,), sealed_at="2026-01-01T00:00:00Z",
            )
        )
    assets.append(
        _write_asset(
            root, asset_id="frozen-dataset", category=SealedAssetCategory.DATASETS,
            data=b"frozen dataset", roles=("base", "adapter", "bf16", "q4"),
        )
    )
    return SealedInventory("iris-sealed-v1", tuple(assets), tuple(oracles))


def _file_snapshot(root: Path) -> dict[str, tuple[bytes, int]]:
    return {
        path.relative_to(root).as_posix(): (path.read_bytes(), path.stat().st_mtime_ns)
        for path in root.rglob("*") if path.is_file()
    }


def test_verifies_every_category_without_mutation_and_preserves_oracle_identity(tmp_path: Path):
    inventory = _sealed_fixture(tmp_path)
    before = _file_snapshot(tmp_path)

    report = verify_sealed_inventory(inventory, tmp_path)

    assert report.compute_allowed
    assert not report.mismatches
    assert set(report.verified_asset_ids) == {item.asset.artifact_id for item in inventory.assets}
    assert [item[0] for item in report.oracle_identities] == ["adapter", "base", "bf16", "q4"]
    assert len({item[1] for item in report.oracle_identities}) == 4
    assert len({item[2] for item in report.oracle_identities}) == 4
    assert _file_snapshot(tmp_path) == before


@pytest.mark.parametrize("category", list(SealedAssetCategory))
def test_each_required_asset_category_blocks_compute_on_hash_drift(
    tmp_path: Path, category: SealedAssetCategory,
):
    inventory = _sealed_fixture(tmp_path)
    target = next(item for item in inventory.assets if item.category is category)
    path = tmp_path / target.relative_path
    original = path.read_bytes()
    path.write_bytes(bytes([original[0] ^ 1]) + original[1:])
    mutated = path.read_bytes()

    report = verify_sealed_inventory(inventory, tmp_path)

    assert not report.compute_allowed
    assert tuple(item.asset_id for item in report.mismatches) == (target.asset.artifact_id,)
    assert report.mismatches[0].reasons == (SealMismatchReason.HASH,)
    assert path.read_bytes() == mutated


def test_reports_all_mismatches_and_never_invokes_allocator(tmp_path: Path):
    inventory = _sealed_fixture(tmp_path)
    targets = inventory.assets[:3]
    (tmp_path / targets[0].relative_path).write_bytes(b"changed-size-and-hash")
    (tmp_path / targets[1].relative_path).unlink()
    original = (tmp_path / targets[2].relative_path).read_bytes()
    (tmp_path / targets[2].relative_path).write_bytes(bytes([original[0] ^ 1]) + original[1:])
    observed_before = _file_snapshot(tmp_path)
    calls: list[object] = []

    with pytest.raises(SealViolation) as raised:
        allocate_after_seal_verification(
            inventory, tmp_path, lambda report: calls.append(report),
        )

    assert not calls
    context = raised.value.context
    assert [item["asset_id"] for item in context["mismatches"]] == [
        target.asset.artifact_id for target in targets
    ]
    assert context["mismatches"][0]["reasons"] == ["size_mismatch", "hash_mismatch"]
    assert context["mismatches"][1]["reasons"] == ["missing"]
    assert context["mismatches"][2]["reasons"] == ["hash_mismatch"]
    assert len(context["oracle_identities"]) == 4
    assert _file_snapshot(tmp_path) == observed_before


def test_successful_preflight_allocates_only_after_complete_verification(tmp_path: Path):
    inventory = _sealed_fixture(tmp_path)
    allocated: list[bool] = []

    result = allocate_after_seal_verification(
        inventory,
        tmp_path,
        lambda report: allocated.append(report.compute_allowed) or "allocated",
    )

    assert result == "allocated"
    assert allocated == [True]


def test_inventory_rejects_aliased_or_incomplete_oracle_roles(tmp_path: Path):
    inventory = _sealed_fixture(tmp_path)

    with pytest.raises(ValueError, match="exactly one base, adapter, bf16, and q4"):
        SealedInventory(inventory.inventory_id, inventory.assets, inventory.oracles[:-1])

    duplicate_identity = (
        inventory.oracles[0], inventory.oracles[0], inventory.oracles[2], inventory.oracles[3]
    )
    with pytest.raises(ValueError):
        SealedInventory(inventory.inventory_id, inventory.assets, duplicate_identity)
