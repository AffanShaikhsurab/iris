import random
from dataclasses import replace

import pytest
import torch

from binary_llm.domain import AmbiguityStatus, ScaleRung, seed_ambiguity_register
from binary_llm.math import ProgressionScheduleConfig
from binary_llm.orchestration import (
    COMPLETE_CHECKPOINT_STATE_KEYS,
    ProgressivePlanKind,
    build_progressive_phase_plan,
    capture_progressive_boundary_checkpoint,
    load_progressive_checkpoint,
    restore_progressive_checkpoint,
    save_progressive_checkpoint,
)
from binary_llm.orchestration.corpus import PartitionRef


def _partitions() -> tuple[PartitionRef, ...]:
    return tuple(
        PartitionRef(
            phase_index=index,
            partition_id=f"partition-{index}",
            manifest_hash=f"{index + 1:064x}",
            record_ids=(f"record-{index}",),
            content_hashes=(f"{index + 101:064x}",),
            semantic_family_ids=(f"semantic-{index}",),
            split_family_ids=(f"split-{index}",),
            group_ids=(f"group-{index}",),
            token_count=index + 10,
            seed=17,
        )
        for index in range(20)
    )


def _resolution(candidate: str = "zero_based"):
    entry = next(
        item
        for item in seed_ambiguity_register("ambiguities", "v1").entries
        if item.ambiguity_id == "progressive.phase_index"
    )
    rejected = tuple(
        item.candidate_id for item in entry.candidates if item.candidate_id != candidate
    )
    return replace(
        entry,
        matched_protocol="matched-phase-index-screening",
        decision_rule="Select only after stability and continuation floors pass.",
        confidence_method="paired-bootstrap",
        required_floors=("finite", "continuation"),
        scale_scope=(ScaleRung.SMALL,),
        status=AmbiguityStatus.RESOLVED,
        selected_candidate=candidate,
        evidence_refs=("phase-index-evidence",),
        rejected_candidates=rejected,
    )


def _plan(*, test_phase_count: int | None = None):
    return build_progressive_phase_plan(
        _partitions(),
        ProgressionScheduleConfig("paper_exponential", "0..19"),
        _resolution(),
        scale_rung=ScaleRung.SMALL,
        test_phase_count=test_phase_count,
    )


def _trained_model_and_optimizer():
    model = torch.nn.Linear(3, 2)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)
    loss = model(torch.ones(2, 3)).square().mean()
    loss.backward()
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    return model, optimizer


def test_production_plan_binds_all_twenty_partitions_to_explicit_schedule_indices() -> None:
    plan = _plan()

    assert plan.kind is ProgressivePlanKind.PRODUCTION
    assert plan.phase_count == 20
    assert tuple(phase.schedule_index for phase in plan.phases) == tuple(range(20))
    assert tuple(phase.partition_phase_index for phase in plan.phases) == tuple(range(20))
    assert tuple(phase.partition_hash for phase in plan.phases) == tuple(
        partition.manifest_hash for partition in _partitions()
    )
    assert plan.phase_index_resolution.selected_candidate == "zero_based"
    assert plan.phases[0].progression_parameter == 0.0


def test_short_test_plan_is_a_deterministic_prefix_of_the_production_plan() -> None:
    production = _plan()
    first = _plan(test_phase_count=3)
    second = _plan(test_phase_count=3)

    assert first.kind is ProgressivePlanKind.DETERMINISTIC_TEST
    assert first.phases == production.phases[:3]
    assert first.source_partition_hashes == production.source_partition_hashes
    assert first.plan_hash == second.plan_hash
    assert first.plan_hash != production.plan_hash


def test_planning_fails_closed_for_unresolved_mismatched_or_unscoped_index_choices() -> None:
    open_entry = next(
        item
        for item in seed_ambiguity_register("ambiguities", "v1").entries
        if item.ambiguity_id == "progressive.phase_index"
    )
    with pytest.raises(ValueError, match="explicitly resolved"):
        build_progressive_phase_plan(
            _partitions(), ProgressionScheduleConfig("paper_exponential", "0..19"),
            open_entry, scale_rung=ScaleRung.SMALL,
        )
    with pytest.raises(ValueError, match="conflicts"):
        build_progressive_phase_plan(
            _partitions(), ProgressionScheduleConfig("paper_exponential", "1..20"),
            _resolution("zero_based"), scale_rung=ScaleRung.SMALL,
        )
    with pytest.raises(ValueError, match="scale rung"):
        build_progressive_phase_plan(
            _partitions(), ProgressionScheduleConfig("paper_exponential", "0..19"),
            _resolution(), scale_rung=ScaleRung.IRIS,
        )


def test_every_test_phase_boundary_captures_complete_resume_state() -> None:
    plan = _plan(test_phase_count=3)
    model, optimizer = _trained_model_and_optimizer()

    checkpoints = tuple(
        capture_progressive_boundary_checkpoint(
            checkpoint_id=f"checkpoint-{boundary}",
            parent_checkpoint_id="stage1-checkpoint",
            stage1_parent_checkpoint_id="stage1-checkpoint",
            plan=plan,
            completed_phase_count=boundary,
            optimizer_step=boundary * 4,
            model=model,
            optimizer=optimizer,
        )
        for boundary in range(1, plan.phase_count + 1)
    )

    for boundary, checkpoint in enumerate(checkpoints, start=1):
        assert checkpoint.complete
        assert checkpoint.schema_version == 2
        assert checkpoint.stage1_parent_checkpoint_id == "stage1-checkpoint"
        assert set(checkpoint.state_hashes) == COMPLETE_CHECKPOINT_STATE_KEYS
        assert checkpoint.schedule_state.completed_phase_count == boundary
        assert checkpoint.schedule_state.completed_schedule_index == boundary - 1
        assert checkpoint.data_cursor.completed_partition_hashes == tuple(
            phase.partition_hash for phase in plan.phases[:boundary]
        )
        assert checkpoint.data_cursor.record_offset == 0
        assert checkpoint.data_cursor.token_offset == 0
        assert checkpoint.consumed_tokens == sum(
            phase.partition_token_count for phase in plan.phases[:boundary]
        )
        expected_next = None if boundary == plan.phase_count else plan.phases[boundary].partition_hash
        assert checkpoint.data_cursor.next_partition_hash == expected_next


def test_atomic_checkpoint_round_trip_restores_model_optimizer_and_rng(tmp_path) -> None:
    random.seed(101)
    torch.manual_seed(101)
    plan = _plan(test_phase_count=2)
    model, optimizer = _trained_model_and_optimizer()
    checkpoint = capture_progressive_boundary_checkpoint(
        checkpoint_id="phase-1-complete",
        parent_checkpoint_id="stage1-checkpoint",
        stage1_parent_checkpoint_id="stage1-checkpoint",
        plan=plan,
        completed_phase_count=1,
        optimizer_step=7,
        model=model,
        optimizer=optimizer,
    )
    expected_python_random = random.random()
    expected_torch_random = torch.rand(4)
    saved_parameters = {
        name: value.detach().clone() for name, value in model.state_dict().items()
    }

    destination = save_progressive_checkpoint(tmp_path / "nested" / "phase.pt", checkpoint)
    for parameter in model.parameters():
        parameter.data.add_(100.0)
    random.random()
    torch.rand(8)

    loaded = load_progressive_checkpoint(destination, plan=plan)
    restore_progressive_checkpoint(loaded, plan=plan, model=model, optimizer=optimizer)

    for name, value in model.state_dict().items():
        torch.testing.assert_close(value, saved_parameters[name])
    assert random.random() == expected_python_random
    torch.testing.assert_close(torch.rand(4), expected_torch_random)
    assert loaded.optimizer_state["state"]
    assert loaded.stage1_parent_checkpoint_id == "stage1-checkpoint"
    assert not tuple(destination.parent.glob("*.tmp"))


def test_stage1_root_is_content_bound_and_schema_v1_is_rejected() -> None:
    plan = _plan(test_phase_count=2)
    model, optimizer = _trained_model_and_optimizer()
    checkpoint = capture_progressive_boundary_checkpoint(
        checkpoint_id="phase-2-complete",
        parent_checkpoint_id="phase-1-complete",
        stage1_parent_checkpoint_id="stage1-a",
        plan=plan,
        completed_phase_count=2,
        optimizer_step=8,
        model=model,
        optimizer=optimizer,
    )

    with pytest.raises(ValueError, match="state hashes"):
        replace(checkpoint, stage1_parent_checkpoint_id="stage1-b")
    with pytest.raises(ValueError, match="schema_version"):
        replace(checkpoint, schema_version=1)
