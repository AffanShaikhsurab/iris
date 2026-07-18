from __future__ import annotations

from dataclasses import replace

import pytest
import torch
from torch import nn
from torch.nn import functional as F

from binary_llm.adapters import (
    ActiveRepresentation,
    BinaryLinear,
    BinaryLinearConfig,
    TrainingActivationMode,
)
from binary_llm.adapters.contracts import TensorRole
from binary_llm.domain import ManifestError
from binary_llm.math import (
    AnalyticalScaleGradient,
    OperatorPrecision,
    ProgressiveOperatorConfig,
    ZeroSignRule,
    binary_sign,
)
from binary_llm.orchestration import (
    FilesystemArtifactStore,
    Stage1TransitionKind,
    apply_stage1_transition,
    build_stage1_transition,
    load_stage1_transition,
    persist_stage1_transition,
)


def _config(mode=TrainingActivationMode.EXPLICIT_ACTIVATION_TRANSFORM):
    return BinaryLinearConfig(
        scale_parameterization="positive_exp",
        training_activation_mode=mode,
        progressive_operator=ProgressiveOperatorConfig(
            precision=OperatorPrecision.FLOAT32,
            analytical_scale_gradient=AnalyticalScaleGradient.DETACHED,
        ),
        zero_sign_rule=ZeroSignRule.POSITIVE,
    )


def _model(mode=TrainingActivationMode.EXPLICIT_ACTIVATION_TRANSFORM):
    torch.manual_seed(71)
    source = nn.Linear(3, 2, bias=True)
    model = nn.Sequential(
        BinaryLinear.from_linear(
            source,
            tensor_name="layer.weight",
            semantic_role=TensorRole.FFN_UP,
            config=_config(mode),
        )
    )
    with torch.no_grad():
        model[0].input_scale.raw_scale.copy_(
            torch.log(torch.tensor([0.5, 2.0, 4.0]))
        )
    return model


def test_explicit_stage1_algebra_is_exact_and_weight_only_is_modified() -> None:
    inputs = torch.tensor([[0.25, -1.5, 2.0]])
    explicit = _model()
    module = explicit[0]
    scale = module.input_scale()
    expected = F.linear(
        scale * inputs,
        binary_sign(module.weight / scale, ZeroSignRule.POSITIVE),
        module.bias,
    )
    torch.testing.assert_close(explicit(inputs), expected, rtol=0, atol=0)

    weight_only = _model(TrainingActivationMode.WEIGHT_ONLY)
    assert not torch.equal(weight_only(inputs), expected)
    transition = build_stage1_transition(
        model=weight_only,
        source_checkpoint_id="stage1",
        source_run_id="run",
    )
    assert transition.kind is Stage1TransitionKind.MODIFIED_WEIGHT_ONLY
    with pytest.raises(ManifestError, match="modified Stage 1 arm"):
        apply_stage1_transition(
            weight_only,
            transition,
            expected_source_checkpoint_id="stage1",
        )


def test_transition_is_deterministic_persistent_and_applies_exactly(tmp_path) -> None:
    first = _model()
    second = _model()
    transition = build_stage1_transition(
        model=first,
        source_checkpoint_id="stage1",
        source_run_id="run",
        parent_artifact_id="dense-parent",
    )
    equivalent = build_stage1_transition(
        model=second,
        source_checkpoint_id="stage1",
        source_run_id="run",
        parent_artifact_id="dense-parent",
    )
    assert transition.payload == equivalent.payload
    assert transition.artifact_ref == equivalent.artifact_ref
    assert transition.module_names == ("0",)

    store = FilesystemArtifactStore(tmp_path)
    persisted = persist_stage1_transition(transition, store)
    reopened = load_stage1_transition(FilesystemArtifactStore(tmp_path), persisted)
    assert reopened == transition

    module = second[0]
    dense_before = module.dense_reference_weight.clone()
    expected = module.transformed_weight().detach().clone()
    receipt = apply_stage1_transition(
        second,
        reopened,
        expected_source_checkpoint_id="stage1",
        expected_parent_artifact_id="dense-parent",
    )
    assert receipt.model_identity == id(second)
    torch.testing.assert_close(module.weight, expected, rtol=0, atol=0)
    torch.testing.assert_close(module.input_scale(), torch.ones(3), rtol=0, atol=0)
    torch.testing.assert_close(module.transformed_weight(), expected, rtol=0, atol=0)
    torch.testing.assert_close(module.dense_reference_weight, dense_before, rtol=0, atol=0)
    module.set_active_representation(
        ActiveRepresentation.PROGRESSIVE, progression_parameter=0.0
    )
    analytical = expected.abs().mean(dim=1)
    torch.testing.assert_close(
        module.effective_weight(),
        analytical.unsqueeze(1) * expected / analytical.unsqueeze(1),
    )


@pytest.mark.parametrize("mismatch", ("checkpoint", "parent", "source", "scale"))
def test_transition_failures_do_not_partially_mutate_model(mismatch) -> None:
    source = _model()
    transition = build_stage1_transition(
        model=source,
        source_checkpoint_id="stage1",
        source_run_id="run",
        parent_artifact_id="parent",
    )
    target = _model()
    if mismatch == "source":
        with torch.no_grad():
            target[0].weight[0, 0].add_(1)
    elif mismatch == "scale":
        with torch.no_grad():
            target[0].input_scale.raw_scale[0].add_(0.25)
    before = {name: value.clone() for name, value in target.state_dict().items()}
    with pytest.raises(ManifestError):
        apply_stage1_transition(
            target,
            transition,
            expected_source_checkpoint_id="wrong" if mismatch == "checkpoint" else "stage1",
            expected_parent_artifact_id="wrong" if mismatch == "parent" else "parent",
        )
    for name, value in target.state_dict().items():
        torch.testing.assert_close(value, before[name], rtol=0, atol=0)


def test_transition_rejects_tampered_payload_and_reference() -> None:
    model = _model()
    transition = build_stage1_transition(
        model=model,
        source_checkpoint_id="stage1",
        source_run_id="run",
    )
    tampered = replace(
        transition,
        payload=transition.payload[:-1] + bytes([transition.payload[-1] ^ 1]),
    )
    with pytest.raises(ManifestError, match="ArtifactRef integrity"):
        apply_stage1_transition(
            model, tampered, expected_source_checkpoint_id="stage1"
        )
