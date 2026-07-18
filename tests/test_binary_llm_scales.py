from __future__ import annotations

import math

import pytest
import torch
from torch import nn

from binary_llm.domain.errors import NumericalFailure, Retryability
from binary_llm.math import (
    AnalyticalScaleGradient,
    DualScaleState,
    InputChannelScale,
    ScaleParameterization,
    ScaleRole,
    Stage1ScaleState,
    ZeroSignRule,
    analytical_row_scales,
    binary_reconstruction_error,
    binary_sign,
    diagnose_finite_state,
    named_trainable_scales,
    scale_trainability_report,
    stage1_diagnostics,
    transformed_weight,
)


@pytest.mark.parametrize(
    "parameterization,kwargs",
    [
        (ScaleParameterization.POSITIVE_EXP, {}),
        (ScaleParameterization.POSITIVE_SOFTPLUS, {"epsilon": 1e-5}),
        (ScaleParameterization.SIGNED_CLAMP, {"minimum_magnitude": 1e-5}),
    ],
)
def test_input_channel_parameterizations_create_one_declared_scale_per_channel(
    parameterization: ScaleParameterization,
    kwargs: dict[str, float],
):
    scales = InputChannelScale(
        4,
        parameterization=parameterization,
        initial_value=1.0,
        dtype=torch.float64,
        **kwargs,
    )

    assert scales().shape == (4,)
    torch.testing.assert_close(scales(), torch.ones(4, dtype=torch.float64))
    declared = scales.declared_scale_parameters()
    assert declared == (("raw_scale", ScaleRole.STAGE1_INPUT, scales.raw_scale),)
    assert tuple(scales.parameters()) == (scales.raw_scale,)


def test_parameterizations_enforce_positivity_and_signed_magnitude_floor():
    positive = InputChannelScale(2, parameterization="positive_exp")
    signed = InputChannelScale(
        2,
        parameterization="signed_clamp",
        minimum_magnitude=0.25,
    )
    with torch.no_grad():
        positive.raw_scale.copy_(torch.tensor([-100.0, 2.0]))
        signed.raw_scale.copy_(torch.tensor([0.0, -0.1]))

    assert torch.all(positive() > 0)
    torch.testing.assert_close(signed(), torch.tensor([0.25, -0.25]))
    with pytest.raises(ValueError, match="initial_value > epsilon"):
        InputChannelScale(
            2,
            parameterization="positive_softplus",
            initial_value=1e-6,
            epsilon=1e-6,
        )


def test_stage1_state_preserves_dense_source_and_uses_binary_transformed_weight():
    original = torch.tensor([[2.0, -4.0, 0.0], [-6.0, 8.0, -10.0]])
    expected_source = original.clone()
    state = Stage1ScaleState(original, parameterization="positive_exp")
    with torch.no_grad():
        state.input_scale.raw_scale.copy_(torch.log(torch.tensor([2.0, 4.0, 5.0])))
        original.fill_(99.0)

    expected_transformed = torch.tensor([[1.0, -1.0, 0.0], [-3.0, 2.0, -2.0]])
    torch.testing.assert_close(state.dense_source, expected_source)
    torch.testing.assert_close(state.transformed_weight(), expected_transformed)
    torch.testing.assert_close(
        state.binary_weight(zero_rule="positive"),
        torch.tensor([[1.0, -1.0, 1.0], [-1.0, 1.0, -1.0]]),
    )
    exposed_snapshot = state.dense_source
    exposed_snapshot.zero_()
    torch.testing.assert_close(state.dense_source, expected_source)
    assert "_dense_source" not in dict(state.named_parameters())


def test_transformed_weight_validates_channel_shape_and_zero_avoidance():
    dense = torch.ones(2, 3)
    with pytest.raises(ValueError, match="shape"):
        transformed_weight(dense, torch.ones(2))
    with pytest.raises(ValueError, match="must not contain zero"):
        transformed_weight(dense, torch.tensor([1.0, 0.0, 1.0]))


def test_zero_sign_policies_are_explicit_and_do_not_hide_nan():
    values = torch.tensor([-2.0, 0.0, 3.0, float("nan")])

    positive = binary_sign(values, ZeroSignRule.POSITIVE)
    negative = binary_sign(values, ZeroSignRule.NEGATIVE)
    torch.testing.assert_close(positive[:3], torch.tensor([-1.0, 1.0, 1.0]))
    torch.testing.assert_close(negative[:3], torch.tensor([-1.0, -1.0, 1.0]))
    assert torch.isnan(positive[-1]) and torch.isnan(negative[-1])
    with pytest.raises(ValueError, match="forbidden"):
        binary_sign(values, ZeroSignRule.ERROR)


def test_dual_scales_recompute_current_row_means_and_merge_for_inference():
    state = DualScaleState(
        2,
        analytical_gradient="detached",
        dtype=torch.float64,
    )
    latent = torch.tensor([[1.0, -3.0], [0.0, 2.0]], dtype=torch.float64)

    torch.testing.assert_close(state.learned(), torch.ones(2, dtype=torch.float64))
    torch.testing.assert_close(state.analytical(latent), torch.tensor([2.0, 1.0], dtype=torch.float64))
    with torch.no_grad():
        state.learned.value.copy_(torch.tensor([2.0, 0.5], dtype=torch.float64))
    torch.testing.assert_close(state.merged(latent), torch.tensor([4.0, 0.5], dtype=torch.float64))
    torch.testing.assert_close(
        state.sign_weight(latent, zero_rule="negative"),
        torch.tensor([[4.0, -4.0], [-0.5, 0.5]], dtype=torch.float64),
    )

    updated_latent = torch.tensor([[2.0, -6.0], [4.0, 4.0]], dtype=torch.float64)
    torch.testing.assert_close(state.analytical(updated_latent), torch.tensor([4.0, 4.0], dtype=torch.float64))
    torch.testing.assert_close(state.merged(updated_latent), torch.tensor([8.0, 2.0], dtype=torch.float64))


def test_analytical_scale_gradient_treatment_is_explicit():
    latent = torch.tensor([[1.0, -3.0]], requires_grad=True)

    detached = analytical_row_scales(latent, gradient=AnalyticalScaleGradient.DETACHED)
    differentiable = analytical_row_scales(latent, gradient=AnalyticalScaleGradient.DIFFERENTIABLE)

    assert not detached.requires_grad
    assert differentiable.requires_grad
    differentiable.sum().backward()
    torch.testing.assert_close(latent.grad, torch.tensor([[0.5, -0.5]]))


def test_trainable_scale_identification_uses_declarations_not_name_matching():
    class Combined(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.stage1 = Stage1ScaleState(torch.ones(2, 3), parameterization="positive_exp")
            self.dual = DualScaleState(2, analytical_gradient="detached")
            self.not_a_scale = nn.Parameter(torch.tensor(1.0))

    module = Combined()
    scales = named_trainable_scales(module)
    report = scale_trainability_report(module)

    assert [(item.name, item.role) for item in scales] == [
        ("stage1.input_scale.raw_scale", ScaleRole.STAGE1_INPUT),
        ("dual.learned.value", ScaleRole.LEARNED_ROW),
    ]
    assert report.declared_trainable == (
        "stage1.input_scale.raw_scale",
        "dual.learned.value",
    )
    assert report.undeclared_trainable == ("not_a_scale",)
    assert not report.only_declared_scales_trainable
    module.not_a_scale.requires_grad_(False)
    assert scale_trainability_report(module).only_declared_scales_trainable


def test_stage1_diagnostics_include_required_metrics_extrema_and_counts():
    report = stage1_diagnostics(
        scales={"q_proj": torch.tensor([0.5, 2.0])},
        transformed_weights={"q_proj": torch.tensor([[1.0, -1.0]])},
        initial_loss=4.0,
        final_loss=2.0,
        held_out_loss=2.5,
        binary_reconstruction_error=0.125,
        gradients={"q_proj": torch.tensor([0.25, -0.25])},
    )

    assert report.initial_loss == 4.0
    assert report.final_loss == 2.0
    assert report.held_out_loss == 2.5
    assert report.binary_reconstruction_error == 0.125
    assert report.scale_min == 0.5 and report.scale_max == 2.0
    assert report.finite_state.is_finite
    assert report.to_dict() == {
        "initial_loss": 4.0,
        "final_loss": 2.0,
        "held_out_loss": 2.5,
        "binary_reconstruction_error": 0.125,
        "scale_min": 0.5,
        "scale_max": 2.0,
        "nonfinite_counts": {
            "scale.q_proj": 0,
            "transformed_weight.q_proj": 0,
            "loss.final": 0,
            "loss.held_out": 0,
            "loss.initial": 0,
            "reconstruction_error.binary": 0,
            "gradient.q_proj": 0,
        },
        "total_nonfinite_count": 0,
        "failing_fields": [],
        "is_finite": True,
    }


def test_nonfinite_diagnostics_cover_every_state_category_and_fail_with_refs():
    diagnostics = diagnose_finite_state(
        scales={"layer": torch.tensor([1.0, float("inf")])},
        transformed_weights={"layer": torch.tensor([float("nan")])},
        losses={"train": float("-inf")},
        reconstruction_errors={"binary": float("nan")},
        gradients={"input": torch.tensor([float("inf")])},
    )

    assert not diagnostics.is_finite
    assert diagnostics.total_nonfinite_count == 5
    assert diagnostics.failing_fields == (
        "scale.layer",
        "transformed_weight.layer",
        "loss.train",
        "reconstruction_error.binary",
        "gradient.input",
    )
    assert diagnostics.extrema("scale") == (1.0, 1.0)
    with pytest.raises(NumericalFailure) as captured:
        diagnostics.raise_if_nonfinite(checkpoint_id="checkpoint-7", batch_id="batch-3")
    failure = captured.value
    assert failure.retryability is Retryability.FROM_CHECKPOINT
    assert failure.affected_ids == {
        "batch_ids": ("batch-3",),
        "checkpoint_ids": ("checkpoint-7",),
    }
    assert failure.context["failing_fields"] == list(diagnostics.failing_fields)
    assert failure.context["total_nonfinite_count"] == 5


def test_binary_reconstruction_error_is_mean_squared_error():
    reference = torch.tensor([1.0, -1.0, 2.0])
    candidate = torch.tensor([1.0, 1.0, 1.0])

    assert math.isclose(
        binary_reconstruction_error(reference, candidate).item(),
        5.0 / 3.0,
        rel_tol=1e-6,
    )
