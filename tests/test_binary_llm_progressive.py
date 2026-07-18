import math

import pytest
import torch

from binary_llm.math import (
    AnalyticalScaleGradient,
    OperatorPrecision,
    PhaseIndexConvention,
    ProgressionScheduleConfig,
    ProgressionScheduleKind,
    ProgressiveOperatorConfig,
    paper_exponential_parameter,
    phase_indices,
    progression_parameters,
    progressive,
    progressive_derivative,
    registered_schedule_kinds,
)


def operator_config(
    precision: OperatorPrecision = OperatorPrecision.FLOAT32,
    gradient: AnalyticalScaleGradient = AnalyticalScaleGradient.DETACHED,
) -> ProgressiveOperatorConfig:
    return ProgressiveOperatorConfig(
        precision=precision,
        analytical_scale_gradient=gradient,
    )


def test_zero_progression_is_the_exact_continuous_identity_in_both_passes() -> None:
    x = torch.tensor([-3.0, -0.0, 0.5, 9.0], requires_grad=True)

    result = progressive(x, 0.0, config=operator_config())
    result.sum().backward()

    assert torch.equal(result, x.detach())
    assert torch.equal(x.grad, torch.ones_like(x))
    assert torch.equal(progressive_derivative(x.detach(), 0.0, config=operator_config()), torch.ones_like(x))


def test_forward_and_custom_backward_match_the_paper_equations() -> None:
    x = torch.tensor([-2.0, -0.25, 0.0, 0.75, 3.0], requires_grad=True)
    upstream = torch.tensor([0.5, -2.0, 1.0, 1.5, -0.75])
    t = 1.7

    result = progressive(x, t, config=operator_config())
    result.backward(upstream)

    reference_x = x.detach().double()
    expected_forward = torch.tanh(t * reference_x) / math.tanh(t)
    expected_derivative = t * (1.0 - torch.tanh(t * reference_x).square()) / math.tanh(t)
    assert torch.allclose(result.double(), expected_forward, atol=2e-6, rtol=2e-6)
    assert torch.allclose(x.grad.double(), upstream.double() * expected_derivative, atol=2e-6, rtol=2e-6)
    assert torch.allclose(
        progressive_derivative(x.detach(), t, config=operator_config()).double(),
        expected_derivative,
        atol=2e-6,
        rtol=2e-6,
    )


def test_small_nonzero_progression_avoids_the_unstable_zero_over_zero_ratio() -> None:
    x = torch.tensor([-4.0, 0.0, 2.5], requires_grad=True)

    result = progressive(x, 1e-12, config=operator_config())
    result.sum().backward()

    assert torch.isfinite(result).all()
    assert torch.isfinite(x.grad).all()
    assert torch.allclose(result, x.detach(), atol=1e-6, rtol=1e-6)
    assert torch.allclose(x.grad, torch.ones_like(x), atol=1e-6, rtol=1e-6)


def test_operator_precision_and_analytical_scale_gradient_are_explicit() -> None:
    x = torch.tensor([-0.7, 0.2, 1.1], dtype=torch.float32)
    bf16_config = operator_config(
        OperatorPrecision.BFLOAT16,
        AnalyticalScaleGradient.DIFFERENTIABLE,
    )

    result = progressive(x, 0.9, config=bf16_config)
    t = torch.tensor(0.9, dtype=torch.bfloat16)
    expected = (torch.tanh(t * x.to(torch.bfloat16)) / torch.tanh(t)).to(torch.float32)

    assert result.dtype is x.dtype
    assert torch.equal(result, expected)
    assert bf16_config.precision is OperatorPrecision.BFLOAT16
    assert bf16_config.analytical_scale_gradient is AnalyticalScaleGradient.DIFFERENTIABLE
    with pytest.raises(TypeError, match="config"):
        progressive(x, 1.0, config=None)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="precision"):
        ProgressiveOperatorConfig("fp16", "detached")
    with pytest.raises(ValueError, match="analytical_scale_gradient"):
        ProgressiveOperatorConfig("fp32", "implicit")


def test_paper_schedule_supports_both_twenty_phase_index_conventions() -> None:
    zero_config = ProgressionScheduleConfig("paper_exponential", "0..19")
    one_config = ProgressionScheduleConfig("paper_exponential", "1..20")

    zero_indices = phase_indices(20, PhaseIndexConvention.ZERO_BASED)
    one_indices = phase_indices(20, PhaseIndexConvention.ONE_BASED)
    zero_values = progression_parameters(zero_config)
    one_values = progression_parameters(one_config)

    assert zero_indices == tuple(range(20))
    assert one_indices == tuple(range(1, 21))
    assert zero_values == tuple(paper_exponential_parameter(c) for c in range(20))
    assert one_values == tuple(paper_exponential_parameter(c) for c in range(1, 21))
    assert zero_values[0] == 0.0
    assert one_values[0] > 0.0


def test_registered_alternatives_are_monotonic_and_match_paper_endpoints() -> None:
    registered = registered_schedule_kinds()
    assert registered == tuple(ProgressionScheduleKind)
    assert ProgressionScheduleKind.LINEAR_ENDPOINT_MATCHED in registered
    assert ProgressionScheduleKind.COSINE_ENDPOINT_MATCHED in registered

    paper = progression_parameters(
        ProgressionScheduleConfig(
            ProgressionScheduleKind.PAPER_EXPONENTIAL,
            PhaseIndexConvention.ZERO_BASED,
        )
    )
    linear = progression_parameters(
        ProgressionScheduleConfig(
            ProgressionScheduleKind.LINEAR_ENDPOINT_MATCHED,
            PhaseIndexConvention.ZERO_BASED,
        )
    )
    cosine = progression_parameters(
        ProgressionScheduleConfig(
            ProgressionScheduleKind.COSINE_ENDPOINT_MATCHED,
            PhaseIndexConvention.ZERO_BASED,
        )
    )

    for alternative in (linear, cosine):
        assert alternative[0] == paper[0]
        assert alternative[-1] == paper[-1]
        assert all(left <= right for left, right in zip(alternative, alternative[1:]))
    assert linear[10] != paper[10]
    assert cosine[10] != paper[10]


def test_invalid_progression_inputs_fail_closed() -> None:
    config = operator_config()
    x = torch.ones(2)

    with pytest.raises(ValueError, match="non-negative"):
        progressive(x, -1.0, config=config)
    with pytest.raises(ValueError, match="finite"):
        progressive(x, math.inf, config=config)
    with pytest.raises(TypeError, match="floating"):
        progressive(torch.ones(2, dtype=torch.int64), 1.0, config=config)
    with pytest.raises(ValueError, match="positive integer"):
        phase_indices(0, PhaseIndexConvention.ZERO_BASED)
    with pytest.raises(ValueError, match="phase_index"):
        ProgressionScheduleConfig("paper_exponential", "implicit")
    with pytest.raises(ValueError, match="non-negative integer"):
        paper_exponential_parameter(-1)
