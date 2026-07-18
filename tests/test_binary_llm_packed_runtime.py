from __future__ import annotations

from dataclasses import replace

import pytest
import torch

from binary_llm.adapters import (
    ActiveRepresentation,
    BinaryLinear,
    BinaryLinearConfig,
    ConfiguredBinaryLinearFactory,
    TensorRole,
    TinyCausalLMConfig,
    TinyCausalModelAdapter,
    TrainingActivationMode,
    build_seeded_tiny_causal_lm,
)
from binary_llm.domain import (
    FormatSpec,
    ParityError,
    ToleranceDefinition,
    ToleranceSet,
)
from binary_llm.export import (
    IdentifiedCheckpoint,
    PackedExporter,
    PackedLinear,
    PackedTensorSource,
    ScalarPackedRuntime,
    bind_packed_linears,
)
from binary_llm.math import (
    AnalyticalScaleGradient,
    OperatorPrecision,
    ProgressiveOperatorConfig,
    ZeroSignRule,
    analytical_row_scales,
)
from binary_llm.orchestration import (
    BINARY_OUTPUT_METRIC,
    GREEDY_TOKEN_METRIC,
    LOGIT_METRIC,
    SELECTED_LAYER_METRIC,
    TOOL_DECISION_METRIC,
    ParityCoordinator,
    capture_parity_trace,
)


def format_spec(*, temporary_limit: int = 32) -> FormatSpec:
    return FormatSpec(
        format_id="scalar-packed-runtime-test",
        version="1",
        bit_order="lsb0",
        row_order="row_major",
        row_alignment=2,
        zero_sign_rule="positive",
        scale_dtype="float32",
        scale_endianness="little",
        metadata_encoding="canonical_json",
        allowed_representation_ids=("binary-v1",),
        temporary_expansion_limit_bytes=temporary_limit,
    )


def exporter() -> PackedExporter:
    return PackedExporter(
        exporter_revision="packed-exporter-test-r1",
        runtime_revision="scalar-runtime-test-r1",
        build_flags=("deterministic", "cpu"),
    )


def runtime() -> ScalarPackedRuntime:
    return ScalarPackedRuntime(
        runtime_revision="scalar-runtime-test-r1",
        build_flags=("deterministic", "cpu"),
    )


def one_tensor_checkpoint() -> IdentifiedCheckpoint:
    return IdentifiedCheckpoint(
        checkpoint_id="sign-checkpoint",
        checkpoint_sha256="b" * 64,
        tensors=(
            PackedTensorSource(
                name="block.attention.q_proj.weight",
                semantic_role=TensorRole.ATTENTION_QUERY,
                representation_id="binary-v1",
                latent_weight=torch.tensor(
                    [[1.0, -2.0, 0.0, 4.0, -5.0], [-1.0, 2.0, -3.0, -4.0, 5.0]]
                ),
                merged_scale=torch.tensor([0.5, 1.25]),
            ),
        ),
    )


def test_scalar_runtime_maps_non_byte_aligned_rows_without_dense_weight_copy(tmp_path) -> None:
    service = exporter()
    source = one_tensor_checkpoint()
    artifact = service.export(
        service.plan(source, format_spec(), scale_tolerance=0.0),
        tmp_path / "runtime.bllmp",
    )
    inputs = torch.tensor([[0.5, -1.0, 2.0, 3.0, -0.25]], dtype=torch.float32)
    expected_signs = torch.tensor([[1, -1, 1, 1, -1], [-1, 1, -1, -1, 1]])
    expected = torch.nn.functional.linear(
        inputs,
        expected_signs.float() * torch.tensor([0.5, 1.25]).unsqueeze(1),
        torch.tensor([0.1, -0.2]),
    )

    with runtime().load(artifact) as session:
        actual = session.linear(
            "block.attention.q_proj.weight", inputs, torch.tensor([0.1, -0.2])
        )
        memory = session.memory_report()
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        assert memory.mapped_packed_bytes == artifact.bytes
        assert memory.persistent_metadata_bytes > 0
        assert memory.persistent_scale_bytes == 0
        assert memory.persistent_dense_weight_bytes == 0
        assert memory.maximum_temporary_bytes == memory.persistent_metadata_bytes
        assert memory.maximum_temporary_expansion_bytes == 0
        assert memory.temporary_expansion_within_limit
        assert memory.maximum_output_bytes == actual.numel() * actual.element_size()


def test_runtime_rejects_an_unregistered_runtime_tuple_before_execution(tmp_path) -> None:
    service = exporter()
    source = one_tensor_checkpoint()
    artifact = service.export(
        service.plan(source, format_spec(), scale_tolerance=0.0),
        tmp_path / "wrong-runtime.bllmp",
    )
    incompatible = ScalarPackedRuntime(
        runtime_revision="different-runtime",
        build_flags=("deterministic", "cpu"),
    )
    with pytest.raises(ParityError, match="runtime revision") as captured:
        incompatible.load(artifact)
    assert captured.value.code == "parity.runtime_revision"


def binary_config() -> BinaryLinearConfig:
    return BinaryLinearConfig(
        scale_parameterization="positive_exp",
        training_activation_mode=TrainingActivationMode.EXPLICIT_ACTIVATION_TRANSFORM,
        progressive_operator=ProgressiveOperatorConfig(
            precision=OperatorPrecision.FLOAT32,
            analytical_scale_gradient=AnalyticalScaleGradient.DETACHED,
        ),
        zero_sign_rule=ZeroSignRule.POSITIVE,
    )


def converted_tiny(seed: int):
    adapter = TinyCausalModelAdapter()
    model = build_seeded_tiny_causal_lm(
        seed=seed,
        config=TinyCausalLMConfig(
            vocab_size=17,
            hidden_size=8,
            intermediate_size=12,
            max_sequence_length=8,
        ),
    )
    adapter.replace_linears(model, ConfiguredBinaryLinearFactory(binary_config()))
    adapter.set_active_representation(model, ActiveRepresentation.SIGN)
    return adapter, model


def checkpoint_from_model(model: torch.nn.Module) -> IdentifiedCheckpoint:
    tensors: list[PackedTensorSource] = []
    for module in model.modules():
        if not isinstance(module, BinaryLinear):
            continue
        latent = module.transformed_weight().detach()
        merged_scale = (
            analytical_row_scales(
                latent,
                gradient=AnalyticalScaleGradient.DETACHED,
            )
            * module.dual_scale.learned().detach()
        )
        tensors.append(
            PackedTensorSource(
                name=module.tensor_name,
                semantic_role=module.semantic_role,
                representation_id="binary-v1",
                latent_weight=latent,
                merged_scale=merged_scale,
            )
        )
    return IdentifiedCheckpoint(
        checkpoint_id="tiny-sign-checkpoint",
        checkpoint_sha256="c" * 64,
        tensors=tuple(tensors),
    )


def tolerance_set(*, include_tools: bool = True) -> ToleranceSet:
    definitions = [
        ToleranceDefinition("binary", BINARY_OUTPUT_METRIC, "all_binary_linears", 2e-6, 1e-6, False),
        ToleranceDefinition("layers", SELECTED_LAYER_METRIC, "selected_layers", 2e-5, 1e-5, False),
        ToleranceDefinition("logits", LOGIT_METRIC, "final_logits", 2e-5, 1e-5, False),
        ToleranceDefinition("tokens", GREEDY_TOKEN_METRIC, "fixed_prompts", None, None, True),
    ]
    if include_tools:
        definitions.append(
            ToleranceDefinition("tools", TOOL_DECISION_METRIC, "fixed_prompts", None, None, True)
        )
    return ToleranceSet("packed-parity-preregistered", tuple(definitions))


def tool_decisions(tokens: torch.Tensor) -> tuple[str, ...]:
    return tuple(f'{{"tool":"route_{int(row[-1])}"}}' for row in tokens)


def test_tiny_model_parity_covers_linears_layers_logits_tokens_and_tools(tmp_path) -> None:
    adapter, training_model = converted_tiny(seed=41)
    _, packed_model = converted_tiny(seed=41)
    checkpoint = checkpoint_from_model(training_model)
    service = exporter()
    artifact = service.export(
        service.plan(checkpoint, format_spec(temporary_limit=0), scale_tolerance=0.0),
        tmp_path / "tiny.bllmp",
    )
    input_ids = torch.tensor([[1, 2, 3], [3, 2, 1]])
    binary_names = tuple(adapter.declared_linear_roles)

    training_trace = capture_parity_trace(
        training_model,
        input_ids,
        binary_linear_names=binary_names,
        selected_layer_names=("block",),
        greedy_steps=2,
        tool_decision=tool_decisions,
    )
    with runtime().load(artifact) as session:
        bind_packed_linears(packed_model, session)
        assert all(
            isinstance(packed_model.get_submodule(name), PackedLinear)
            for name in binary_names
        )
        assert not any(
            hasattr(packed_model.get_submodule(name), "weight") for name in binary_names
        )
        packed_trace = capture_parity_trace(
            packed_model,
            input_ids,
            binary_linear_names=binary_names,
            selected_layer_names=("block",),
            greedy_steps=2,
            tool_decision=tool_decisions,
        )
        evidence = ParityCoordinator().coordinate(
            artifact=artifact,
            training_operator_revision="binary-linear-sign-r1",
            tolerances=tolerance_set(),
            training_trace=training_trace,
            packed_trace=packed_trace,
            memory=session.memory_report(),
            decoded_signs_exact=True,
            scale_error_max=0.0,
            traces_ref=("trace://training", "trace://packed"),
        )

    assert evidence.passed
    assert not evidence.failures
    assert evidence.greedy_sequences_exact
    assert evidence.iris_decisions_exact
    assert not evidence.persistent_dense_copy_detected
    assert set(evidence.binary_linear_error_stats) == set(binary_names)
    assert set(evidence.layer_error_stats) == {"block"}
    assert evidence.logit_error_stats.passed


def test_parity_fails_exact_tool_mismatch_and_missing_preregistration(tmp_path) -> None:
    source = one_tensor_checkpoint()
    service = exporter()
    artifact = service.export(
        service.plan(source, format_spec(), scale_tolerance=0.0),
        tmp_path / "parity-failure.bllmp",
    )
    tensor = torch.tensor([[1.0]])
    from binary_llm.orchestration import ParityTrace

    reference = ParityTrace(
        {"linear": tensor}, {"layer": tensor}, tensor,
        ((1, 2),), ('{"tool":"calendar"}',),
    )
    mismatch = replace(reference, tool_decisions=('{"tool":"messages"}',))
    with runtime().load(artifact) as session:
        evidence = ParityCoordinator().coordinate(
            artifact=artifact,
            training_operator_revision="binary-linear-sign-r1",
            tolerances=tolerance_set(),
            training_trace=reference,
            packed_trace=mismatch,
            memory=session.memory_report(),
            decoded_signs_exact=True,
            scale_error_max=0.0,
            traces_ref=("trace://training", "trace://packed"),
        )
        assert not evidence.passed
        assert evidence.failures == ("tool_decisions",)

        with pytest.raises(ParityError, match="preregistered") as captured:
            ParityCoordinator().coordinate(
                artifact=artifact,
                training_operator_revision="binary-linear-sign-r1",
                tolerances=tolerance_set(include_tools=False),
                training_trace=reference,
                packed_trace=reference,
                memory=session.memory_report(),
                decoded_signs_exact=True,
                scale_error_max=0.0,
                traces_ref=("trace://training", "trace://packed"),
            )
        assert captured.value.code == "parity.missing_tolerance"
