from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F

from binary_llm.adapters import (
    ActiveRepresentation,
    BinaryLinear,
    BinaryLinearConfig,
    ConfiguredBinaryLinearFactory,
    TensorRole,
    TensorScope,
    TinyCausalLMConfig,
    TinyCausalModelAdapter,
    TrainingActivationMode,
    build_seeded_tiny_causal_lm,
)
from binary_llm.math import (
    AnalyticalScaleGradient,
    OperatorPrecision,
    ProgressiveOperatorConfig,
    ZeroSignRule,
    binary_sign,
    dual_scaled_sign_weight,
)


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


def converted_model(seed: int = 17):
    adapter = TinyCausalModelAdapter()
    model = build_seeded_tiny_causal_lm(
        seed=seed,
        config=TinyCausalLMConfig(
            vocab_size=19,
            hidden_size=8,
            intermediate_size=12,
            max_sequence_length=8,
        ),
    )
    adapter.replace_linears(model, ConfiguredBinaryLinearFactory(binary_config()))
    return adapter, model


def test_inventory_classifies_every_named_tensor_once_by_declared_role() -> None:
    adapter = TinyCausalModelAdapter()
    model = build_seeded_tiny_causal_lm(seed=4)

    inventory = adapter.enumerate_tensors(model)
    scope = adapter.binary_body(inventory)
    actual_names = {
        name for name, _ in tuple(model.named_parameters()) + tuple(model.named_buffers())
    }

    assert {item.name for item in inventory} == actual_names
    assert len(inventory) == len(actual_names)
    assert {item.semantic_role for item in scope.binary_body} == {
        TensorRole.ATTENTION_QUERY,
        TensorRole.ATTENTION_KEY,
        TensorRole.ATTENTION_VALUE,
        TensorRole.ATTENTION_OUTPUT,
        TensorRole.FFN_GATE,
        TensorRole.FFN_UP,
        TensorRole.FFN_DOWN,
    }
    assert all(item.name.endswith(".weight") for item in scope.binary_body)
    excluded_by_name = {item.name: item for item in scope.excluded}
    assert excluded_by_name["token_embedding.weight"].semantic_role is TensorRole.TOKEN_EMBEDDING
    assert excluded_by_name["lm_head.weight"].semantic_role is TensorRole.LANGUAGE_MODEL_HEAD
    assert excluded_by_name["block.input_norm.weight"].semantic_role is TensorRole.NORMALIZATION
    assert excluded_by_name["block.attention.q_proj.bias"].semantic_role is TensorRole.BIAS
    assert excluded_by_name["block.attention.causal_mask"].semantic_role is TensorRole.BUFFER
    assert all(item.scope is TensorScope.EXCLUDED for item in scope.excluded)


def test_adapter_replaces_only_declared_linears_and_preserves_names_and_roles() -> None:
    adapter, model = converted_model()

    replaced = {
        name: module
        for name, module in model.named_modules()
        if isinstance(module, BinaryLinear)
    }
    assert tuple(replaced) == tuple(adapter.declared_linear_roles)
    assert isinstance(model.lm_head, nn.Linear)
    assert isinstance(model.token_embedding, nn.Embedding)
    assert isinstance(model.block.input_norm, nn.LayerNorm)
    for module_name, module in replaced.items():
        assert module.tensor_name == f"{module_name}.weight"
        assert module.semantic_role is adapter.declared_linear_roles[module_name]
    inventory = adapter.enumerate_tensors(model)
    assert {item.name for item in inventory} == {
        name for name, _ in tuple(model.named_parameters()) + tuple(model.named_buffers())
    }


def test_binary_linear_exposes_exact_training_progressive_sign_and_dense_views() -> None:
    source = nn.Linear(3, 2, bias=True)
    with torch.no_grad():
        source.weight.copy_(torch.tensor([[0.2, -1.5, 0.7], [-0.3, 0.6, -0.9]]))
        source.bias.copy_(torch.tensor([0.1, -0.2]))
    dense_weight = source.weight.detach().clone()
    dense_bias = source.bias.detach().clone()
    module = BinaryLinear.from_linear(
        source,
        tensor_name="block.attention.q_proj.weight",
        semantic_role=TensorRole.ATTENTION_QUERY,
        config=binary_config(),
    )
    inputs = torch.tensor([[[0.5, -1.0, 2.0]]])

    module.set_active_representation(ActiveRepresentation.DENSE_REFERENCE)
    torch.testing.assert_close(module(inputs), F.linear(inputs, dense_weight, dense_bias))

    module.set_active_representation(ActiveRepresentation.TRAINING)
    expected_training = F.linear(
        inputs * module.input_scale(),
        binary_sign(module.transformed_weight(), ZeroSignRule.POSITIVE),
        dense_bias,
    )
    torch.testing.assert_close(module(inputs), expected_training)
    assert module.active_representation is ActiveRepresentation.TRAINING
    assert module.input_scale.raw_scale.requires_grad
    assert not module.weight.requires_grad

    module.set_active_representation(
        ActiveRepresentation.PROGRESSIVE,
        progression_parameter=0.0,
    )
    torch.testing.assert_close(module(inputs), F.linear(inputs, dense_weight, dense_bias))
    assert module.weight.requires_grad and module.dual_scale.learned.value.requires_grad

    module.set_active_representation(ActiveRepresentation.SIGN)
    expected_sign = dual_scaled_sign_weight(
        module.transformed_weight(),
        module.dual_scale.learned(),
        gradient=AnalyticalScaleGradient.DETACHED,
        zero_rule=ZeroSignRule.POSITIVE,
    )
    torch.testing.assert_close(module(inputs), F.linear(inputs, expected_sign, dense_bias))
    assert not any(parameter.requires_grad for parameter in module.parameters())


def test_seeded_tiny_causal_model_switches_observable_views_deterministically() -> None:
    adapter, model = converted_model(seed=23)
    _, same_model = converted_model(seed=23)
    input_ids = torch.tensor([[1, 2, 3, 4], [4, 3, 2, 1]])

    assert adapter.active_representation(model) is ActiveRepresentation.TRAINING
    training_logits = model(input_ids)
    torch.testing.assert_close(training_logits, same_model(input_ids), rtol=0, atol=0)
    assert training_logits.shape == (2, 4, 19)

    trainable_names = tuple(
        name for name, parameter in model.named_parameters() if parameter.requires_grad
    )
    assert trainable_names
    assert all(name.endswith("input_scale.raw_scale") for name in trainable_names)

    adapter.set_active_representation(
        model,
        ActiveRepresentation.PROGRESSIVE,
        progression_parameter=0.0,
    )
    assert adapter.active_representation(model) is ActiveRepresentation.PROGRESSIVE
    progressive_logits = model(input_ids)

    adapter.set_active_representation(model, ActiveRepresentation.DENSE_REFERENCE)
    assert adapter.active_representation(model) is ActiveRepresentation.DENSE_REFERENCE
    dense_logits = model(input_ids)
    torch.testing.assert_close(progressive_logits, dense_logits, rtol=1e-5, atol=1e-6)

    adapter.set_active_representation(model, ActiveRepresentation.SIGN)
    assert adapter.active_representation(model) is ActiveRepresentation.SIGN
    sign_logits = model(input_ids)
    assert torch.isfinite(sign_logits).all()
    assert not torch.equal(sign_logits, dense_logits)
