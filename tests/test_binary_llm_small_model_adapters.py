from __future__ import annotations

import pytest
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace
from transformers import (
    GPTNeoXConfig,
    GPTNeoXForCausalLM,
    LlamaConfig,
    LlamaForCausalLM,
    PreTrainedTokenizerFast,
)

from binary_llm.adapters import (
    ActiveRepresentation,
    BinaryLinear,
    BinaryLinearConfig,
    ConfiguredBinaryLinearFactory,
    LocalSmallModelRequest,
    PYTHIA_70M,
    Pythia70MAdapter,
    SMOLLM_135M,
    SmallModelUse,
    SmolLM135MAdapter,
    TensorRole,
    TensorScope,
    TrainingActivationMode,
)
from binary_llm.domain import ModelIdentity
from binary_llm.math import (
    AnalyticalScaleGradient,
    OperatorPrecision,
    ProgressiveOperatorConfig,
    ZeroSignRule,
)


def _pythia_fixture() -> GPTNeoXForCausalLM:
    return GPTNeoXForCausalLM(
        GPTNeoXConfig(
            vocab_size=32,
            hidden_size=8,
            intermediate_size=16,
            num_hidden_layers=2,
            num_attention_heads=2,
            max_position_embeddings=16,
            tie_word_embeddings=False,
        )
    )


def _smollm_fixture() -> LlamaForCausalLM:
    return LlamaForCausalLM(
        LlamaConfig(
            vocab_size=32,
            hidden_size=8,
            intermediate_size=16,
            num_hidden_layers=2,
            num_attention_heads=2,
            num_key_value_heads=1,
            max_position_embeddings=16,
            tie_word_embeddings=True,
        )
    )

def _binary_factory() -> ConfiguredBinaryLinearFactory:
    return ConfiguredBinaryLinearFactory(
        BinaryLinearConfig(
            scale_parameterization="positive_exp",
            training_activation_mode=TrainingActivationMode.EXPLICIT_ACTIVATION_TRANSFORM,
            progressive_operator=ProgressiveOperatorConfig(
                precision=OperatorPrecision.FLOAT32,
                analytical_scale_gradient=AnalyticalScaleGradient.DETACHED,
            ),
            zero_sign_rule=ZeroSignRule.POSITIVE,
        )
    )


def _identity(adapter, model) -> ModelIdentity:
    return ModelIdentity(
        model_id=adapter.registration.model_id,
        revision="sealed-test-model-revision",
        tokenizer_revision="sealed-test-tokenizer-revision",
        architecture=adapter.registration.architecture,
        parameter_count=sum(parameter.numel() for parameter in model.parameters()),
        config_hash="config-sha256",
        tokenizer_hashes=("tokenizer-sha256",),
        template_hashes=("no-template-sha256",),
        weight_hashes=("weights-sha256",),
        tied_weights=adapter.registration.tied_weights,
    )


def test_registered_small_models_have_distinct_scientific_roles_and_tokenizers() -> None:
    assert PYTHIA_70M.model_id == "EleutherAI/pythia-70m-deduped"
    assert PYTHIA_70M.tokenizer_id == PYTHIA_70M.model_id
    assert PYTHIA_70M.use is SmallModelUse.CHEAP_SCREENING
    assert PYTHIA_70M.architecture == "GPTNeoXForCausalLM"
    assert not PYTHIA_70M.tied_weights
    assert SMOLLM_135M.model_id == "HuggingFaceTB/SmolLM-135M"
    assert SMOLLM_135M.tokenizer_id == SMOLLM_135M.model_id
    assert SMOLLM_135M.use is SmallModelUse.PAPER_COMPATIBLE_REFERENCE
    assert SMOLLM_135M.architecture == "LlamaForCausalLM"
    assert SMOLLM_135M.tied_weights
    assert PYTHIA_70M.activations_are_excluded
    assert SMOLLM_135M.activations_are_excluded


def test_pythia_inventory_uses_explicit_fused_attention_and_gelu_ffn_roles() -> None:
    adapter = Pythia70MAdapter()
    model = _pythia_fixture()
    inventory = adapter.enumerate_tensors(model)
    scope = adapter.binary_body(inventory)
    body = {item.name: item.semantic_role for item in scope.binary_body}

    assert len(body) == 8
    for layer in range(2):
        prefix = f"gpt_neox.layers.{layer}"
        assert body[f"{prefix}.attention.query_key_value.weight"] is TensorRole.ATTENTION_QUERY_KEY_VALUE
        assert body[f"{prefix}.attention.dense.weight"] is TensorRole.ATTENTION_OUTPUT
        assert body[f"{prefix}.mlp.dense_h_to_4h.weight"] is TensorRole.FFN_UP
        assert body[f"{prefix}.mlp.dense_4h_to_h.weight"] is TensorRole.FFN_DOWN
    assert all(item.scope is TensorScope.BINARY_BODY for item in scope.binary_body)
    excluded = {item.name: item for item in scope.excluded}
    assert excluded["gpt_neox.embed_in.weight"].semantic_role is TensorRole.TOKEN_EMBEDDING
    assert excluded["embed_out.weight"].semantic_role is TensorRole.LANGUAGE_MODEL_HEAD
    assert excluded["embed_out.weight"].tied_to is None

def test_smollm_inventory_enumerates_separate_attention_gated_ffn_and_tied_head() -> None:
    adapter = SmolLM135MAdapter()
    model = _smollm_fixture()
    inventory = adapter.enumerate_tensors(model)
    scope = adapter.binary_body(inventory)
    body = {item.name: item.semantic_role for item in scope.binary_body}

    assert len(body) == 14
    expected_per_layer = {
        "self_attn.q_proj.weight": TensorRole.ATTENTION_QUERY,
        "self_attn.k_proj.weight": TensorRole.ATTENTION_KEY,
        "self_attn.v_proj.weight": TensorRole.ATTENTION_VALUE,
        "self_attn.o_proj.weight": TensorRole.ATTENTION_OUTPUT,
        "mlp.gate_proj.weight": TensorRole.FFN_GATE,
        "mlp.up_proj.weight": TensorRole.FFN_UP,
        "mlp.down_proj.weight": TensorRole.FFN_DOWN,
    }
    for layer in range(2):
        for suffix, role in expected_per_layer.items():
            assert body[f"model.layers.{layer}.{suffix}"] is role
    excluded = {item.name: item for item in scope.excluded}
    assert excluded["model.embed_tokens.weight"].semantic_role is TensorRole.TOKEN_EMBEDDING
    assert excluded["lm_head.weight"].semantic_role is TensorRole.LANGUAGE_MODEL_HEAD
    assert excluded["lm_head.weight"].tied_to == "model.embed_tokens.weight"
    assert model.lm_head.weight is model.model.embed_tokens.weight


@pytest.mark.parametrize(
    ("adapter", "model", "expected_count"),
    [
        (Pythia70MAdapter(), _pythia_fixture(), 8),
        (SmolLM135MAdapter(), _smollm_fixture(), 14),
    ],
)
def test_adapters_replace_only_declared_body_linears(adapter, model, expected_count) -> None:
    adapter.replace_linears(model, _binary_factory())
    active = adapter.active_representations(model)

    assert len(active) == expected_count
    assert all(item.representation is ActiveRepresentation.TRAINING for item in active)
    assert sum(isinstance(module, BinaryLinear) for module in model.modules()) == expected_count
    assert not isinstance(model.get_output_embeddings(), BinaryLinear)
    inventory = adapter.enumerate_tensors(model)
    assert len(adapter.binary_body(inventory).binary_body) == expected_count


def _write_tokenizer(directory) -> None:
    backend = Tokenizer(WordLevel({"<unk>": 0, "hello": 1, "world": 2}, unk_token="<unk>"))
    backend.pre_tokenizer = Whitespace()
    tokenizer = PreTrainedTokenizerFast(tokenizer_object=backend, unk_token="<unk>")
    tokenizer.save_pretrained(directory)


def test_local_loader_uses_sealed_identity_and_local_artifacts(tmp_path) -> None:
    adapter = SmolLM135MAdapter()
    source = _smollm_fixture()
    source.save_pretrained(tmp_path)
    _write_tokenizer(tmp_path)
    identity = _identity(adapter, source)

    loaded = adapter.load_local(LocalSmallModelRequest(tmp_path, tmp_path, identity))

    assert type(loaded.model) is LlamaForCausalLM
    assert loaded.model.training is False
    assert loaded.tokenizer.get_vocab() == {"<unk>": 0, "hello": 1, "world": 2}
    assert loaded.identity == identity
    assert loaded.registration is SMOLLM_135M


def test_local_loader_rejects_cross_architecture_identity_before_loading(tmp_path) -> None:
    adapter = Pythia70MAdapter()
    identity = _identity(adapter, _pythia_fixture())
    wrong = ModelIdentity(**{**identity.to_dict(), "model_id": SMOLLM_135M.model_id})

    with pytest.raises(ValueError, match="identity does not match"):
        adapter.load_local(LocalSmallModelRequest(tmp_path, tmp_path, wrong))
