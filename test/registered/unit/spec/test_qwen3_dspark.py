from types import SimpleNamespace

import pytest
from transformers import PretrainedConfig

from sglang.srt.arg_groups.speculative_hook import (
    _handle_dflash,
    _resolve_speculative_algorithm_alias,
)
from sglang.srt.speculative.dflash_utils import (
    DEFAULT_DFLASH_NUMERIC_MASK_TOKEN,
    parse_dflash_draft_config,
)


def _glm_dspark_config_dict(**overrides):
    cfg = {
        "architectures": ["DSparkDraftModel"],
        "aux_hidden_state_layer_ids": [8, 23, 39, 55, 70],
        "block_size": 8,
        "enable_confidence_head": True,
        "confidence_head_with_markov": True,
        "markov_rank": 256,
        "mask_token_id": 154856,
        "max_position_embeddings": 4096,
        "model_type": "dspark",
        "transformer_layer_config": {
            "hidden_size": 6144,
            "intermediate_size": 24576,
            "layer_types": ["full_attention"] * 5,
            "num_attention_heads": 64,
            "num_hidden_layers": 5,
            "num_key_value_heads": 64,
            "vocab_size": 154880,
        },
    }
    cfg.update(overrides)
    return cfg


def _glm52_target_config_dict(**overrides):
    cfg = {
        "architectures": ["GlmMoeDsaForCausalLM"],
        "hidden_size": 6144,
        "index_head_dim": 128,
        "index_topk": 2048,
        "kv_lora_rank": 512,
        "max_position_embeddings": 4096,
        "model_type": "glm_moe_dsa",
        "num_attention_heads": 64,
        "num_hidden_layers": 78,
        "num_key_value_heads": 64,
        "q_lora_rank": 2048,
        "qk_head_dim": 256,
        "qk_nope_head_dim": 192,
        "qk_rope_head_dim": 64,
        "rope_scaling": None,
        "v_head_dim": 256,
        "vocab_size": 154880,
    }
    cfg.update(overrides)
    return cfg


@pytest.mark.parametrize("arch", ["Qwen3DSparkDraftModel", "DSparkDraftModel"])
def test_dspark_alias_routes_dspark_draft_to_dflash(monkeypatch, arch):
    import sglang.srt.utils.hf_transformers_utils as hf_utils

    def fake_get_config(*args, **kwargs):
        return SimpleNamespace(architectures=[arch])

    monkeypatch.setattr(hf_utils, "get_config", fake_get_config)
    assert (
        _resolve_speculative_algorithm_alias("DSPARK", "unused-dspark-draft")
        == "DFLASH"
    )
    assert (
        _resolve_speculative_algorithm_alias("DFLASH", "unused-dspark-draft")
        == "DFLASH"
    )


def test_dspark_without_dspark_draft_stays_dspark():
    assert _resolve_speculative_algorithm_alias("DSPARK", None) == "DSPARK"


def test_dspark_alias_falls_back_to_raw_config_without_model_type(monkeypatch):
    import sglang.srt.utils.hf_transformers_utils as hf_utils

    raw_config = _glm_dspark_config_dict()
    raw_config.pop("model_type")

    def fake_get_config(*args, **kwargs):
        raise ValueError("Unrecognized model. Should have a model_type key.")

    def fake_get_config_dict(cls, model_path, **kwargs):
        assert model_path == "unused-dspark-draft"
        assert kwargs["trust_remote_code"] is True
        return raw_config, {}

    monkeypatch.setattr(hf_utils, "get_config", fake_get_config)
    monkeypatch.setattr(
        PretrainedConfig, "get_config_dict", classmethod(fake_get_config_dict)
    )

    assert (
        _resolve_speculative_algorithm_alias(
            "DSPARK", "unused-dspark-draft", trust_remote_code=True
        )
        == "DFLASH"
    )


def test_dflash_block_size_inference_falls_back_to_raw_config_without_model_type(
    monkeypatch,
):
    import sglang.srt.utils.hf_transformers_utils as hf_utils

    raw_config = _glm_dspark_config_dict()
    raw_config.pop("model_type")

    def fake_get_config(*args, **kwargs):
        raise ValueError("Unrecognized model. Should have a model_type key.")

    def fake_get_config_dict(cls, model_path, **kwargs):
        assert model_path == "unused-dspark-draft"
        assert kwargs["trust_remote_code"] is True
        assert kwargs["revision"] == "main"
        return raw_config, {}

    monkeypatch.setattr(hf_utils, "get_config", fake_get_config)
    monkeypatch.setattr(
        PretrainedConfig, "get_config_dict", classmethod(fake_get_config_dict)
    )

    server_args = SimpleNamespace(
        speculative_dflash_block_size=None,
        speculative_dspark_block_size=None,
        enable_dp_attention=False,
        pp_size=1,
        speculative_draft_model_path="unused-dspark-draft",
        speculative_num_steps=None,
        speculative_eagle_topk=None,
        speculative_num_draft_tokens=None,
        json_model_override_args="{}",
        trust_remote_code=True,
        speculative_draft_model_revision="main",
        speculative_draft_window_size=None,
        max_running_requests=None,
        enable_mixed_chunk=False,
    )

    _handle_dflash(server_args)

    assert server_args.speculative_num_draft_tokens == 8


def test_model_config_loads_raw_glm_dspark_config_without_model_type(monkeypatch):
    import sglang.srt.configs.model_config as model_config_module
    from sglang.srt.configs.model_config import ModelConfig

    raw_config = _glm_dspark_config_dict()
    raw_config.pop("model_type")

    def fake_get_config(*args, **kwargs):
        raise ValueError("Unrecognized model. Should have a model_type key.")

    def fake_get_config_dict(cls, model_path, **kwargs):
        assert model_path == "unused-dspark-draft"
        assert kwargs["trust_remote_code"] is True
        return raw_config, {}

    monkeypatch.setattr(model_config_module, "get_config", fake_get_config)
    monkeypatch.setattr(
        model_config_module, "get_generation_config", lambda *args, **kwargs: None
    )
    monkeypatch.setattr(
        PretrainedConfig, "get_config_dict", classmethod(fake_get_config_dict)
    )
    monkeypatch.setattr(ModelConfig, "_maybe_pull_model_for_runai", lambda *args: None)
    monkeypatch.setattr(
        ModelConfig, "_maybe_pull_model_tokenizer_from_remote", lambda *args: None
    )

    model_config = ModelConfig(
        model_path="unused-dspark-draft",
        trust_remote_code=True,
        dtype="float32",
        is_draft_model=True,
    )

    assert model_config.hf_config.architectures == ["DSparkDraftModel"]
    assert model_config.hf_config.hidden_size == 6144
    assert model_config.hidden_size == 6144
    assert model_config.num_hidden_layers == 5
    assert model_config.vocab_size == 154880


def test_parse_glm_dspark_config_shape():
    parsed = parse_dflash_draft_config(draft_hf_config=_glm_dspark_config_dict())

    assert parsed.num_hidden_layers == 5
    assert parsed.block_size == 8
    assert parsed.target_layer_ids == [8, 23, 39, 55, 70]
    assert parsed.mask_token == DEFAULT_DFLASH_NUMERIC_MASK_TOKEN
    assert parsed.mask_token_id == 154856


def test_normalize_glm_dspark_config_for_dflash_backbone():
    from sglang.srt.models.qwen3_dspark import _normalize_dspark_config

    cfg = PretrainedConfig.from_dict(_glm_dspark_config_dict())

    normalized = _normalize_dspark_config(cfg)

    assert normalized is not cfg
    assert normalized.hidden_size == 6144
    assert normalized.num_hidden_layers == 5
    assert normalized.num_target_layers == 71
    assert normalized.vocab_size == 154880


def test_model_config_normalizes_glm_dspark_before_shape_derivation(monkeypatch):
    import sglang.srt.configs.model_config as model_config_module
    from sglang.srt.configs.model_config import ModelConfig

    cfg = PretrainedConfig.from_dict(_glm_dspark_config_dict())

    monkeypatch.setattr(model_config_module, "get_config", lambda *args, **kwargs: cfg)
    monkeypatch.setattr(
        model_config_module, "get_generation_config", lambda *args, **kwargs: None
    )
    monkeypatch.setattr(ModelConfig, "_maybe_pull_model_for_runai", lambda *args: None)
    monkeypatch.setattr(
        ModelConfig, "_maybe_pull_model_tokenizer_from_remote", lambda *args: None
    )

    model_config = ModelConfig(
        model_path="unused-dspark-draft",
        dtype="float32",
        is_draft_model=True,
    )

    assert model_config.hf_config.hidden_size == 6144
    assert model_config.hf_config.num_hidden_layers == 5
    assert model_config.hf_config.num_attention_heads == 64
    assert model_config.hf_config.vocab_size == 154880
    assert model_config.hidden_size == 6144
    assert model_config.num_hidden_layers == 5
    assert model_config.num_attention_heads == 64
    assert model_config.vocab_size == 154880


def test_model_config_restores_glm_moe_dsa_raw_head_dims(monkeypatch):
    import sglang.srt.configs.model_config as model_config_module
    from sglang.srt.configs.model_config import ModelConfig

    raw_config = _glm52_target_config_dict()
    mutated_config = PretrainedConfig.from_dict(
        _glm52_target_config_dict(qk_rope_head_dim=192)
    )

    monkeypatch.setattr(
        model_config_module, "get_config", lambda *args, **kwargs: mutated_config
    )
    monkeypatch.setattr(
        model_config_module, "get_generation_config", lambda *args, **kwargs: None
    )
    monkeypatch.setattr(
        PretrainedConfig,
        "get_config_dict",
        classmethod(lambda cls, *args, **kwargs: (raw_config, {})),
    )
    monkeypatch.setattr(ModelConfig, "_maybe_pull_model_for_runai", lambda *args: None)
    monkeypatch.setattr(
        ModelConfig, "_maybe_pull_model_tokenizer_from_remote", lambda *args: None
    )

    model_config = ModelConfig(
        model_path="unused-glm52-target",
        trust_remote_code=True,
        dtype="float32",
    )

    assert model_config.hf_config.qk_nope_head_dim == 192
    assert model_config.hf_config.qk_rope_head_dim == 64
    assert model_config.hf_config.qk_head_dim == 256
    assert model_config.qk_nope_head_dim == 192
    assert model_config.qk_rope_head_dim == 64


def test_dspark_draft_model_builds_glm_markov_and_confidence_heads(monkeypatch):
    from sglang.srt.models.qwen3_dspark import DSparkDraftModel
    from sglang.srt.runtime_context import get_parallel

    import sglang.srt.server_args as server_args

    cfg = PretrainedConfig.from_dict(
        _glm_dspark_config_dict(
            aux_hidden_state_layer_ids=[0],
            block_size=2,
            markov_rank=8,
            mask_token_id=63,
            max_position_embeddings=16,
            transformer_layer_config={
                "hidden_size": 16,
                "intermediate_size": 32,
                "layer_types": ["full_attention"],
                "num_attention_heads": 4,
                "num_hidden_layers": 1,
                "num_key_value_heads": 4,
                "vocab_size": 64,
            },
        )
    )
    monkeypatch.setattr(
        server_args,
        "_global_server_args",
        SimpleNamespace(enable_dp_lm_head=False, rl_on_policy_target=None),
    )

    with get_parallel().override(tp_rank=0, tp_size=1):
        model = DSparkDraftModel(cfg)

    assert model.markov_rank == 8
    assert model.markov_head is not None
    assert model.confidence_head is not None
    assert model.confidence_head.proj.in_features == 24


def test_model_registry_resolves_dspark_draft_model():
    from sglang.srt.models.qwen3_dspark import Qwen3DSparkDraftModel
    from sglang.srt.models.registry import ModelRegistry

    model_cls, arch = ModelRegistry.resolve_model_cls(["DSparkDraftModel"])

    assert arch == "DSparkDraftModel"
    assert model_cls.__name__ == "DSparkDraftModel"
    assert issubclass(model_cls, Qwen3DSparkDraftModel)
