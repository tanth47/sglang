import unittest
from types import SimpleNamespace
from unittest.mock import patch

from transformers import PretrainedConfig

from sglang.srt.configs.model_config import (
    _normalize_nested_transformer_config,
    _restore_glm_moe_dsa_head_dims_from_raw_config,
)
from sglang.srt.model_executor.model_runner import (
    _resolve_dflash_or_dspark_capture_spec,
)
from sglang.srt.models.dspark import (
    EntryClass,
    build_confidence_head,
    normalize_dspark_draft_config,
)
from sglang.srt.server_args import ServerArgs
from sglang.srt.speculative.dflash_utils import parse_dflash_draft_config
from sglang.srt.speculative.draft_worker_common import (
    _resolve_draft_attention_backend,
    draft_is_deepseek_v4,
)
from sglang.srt.speculative.dspark_components.dspark_utils import (
    parse_dspark_draft_config,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=10, suite="base-a-test-cpu")


def _glm52_redhat_dspark_config() -> SimpleNamespace:
    return SimpleNamespace(
        architectures=["DSparkDraftModel"],
        aux_hidden_state_layer_ids=[8, 23, 39, 55, 70],
        block_size=8,
        confidence_head_with_markov=True,
        enable_confidence_head=True,
        markov_head_type="vanilla",
        markov_rank=256,
        mask_token_id=154856,
        speculators_config={
            "algorithm": "dspark",
            "default_proposal_method": "greedy",
            "proposal_methods": [
                {
                    "proposal_type": "greedy",
                    "speculative_tokens": 7,
                    "verifier_accept_k": 1,
                }
            ],
        },
        transformer_layer_config={
            "attention_bias": False,
            "head_dim": 64,
            "hidden_size": 6144,
            "intermediate_size": 12288,
            "layer_types": ["full_attention"] * 5,
            "model_type": "qwen3",
            "num_attention_heads": 64,
            "num_hidden_layers": 5,
            "num_key_value_heads": 64,
            "rms_norm_eps": 1e-5,
            "vocab_size": 154880,
        },
    )


def _glm52_redhat_dspark_raw_config() -> dict:
    return dict(vars(_glm52_redhat_dspark_config()))


def _make_draft_server_args() -> ServerArgs:
    server_args = ServerArgs(model_path="dummy")
    server_args.speculative_draft_model_path = "RedHatAI/GLM-5.2-speculator.dspark"
    server_args.speculative_draft_model_revision = None
    server_args.speculative_draft_attention_backend = "triton"
    server_args.trust_remote_code = False
    server_args.json_model_override_args = "{}"
    server_args.model_config_parser = "auto"
    return server_args


class TestGLM52RedHatDSparkConfig(CustomTestCase):
    def test_static_mode_skips_confidence_head_without_sts_collection(self):
        config = normalize_dspark_draft_config(_glm52_redhat_dspark_config())
        with patch.dict(
            "os.environ",
            {
                "SGLANG_RAGGED_VERIFY_MODE": "static",
                "SGLANG_DSPARK_STS_COLLECT_PATH": "",
            },
        ):
            self.assertIsNone(build_confidence_head(config))

    def test_static_mode_keeps_confidence_head_for_sts_collection(self):
        config = normalize_dspark_draft_config(_glm52_redhat_dspark_config())
        with patch.dict(
            "os.environ",
            {
                "SGLANG_RAGGED_VERIFY_MODE": "static",
                "SGLANG_DSPARK_STS_COLLECT_PATH": "/tmp/dspark-sts",
            },
        ):
            self.assertIsNotNone(build_confidence_head(config))

    def test_disabled_confidence_head_stays_disabled_for_sts_collection(self):
        config = normalize_dspark_draft_config(_glm52_redhat_dspark_config())
        config.enable_confidence_head = False
        with patch.dict(
            "os.environ",
            {
                "SGLANG_RAGGED_VERIFY_MODE": "static",
                "SGLANG_DSPARK_STS_COLLECT_PATH": "/tmp/dspark-sts",
            },
        ):
            self.assertIsNone(build_confidence_head(config))

    def test_model_config_normalizer_materializes_nested_transformer_config(self):
        config = PretrainedConfig(
            architectures=["DSparkDraftModel"],
            aux_hidden_state_layer_ids=[8, 23, 39, 55, 70],
            transformer_layer_config={
                "hidden_size": 6144,
                "num_hidden_layers": 5,
                "vocab_size": 154880,
            },
        )

        _normalize_nested_transformer_config(config)

        self.assertEqual(config.hidden_size, 6144)
        self.assertEqual(config.num_hidden_layers, 5)
        self.assertEqual(config.vocab_size, 154880)
        self.assertEqual(config.num_target_layers, 71)

    def test_dflash_parser_reads_nested_transformer_config_and_aux_layers(self):
        parsed = parse_dflash_draft_config(
            draft_hf_config=_glm52_redhat_dspark_config()
        )

        self.assertEqual(parsed.num_hidden_layers, 5)
        self.assertEqual(parsed.block_size, 8)
        self.assertEqual(parsed.target_layer_ids, [8, 23, 39, 55, 70])
        self.assertEqual(parsed.num_target_layers, 71)

    def test_dspark_parser_reads_redhat_fields(self):
        parsed = parse_dspark_draft_config(
            draft_hf_config=_glm52_redhat_dspark_config()
        )

        self.assertEqual(parsed.gamma, 7)
        self.assertEqual(parsed.target_layer_ids, [8, 23, 39, 55, 70])
        self.assertEqual(parsed.markov_rank, 256)
        self.assertEqual(parsed.markov_head_type, "vanilla")
        self.assertEqual(parsed.mask_token_id, 154856)

    def test_dspark_draft_model_is_registered(self):
        self.assertIn("DSparkDraftModel", {cls.__name__ for cls in EntryClass})

    def test_normalize_dspark_draft_config_materializes_backbone_fields(self):
        config = _glm52_redhat_dspark_config()
        normalize_dspark_draft_config(config)

        self.assertEqual(config.hidden_size, 6144)
        self.assertEqual(config.num_hidden_layers, 5)
        self.assertEqual(config.vocab_size, 154880)
        self.assertEqual(config.num_target_layers, 71)

    def test_normalize_dspark_draft_config_canonicalizes_prefixed_target_layers(self):
        config = _glm52_redhat_dspark_config()
        config.dspark_target_layer_ids = [3, 11, 19]

        normalize_dspark_draft_config(config)
        parsed = parse_dflash_draft_config(draft_hf_config=config)

        self.assertEqual(config.target_layer_ids, [3, 11, 19])
        self.assertEqual(config.num_target_layers, 20)
        self.assertEqual(parsed.target_layer_ids, [3, 11, 19])
        self.assertEqual(parsed.num_target_layers, 20)

    def test_dspark_capture_spec_keeps_explicit_redhat_aux_layers_for_glm52(self):
        with self.assertLogs(
            "sglang.srt.model_executor.model_runner", level="WARNING"
        ) as logs:
            capture_spec = _resolve_dflash_or_dspark_capture_spec(
                draft_hf_config=_glm52_redhat_dspark_config(),
                target_num_layers=78,
                is_dspark=True,
            )

        self.assertEqual(capture_spec.draft_num_layers, 5)
        self.assertEqual(capture_spec.target_layer_ids, [8, 23, 39, 55, 70])
        warning_text = "\n".join(logs.output)
        self.assertIn(
            "using explicit target_layer_ids=[8, 23, 39, 55, 70]",
            warning_text,
        )
        self.assertNotIn(
            "selecting capture layers based on the runtime target", warning_text
        )

    def test_dspark_capture_spec_rejects_mismatch_without_explicit_layers(self):
        config = _glm52_redhat_dspark_config()
        delattr(config, "aux_hidden_state_layer_ids")
        config.num_target_layers = 71

        with self.assertRaisesRegex(ValueError, "does not provide explicit"):
            _resolve_dflash_or_dspark_capture_spec(
                draft_hf_config=config,
                target_num_layers=78,
                is_dspark=True,
            )

    def test_dspark_capture_spec_rejects_final_layer_id(self):
        config = _glm52_redhat_dspark_config()
        config.aux_hidden_state_layer_ids = [8, 23, 77]

        with self.assertRaisesRegex(
            ValueError, "cannot include the final target layer"
        ):
            _resolve_dflash_or_dspark_capture_spec(
                draft_hf_config=config,
                target_num_layers=78,
                is_dspark=True,
            )


class TestGLM52RedHatDSparkWorkerConfig(CustomTestCase):
    def test_draft_worker_common_accepts_missing_model_type_config(self):
        server_args = _make_draft_server_args()
        missing_model_type = ValueError(
            "Unrecognized model in RedHatAI/GLM-5.2-speculator.dspark. "
            "Should have a `model_type` key in its config.json."
        )

        with (
            patch(
                "sglang.srt.utils.hf_transformers.config.get_config",
                side_effect=missing_model_type,
            ),
            patch(
                "transformers.PretrainedConfig.get_config_dict",
                return_value=(_glm52_redhat_dspark_raw_config(), {}),
            ),
        ):
            self.assertFalse(draft_is_deepseek_v4(server_args=server_args))
            self.assertEqual(
                _resolve_draft_attention_backend(
                    draft_server_args=server_args, algo_label="DSpark"
                ),
                "triton",
            )


class TestGLM52RawHeadDims(CustomTestCase):
    def test_restore_raw_glm_moe_dsa_head_dims(self):
        config = PretrainedConfig(
            architectures=["GlmMoeDsaForCausalLM"],
            qk_nope_head_dim=192,
            qk_rope_head_dim=192,
            qk_head_dim=384,
            v_head_dim=256,
        )
        raw_config = {
            "qk_nope_head_dim": 192,
            "qk_rope_head_dim": 64,
            "qk_head_dim": 256,
            "v_head_dim": 256,
        }

        _restore_glm_moe_dsa_head_dims_from_raw_config(
            config,
            raw_config,
            {},
            "unused",
            False,
            None,
            {},
        )

        self.assertEqual(config.qk_nope_head_dim, 192)
        self.assertEqual(config.qk_rope_head_dim, 64)
        self.assertEqual(config.qk_head_dim, 256)
        self.assertEqual(config.v_head_dim, 256)

    def test_json_override_wins_over_raw_head_dims(self):
        config = PretrainedConfig(
            architectures=["GlmMoeDsaForCausalLM"],
            qk_nope_head_dim=192,
            qk_rope_head_dim=192,
            qk_head_dim=384,
            v_head_dim=256,
        )

        _restore_glm_moe_dsa_head_dims_from_raw_config(
            config,
            {"qk_rope_head_dim": 64},
            {"qk_rope_head_dim": 128},
            "unused",
            False,
            None,
            {},
        )

        self.assertEqual(config.qk_rope_head_dim, 128)


if __name__ == "__main__":
    unittest.main()
