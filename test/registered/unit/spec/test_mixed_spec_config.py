import types
import unittest

from sglang.srt.arg_groups.speculative_hook import _validate_mixed_spec_chunk
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=2, suite="base-a-test-cpu")


def _valid_args(**overrides):
    values = dict(
        enable_mixed_spec_chunk=True,
        enable_mixed_chunk=False,
        speculative_algorithm="EAGLE",
        chunked_prefill_size=4096,
        speculative_num_steps=5,
        speculative_eagle_topk=1,
        speculative_num_draft_tokens=6,
        speculative_attention_mode="prefill",
        tp_size=4,
        dp_size=1,
        pp_size=1,
        speculative_adaptive=False,
        speculative_use_rejection_sampling=False,
        disaggregation_mode="null",
        enable_lora=False,
    )
    values.update(overrides)
    return types.SimpleNamespace(**values)


def _valid_view(**overrides):
    values = dict(
        disable_overlap_schedule=True,
        disable_cuda_graph=True,
        enable_dp_attention=False,
        dsa_prefill_backend="tilelang",
        dsa_decode_backend="tilelang",
        enable_multi_layer_eagle=False,
        enable_hisparse=False,
        enable_hierarchical_cache=False,
    )
    values.update(overrides)
    return types.SimpleNamespace(**values)


class TestMixedSpecConfig(unittest.TestCase):
    def test_accepts_narrow_glm52_target_only_scope(self):
        for tp_size in (4, 8):
            with self.subTest(tp_size=tp_size):
                _validate_mixed_spec_chunk(
                    _valid_args(tp_size=tp_size),
                    "GlmMoeDsaForCausalLM",
                    _valid_view(),
                )

    def test_rejects_overlap_and_wrong_verify_shape_together(self):
        with self.assertRaisesRegex(ValueError, "disable-overlap-schedule") as cm:
            _validate_mixed_spec_chunk(
                _valid_args(speculative_num_steps=3),
                "GlmMoeDsaForCausalLM",
                _valid_view(disable_overlap_schedule=False),
            )

        self.assertIn("speculative-num-steps 5", str(cm.exception))
        self.assertIn("Disable the flag", str(cm.exception))

    def test_rejects_legacy_and_spec_mixed_modes_together(self):
        with self.assertRaisesRegex(ValueError, "enable-mixed-chunk disabled"):
            _validate_mixed_spec_chunk(
                _valid_args(enable_mixed_chunk=True),
                "GlmMoeDsaForCausalLM",
                _valid_view(),
            )

    def test_rejects_non_prefill_speculative_attention_mode(self):
        with self.assertRaisesRegex(ValueError, "speculative-attention-mode prefill"):
            _validate_mixed_spec_chunk(
                _valid_args(speculative_attention_mode="decode"),
                "GlmMoeDsaForCausalLM",
                _valid_view(),
            )

    def test_flag_off_preserves_all_existing_combinations(self):
        _validate_mixed_spec_chunk(
            types.SimpleNamespace(enable_mixed_spec_chunk=False),
            "Anything",
            types.SimpleNamespace(),
        )


if __name__ == "__main__":
    unittest.main()
