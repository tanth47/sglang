from types import SimpleNamespace

from sglang.srt.speculative.draft_worker_common import (
    _sanitize_draft_quantization_for_config,
)
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=1, suite="base-a-test-cpu")


def _server_args(*, draft_quantization="quark", draft_model_path="draft"):
    return SimpleNamespace(
        quantization="quark",
        speculative_draft_model_quantization=draft_quantization,
        speculative_draft_model_path=draft_model_path,
    )


def test_draft_worker_clears_inherited_quark_for_unquantized_draft(tmp_path):
    server_args = _server_args(draft_model_path=str(tmp_path))
    draft_hf_config = SimpleNamespace()

    _sanitize_draft_quantization_for_config(
        draft_server_args=server_args,
        draft_hf_config=draft_hf_config,
        algo_label="DSPARK",
    )

    assert server_args.speculative_draft_model_quantization is None
    assert server_args.quantization is None


def test_draft_worker_keeps_quark_when_draft_declares_quantization(tmp_path):
    server_args = _server_args(draft_model_path=str(tmp_path))
    draft_hf_config = SimpleNamespace(quantization_config={"quant_method": "quark"})

    _sanitize_draft_quantization_for_config(
        draft_server_args=server_args,
        draft_hf_config=draft_hf_config,
        algo_label="DSPARK",
    )

    assert server_args.speculative_draft_model_quantization == "quark"
    assert server_args.quantization == "quark"


def test_draft_worker_keeps_non_quark_draft_quantization(tmp_path):
    server_args = _server_args(
        draft_quantization="fp8",
        draft_model_path=str(tmp_path),
    )
    draft_hf_config = SimpleNamespace()

    _sanitize_draft_quantization_for_config(
        draft_server_args=server_args,
        draft_hf_config=draft_hf_config,
        algo_label="DSPARK",
    )

    assert server_args.speculative_draft_model_quantization == "fp8"
    assert server_args.quantization == "fp8"

