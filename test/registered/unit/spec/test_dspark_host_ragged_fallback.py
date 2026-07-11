import pytest
import torch

from sglang.srt.environ import envs
from sglang.srt.speculative.dspark_components.kernels import (
    qo_indptr as _qo_indptr_mod,
)
from sglang.srt.speculative.dspark_components.dspark_verify_planner import (
    DSparkVerifyPlanner,
)
from sglang.srt.speculative.ragged_verify import RaggedVerifyLayout, RaggedVerifyMode
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=1, suite="base-a-test-cpu")

_OLD_QO_INDPTR_KERNEL_IMPL = _qo_indptr_mod._KERNEL_IMPL


def setup_module():
    _qo_indptr_mod._KERNEL_IMPL = "torch"


def teardown_module():
    _qo_indptr_mod._KERNEL_IMPL = _OLD_QO_INDPTR_KERNEL_IMPL


def _planner(mode=RaggedVerifyMode.COMPACT):
    planner = object.__new__(DSparkVerifyPlanner)
    planner._ragged_verify_mode = mode
    return planner


def test_ragged_verify_still_requires_cuda_graph_prep_by_default():
    planner = _planner()
    with (
        envs.SGLANG_PREP_IN_CUDA_GRAPH.override(False),
        envs.SGLANG_DSPARK_ALLOW_HOST_RAGGED_VERIFY.override(False),
    ):
        with pytest.raises(ValueError, match="requires SGLANG_PREP_IN_CUDA_GRAPH=1"):
            planner._require_prep_in_cuda_graph()


def test_ragged_verify_allows_explicit_host_fallback(caplog):
    planner = _planner()
    with (
        envs.SGLANG_PREP_IN_CUDA_GRAPH.override(False),
        envs.SGLANG_DSPARK_ALLOW_HOST_RAGGED_VERIFY.override(True),
    ):
        planner._require_prep_in_cuda_graph()

    assert "functionality/debug fallback" in caplog.text


def test_ragged_verify_cuda_graph_prep_path_remains_silent(caplog):
    planner = _planner()
    with (
        envs.SGLANG_PREP_IN_CUDA_GRAPH.override(True),
        envs.SGLANG_DSPARK_ALLOW_HOST_RAGGED_VERIFY.override(False),
    ):
        planner._require_prep_in_cuda_graph()

    assert caplog.text == ""


def test_compact_verify_bypasses_known_full_width_layout():
    planner = _planner()
    layout = RaggedVerifyLayout.uniform(
        bs=2, num_draft_tokens=8, device=torch.device("cpu"), grid=[16]
    )

    assert planner.should_run_compact(layout=layout) is False


def test_compact_verify_keeps_non_full_width_layout():
    planner = _planner()
    layout = RaggedVerifyLayout.from_verify_lens(
        verify_lens_cpu=[8, 4],
        device=torch.device("cpu"),
        grid=[12],
        num_draft_tokens=8,
    )

    assert planner.should_run_compact(layout=layout) is True
