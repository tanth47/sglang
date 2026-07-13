from types import SimpleNamespace

import torch

from sglang.srt.speculative.dflash_utils import build_dflash_verify_target_probs
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=3, suite="base-a-test-cpu")


def test_dflash_verify_target_probs_uses_torch_renorm_fallback():
    logits = torch.tensor(
        [
            [4.0, 3.0, 2.0, 1.0],
            [1.0, 4.0, 2.0, 3.0],
            [2.0, 1.0, 4.0, 3.0],
            [3.0, 2.0, 1.0, 4.0],
        ]
    )
    sampling_info = SimpleNamespace(
        temperatures=torch.ones(2),
        top_ks=torch.tensor([2, 3], dtype=torch.int64),
        top_ps=torch.tensor([0.8, 0.75]),
        need_top_k_sampling=True,
        need_top_p_sampling=True,
    )

    probs = build_dflash_verify_target_probs(
        next_token_logits=logits,
        sampling_info=sampling_info,
        draft_token_num=2,
        bs=2,
        use_sparse_topk=True,
    )

    assert probs.shape == (2, 2, 4)
    torch.testing.assert_close(
        probs.sum(dim=-1),
        torch.ones((2, 2)),
        rtol=1e-5,
        atol=1e-5,
    )
    assert torch.count_nonzero(probs[0, 0]).item() <= 2
    assert torch.count_nonzero(probs[0, 1]).item() <= 2
    assert torch.count_nonzero(probs[1, 0]).item() <= 3
    assert torch.count_nonzero(probs[1, 1]).item() <= 3
