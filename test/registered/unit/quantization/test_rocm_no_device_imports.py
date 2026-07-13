import os
import subprocess
import sys
import textwrap
import unittest

import torch

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=15, suite="base-a-test-cpu")


class TestRocmNoDeviceImports(CustomTestCase):
    @unittest.skipIf(
        torch.version.hip is None or torch.cuda.is_available(),
        "requires a ROCm build without a visible HIP device",
    )
    def test_rocm_quant_modules_import_without_visible_device(self):
        code = textwrap.dedent("""
            import sglang.kernels.ops.attention.rocm_mla_decode_rope  # noqa: F401
            import sglang.srt.layers.quantization.fp8_kernel  # noqa: F401
            import sglang.srt.layers.quantization.fp8_utils  # noqa: F401
            import sglang.srt.layers.quantization.mxfp4  # noqa: F401
            import sglang.srt.layers.quantization.quark.utils  # noqa: F401
            import sglang.srt.layers.quantization.quark.schemes.quark_w4a4_mxfp4  # noqa: F401
            import sglang.srt.layers.quantization.quark.schemes.quark_w4a4_mxfp4_moe  # noqa: F401
            import sglang.srt.layers.quantization.quark_int4fp8_moe  # noqa: F401
            """)
        env = os.environ.copy()
        env["SGLANG_USE_AITER"] = "0"

        subprocess.run([sys.executable, "-c", code], check=True, env=env)

    @unittest.skipIf(
        torch.version.hip is None or torch.cuda.is_available(),
        "requires a ROCm build without a visible HIP device",
    )
    def test_rocm_dsa_imports_with_aiter_env_without_visible_device(self):
        code = textwrap.dedent("""
            import torch

            assert torch.version.hip is not None
            assert not torch.cuda.is_available()

            import sglang.srt.layers.attention.dsa.index_buf_accessor  # noqa: F401
            import sglang.srt.layers.attention.dsa.dsa_indexer  # noqa: F401
        """)
        env = os.environ.copy()
        env["SGLANG_USE_AITER"] = "1"

        subprocess.run([sys.executable, "-c", code], check=True, env=env)

    @unittest.skipIf(
        torch.version.hip is None or torch.cuda.is_available(),
        "requires a ROCm build without a visible HIP device",
    )
    def test_rocm_quantization_package_imports_with_aiter_env_without_visible_device(
        self,
    ):
        code = textwrap.dedent("""
            import torch

            assert torch.version.hip is not None
            assert not torch.cuda.is_available()

            import sglang.srt.layers.quantization  # noqa: F401
        """)
        env = os.environ.copy()
        env["SGLANG_USE_AITER"] = "1"

        subprocess.run([sys.executable, "-c", code], check=True, env=env)

    @unittest.skipIf(
        torch.version.hip is None or torch.cuda.is_available(),
        "requires a ROCm build without a visible HIP device",
    )
    def test_rocm_attention_kernels_import_without_visible_device(self):
        code = textwrap.dedent("""
            import torch

            assert torch.version.hip is not None
            assert not torch.cuda.is_available()

            import sglang.kernels.ops.attention.extend_attention  # noqa: F401
            import sglang.kernels.ops.attention.prefill_attention  # noqa: F401
        """)
        env = os.environ.copy()
        env["SGLANG_USE_AITER"] = "1"

        subprocess.run([sys.executable, "-c", code], check=True, env=env)


if __name__ == "__main__":
    unittest.main()
