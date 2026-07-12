import json
import tempfile
import unittest
from pathlib import Path

from benchmark import dspark_launch_report as report
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=1, suite="base-a-test-cpu")


class TestDSparkLaunchReport(CustomTestCase):
    def test_build_report_from_launch_log_and_cache_dir(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            log = root / "server.log"
            cache = root / "cache"
            nested = cache / "nested"
            nested.mkdir(parents=True)
            (cache / "module.so").write_bytes(b"abc")
            (nested / "kernel.cpp").write_bytes(b"12345")
            log.write_text(
                "\n".join(
                    [
                        "Sat Jul 11 19:03:15 UTC 2026",
                        "Multi-thread loading shards:   0% Completed | 0/141 [00:00<?, ?it/s] Multi-thread loading shards: 100% Completed | 141/141 [00:22<00:00,  6.30it/s]",
                        "[2026-07-11 19:06:10 TP3] Load weight end. elapsed=114.88 s, type=GlmMoeDsaForCausalLM, quant=fp8.",
                        "[2026-07-11 19:06:11 TP2] Load weight end. elapsed=115.28 s, type=GlmMoeDsaForCausalLM, quant=fp8.",
                        "[2026-07-11 19:06:48 TP1] Load weight end. elapsed=152.65 s, type=GlmMoeDsaForCausalLM, quant=fp8.",
                        "[2026-07-11 19:07:02 TP0] Load weight end. elapsed=167.05 s, type=GlmMoeDsaForCausalLM, quant=fp8.",
                        "Multi-thread loading shards:   0% Completed | 0/1 [00:00<?, ?it/s] Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00, 940.43it/s]",
                        "[2026-07-11 19:07:08 TP3] Load weight end. elapsed=0.82 s, type=DSparkDraftModel, avail mem=107.95 GB.",
                        "[2026-07-11 19:07:26 TP3] Capture draft verify CUDA graph end. elapsed=14.74 s, mem usage=1.53 GB.",
                        "[aiter] import [module_moe] under /sgl-workspace/aiter/aiter/jit/module_moe.so",
                        "[aiter] [pid=254 pname=Process-4] start build [module_moe] under /sgl-workspace/aiter/aiter/jit/build/module_moe",
                        "[aiter] [pid=254 pname=Process-4] \x1b[32mfinish build [module_moe], cost 57.8s \x1b[0m",
                        "[aiter] shape is M:6, N:6144, K:4096, not found tuned config in /tmp/aiter_configs/a8w8.csv, will use default config!",
                        "[2026-07-11 19:09:06] The server is fired up and ready to roll!",
                    ]
                )
                + "\n",
                encoding="utf-8",
            )

            launch_report = report.build_report(
                runs=[("r4b", log)],
                cache_dirs=[("aiter_jit", cache)],
                provenance={"metadata": {"git_commit": "abc123"}},
            )
            run = launch_report["runs"][0]

            self.assertEqual(launch_report["schema"], report.SCHEMA)
            self.assertEqual(
                launch_report["provenance"]["metadata"]["git_commit"], "abc123"
            )
            self.assertEqual(run["start_to_ready_s"], 351.0)
            self.assertEqual(run["target_weight_load"]["rank_count"], 4)
            self.assertEqual(run["target_weight_load"]["slowest_rank"], 0)
            self.assertAlmostEqual(run["target_weight_load"]["max_s"], 167.05)
            self.assertAlmostEqual(run["target_weight_load"]["skew_s"], 52.17)
            self.assertEqual(run["target_shard_loading_progress"]["event_count"], 1)
            self.assertEqual(
                run["target_shard_loading_progress"]["total_shards_max"], 141
            )
            self.assertAlmostEqual(
                run["target_shard_loading_progress"]["elapsed_s_max"], 22.0
            )
            self.assertEqual(run["draft_shard_loading_progress"]["event_count"], 1)
            self.assertEqual(run["draft_shard_loading_progress"]["total_shards_max"], 1)
            self.assertEqual(run["draft_verify_graph_capture"]["rank_count"], 1)
            self.assertAlmostEqual(run["draft_weight_load"]["max_s"], 0.82)
            self.assertAlmostEqual(run["draft_verify_graph_capture"]["max_s"], 14.74)
            self.assertEqual(run["aiter_builds"][0]["module"], "module_moe")
            self.assertAlmostEqual(run["aiter_builds"][0]["total_cost_s"], 57.8)
            self.assertAlmostEqual(run["aiter_builds"][0]["pre_ready_cost_s"], 57.8)
            self.assertAlmostEqual(run["aiter_builds"][0]["post_ready_cost_s"], 0.0)
            self.assertEqual(run["aiter_build_events"][0]["phase"], "pre_ready")
            self.assertEqual(
                run["aiter_imports"][0]["path"],
                "/sgl-workspace/aiter/aiter/jit/module_moe.so",
            )
            self.assertEqual(run["tuned_config_misses"][0]["m"], 6)
            self.assertEqual(run["tuned_config_miss_summary"]["event_count"], 1)
            self.assertEqual(run["tuned_config_miss_summary"]["unique_shape_count"], 1)
            self.assertEqual(
                run["launch_insight"]["dominant_observed_stage"]["stage"],
                "target_weight_load",
            )
            self.assertAlmostEqual(
                run["launch_insight"]["dominant_observed_stage"]["seconds"], 167.05
            )
            self.assertAlmostEqual(
                run["launch_insight"]["target_shard_progress_elapsed_s"], 22.0
            )
            self.assertEqual(
                run["launch_insight"]["aiter_cache_action"],
                "prewarm_or_reuse_aiter_jit_cache",
            )
            self.assertEqual(
                run["launch_insight"]["tuned_miss_action"],
                "generate_tuned_miss_inputs",
            )
            self.assertEqual(launch_report["cache_dirs"][0]["file_count"], 2)
            self.assertEqual(launch_report["cache_dirs"][0]["total_size_bytes"], 8)
            self.assertIn("sha256", run["log"])

            markdown = report.render_markdown(launch_report)
            self.assertIn(
                "| r4b | 351.00 | 4 | 167.05 | 52.17 | 1 | 14.74 | 57.80 | 57.80 | 1 | 1 |",
                markdown,
            )
            self.assertIn(
                "| r4b | target_weight_load | 167.05 | prewarm_or_reuse_aiter_jit_cache | generate_tuned_miss_inputs |",
                markdown,
            )
            self.assertIn(
                "| r4b | target | GlmMoeDsaForCausalLM | 141 | 141 | 22.00 | 2 | 3 |",
                markdown,
            )
            self.assertIn("aiter_jit", markdown)

    def test_launch_insight_preserves_warm_aiter_cache(self):
        with tempfile.TemporaryDirectory() as td:
            log = Path(td) / "server.log"
            log.write_text(
                "\n".join(
                    [
                        "Sat Jul 11 19:03:15 UTC 2026",
                        "[2026-07-11 19:04:52 TP0] Load weight end. elapsed=97.58 s, type=GlmMoeDsaForCausalLM, quant=fp8.",
                        "[aiter] import [module_moe] under /sgl-workspace/aiter/aiter/jit/module_moe.so",
                        "[2026-07-11 19:06:39] The server is fired up and ready to roll!",
                    ]
                )
                + "\n",
                encoding="utf-8",
            )

            launch_report = report.build_report(runs=[("p9c", log)], cache_dirs=[])
            insight = launch_report["runs"][0]["launch_insight"]

            self.assertEqual(insight["aiter_build_pre_ready_s"], 0.0)
            self.assertEqual(
                insight["aiter_cache_action"], "preserve_warm_aiter_jit_cache"
            )
            self.assertEqual(insight["tuned_miss_action"], "no_tuned_miss_action")

    def test_parse_without_start_line_uses_first_timestamp(self):
        with tempfile.TemporaryDirectory() as td:
            log = Path(td) / "server.log"
            log.write_text(
                "\n".join(
                    [
                        "[2026-07-11 19:16:33] server args",
                        "[2026-07-11 19:21:33] The server is fired up and ready to roll!",
                    ]
                )
                + "\n",
                encoding="utf-8",
            )

            launch_report = report.build_report(runs=[("r4c", log)], cache_dirs=[])

            self.assertEqual(launch_report["runs"][0]["start_to_ready_s"], 300.0)

    def test_json_report_is_stable_serializable(self):
        with tempfile.TemporaryDirectory() as td:
            log = Path(td) / "server.log"
            log.write_text(
                "[2026-07-11 19:16:33] server args\n",
                encoding="utf-8",
            )

            launch_report = report.build_report(runs=[("no_ready", log)], cache_dirs=[])
            encoded = json.dumps(launch_report, sort_keys=True)

            self.assertIn("no_ready", encoded)
            self.assertIsNone(launch_report["runs"][0]["start_to_ready_s"])

    def test_tuned_miss_exports(self):
        launch_report = {
            "runs": [
                {
                    "label": "r4c",
                    "tuned_config_misses": [
                        {
                            "config": "/tmp/aiter_configs/a8w8.csv",
                            "m": 8,
                            "n": 6144,
                            "k": 4096,
                            "count": 3,
                        }
                    ],
                },
                {
                    "label": "p9c",
                    "tuned_config_misses": [
                        {
                            "config": "/tmp/aiter_configs/bf16.csv",
                            "m": 32,
                            "n": 3072,
                            "k": 6144,
                            "count": 2,
                        }
                    ],
                },
            ]
        }

        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            csv_path = root / "misses.csv"
            jsonl_path = root / "misses.jsonl"

            report.write_tuned_misses_csv(csv_path, launch_report)
            report.write_tuned_misses_jsonl(jsonl_path, launch_report)

            self.assertEqual(
                csv_path.read_text(encoding="utf-8").splitlines(),
                [
                    "run,config,m,n,k,count",
                    "r4c,/tmp/aiter_configs/a8w8.csv,8,6144,4096,3",
                    "p9c,/tmp/aiter_configs/bf16.csv,32,3072,6144,2",
                ],
            )
            self.assertEqual(
                [
                    json.loads(line)
                    for line in jsonl_path.read_text(encoding="utf-8").splitlines()
                ],
                [
                    {
                        "run": "r4c",
                        "config": "/tmp/aiter_configs/a8w8.csv",
                        "m": 8,
                        "n": 6144,
                        "k": 4096,
                        "count": 3,
                    },
                    {
                        "run": "p9c",
                        "config": "/tmp/aiter_configs/bf16.csv",
                        "m": 32,
                        "n": 3072,
                        "k": 6144,
                        "count": 2,
                    },
                ],
            )


if __name__ == "__main__":
    unittest.main()
