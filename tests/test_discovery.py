import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from atlas_local.config import load_config
from atlas_local.discovery import (
    RecommendedModel,
    _estimate_model_fit,
    _normalize_windows_gpu_entries,
    _recommended_model_from_ollama_metadata,
    _resolve_windows_system_label,
    build_discovery_report,
    load_discovery_models,
)
from atlas_local.llm import OllamaCatalogSnapshot, OllamaModelInfo


class DiscoveryReportTests(unittest.TestCase):
    @patch("atlas_local.discovery.list_installed_ollama_model_names")
    @patch("atlas_local.discovery.detect_local_hardware")
    def test_report_marks_memory_degraded_and_requires_chat_model_selection(
        self,
        detect_local_hardware_mock,
        list_installed_models_mock,
    ) -> None:
        detect_local_hardware_mock.return_value = {
            "os": "Windows 11",
            "platform": "win32",
            "cpu": {"model": "AMD Ryzen", "logical_cores": 16},
            "memory": {"total_gb": 32.0},
            "gpus": [{"name": "RTX 4070", "memory_gb": 12.0}],
            "detection": {"confidence": "full", "notes": []},
        }
        list_installed_models_mock.return_value = ["gemma3:4b"]

        with tempfile.TemporaryDirectory() as temp_dir:
            config = load_config(
                project_root=Path(temp_dir),
                env={
                    "EMBED_MODEL": "nomic-embed-text:latest",
                },
            )
            catalog = OllamaCatalogSnapshot(
                models=(OllamaModelInfo(name="gemma3:4b", supports_images=True),),
                ollama_online=True,
                has_local_models=True,
                source="ollama",
            )

            payload = build_discovery_report(config, catalog)

        self.assertEqual(payload["atlas"]["status"], "memory-degraded")
        self.assertFalse(payload["atlas"]["configured_embed_model_installed"])
        self.assertTrue(
            any("Choose any installed chat model" in note for note in payload["atlas"]["notes"])
        )
        self.assertEqual(payload["installed_models"][0]["atlas_role"], "vision")

    @patch("atlas_local.discovery.list_installed_ollama_model_names")
    @patch("atlas_local.discovery.detect_local_hardware")
    def test_report_marks_ready_when_embed_model_is_installed(
        self,
        detect_local_hardware_mock,
        list_installed_models_mock,
    ) -> None:
        detect_local_hardware_mock.return_value = {
            "os": "Windows 11",
            "platform": "win32",
            "cpu": {"model": "AMD Ryzen", "logical_cores": 16},
            "memory": {"total_gb": 64.0},
            "gpus": [{"name": "RTX 4090", "memory_gb": 24.0}],
            "detection": {"confidence": "full", "notes": []},
        }
        list_installed_models_mock.return_value = ["qwen3:8b", "nomic-embed-text:latest"]

        with tempfile.TemporaryDirectory() as temp_dir:
            config = load_config(
                project_root=Path(temp_dir),
                env={
                    "EMBED_MODEL": "nomic-embed-text:latest",
                },
            )
            catalog = OllamaCatalogSnapshot(
                models=(
                    OllamaModelInfo(
                        name="qwen3:8b",
                        size_bytes=5_200_000_000,
                        supports_reasoning=True,
                    ),
                ),
                ollama_online=True,
                has_local_models=True,
                source="ollama",
            )

            payload = build_discovery_report(config, catalog)

        self.assertEqual(payload["atlas"]["status"], "ready")
        self.assertTrue(payload["atlas"]["configured_embed_model_installed"])
        self.assertEqual(payload["recommended_models"][0]["name"], "qwen3:8b")
        self.assertEqual(payload["recommended_models"][0]["fit"], "good")
        self.assertEqual(payload["recommended_models"][0]["model_size_gb"], 4.8)
        self.assertTrue(payload["recommended_models"][0]["supports_reasoning"])
        recommended_names = [item["name"] for item in payload["recommended_models"]]
        self.assertIn("llama3.1:8b", recommended_names)
        self.assertIn("qwen3:8b", recommended_names)
        self.assertIn("deepseek-r1:8b", recommended_names)
        self.assertNotIn("qwen2.5:7b", recommended_names)

    @patch("atlas_local.discovery.list_installed_ollama_model_names")
    @patch("atlas_local.discovery.detect_local_hardware")
    def test_report_uses_manifest_and_live_ollama_metadata(
        self,
        detect_local_hardware_mock,
        list_installed_models_mock,
    ) -> None:
        detect_local_hardware_mock.return_value = {
            "os": "Windows 11",
            "platform": "win32",
            "cpu": {"model": "AMD Ryzen", "logical_cores": 16},
            "memory": {"total_gb": 32.0},
            "gpus": [{"name": "RTX 4070", "memory_gb": 12.0}],
            "detection": {"confidence": "full", "notes": []},
        }
        list_installed_models_mock.return_value = ["custom-chat:1b", "live-local:latest"]

        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            manifest = root / "manifest.json"
            manifest.write_text(
                """
                {
                  "version": 1,
                  "models": [
                    {
                      "name": "custom-chat:1b",
                      "title": "Custom local chat",
                      "use_case": "chat",
                      "atlas_role": "chat",
                      "min_ram_gb": 4,
                      "good_ram_gb": 8,
                      "min_vram_gb": 2,
                      "good_vram_gb": 4
                    }
                  ]
                }
                """,
                encoding="utf-8",
            )
            config = load_config(project_root=root, env={})
            catalog = OllamaCatalogSnapshot(
                models=(
                    OllamaModelInfo(name="custom-chat:1b"),
                    OllamaModelInfo(name="live-local:latest", supports_reasoning=True),
                ),
                ollama_online=True,
                has_local_models=True,
                source="ollama",
            )
            with patch.dict("os.environ", {"ATLAS_DISCOVERY_MANIFEST": str(manifest)}):
                payload = build_discovery_report(config, catalog)
                loaded = load_discovery_models(config)

        self.assertEqual(loaded[0].name, "custom-chat:1b")
        names = [item["name"] for item in payload["recommended_models"]]
        self.assertIn("custom-chat:1b", names)
        self.assertIn("live-local:latest", names)
        live_item = next(item for item in payload["recommended_models"] if item["name"] == "live-local:latest")
        self.assertEqual(live_item["source"], "ollama")

    @patch("atlas_local.discovery.platform.version", return_value="10.0.26200")
    @patch("atlas_local.discovery.platform.release", return_value="10")
    def test_windows_build_above_22000_maps_to_windows_11(self, release_mock, version_mock) -> None:
        self.assertEqual(_resolve_windows_system_label(), "Windows 11")

    def test_windows_gpu_entries_prefer_nvidia_runtime_and_hide_integrated_vram(self) -> None:
        payload = [
            {"Name": "NVIDIA GeForce RTX 4080 Laptop GPU", "AdapterRAM": 4293918720},
            {"Name": "Intel(R) UHD Graphics", "AdapterRAM": 2147479552},
        ]

        gpus = _normalize_windows_gpu_entries(
            payload,
            nvidia_gpus=[
                {
                    "name": "NVIDIA GeForce RTX 4080 Laptop GPU",
                    "memory_gb": 12.0,
                    "kind": "dedicated",
                    "memory_source": "nvidia-smi",
                }
            ],
        )

        self.assertEqual(gpus[0]["name"], "NVIDIA GeForce RTX 4080 Laptop GPU")
        self.assertEqual(gpus[0]["memory_gb"], 12.0)
        self.assertEqual(gpus[0]["kind"], "dedicated")
        self.assertEqual(gpus[0]["memory_source"], "nvidia-smi")
        self.assertEqual(gpus[1]["kind"], "integrated")
        self.assertIsNone(gpus[1]["memory_gb"])
        self.assertEqual(gpus[1]["memory_source"], "shared")

    def test_fit_estimation_ignores_integrated_gpu_memory(self) -> None:
        candidate = RecommendedModel(
            name="qwen3:8b",
            title="Current Qwen all-rounder",
            use_case="reasoning",
            atlas_role="chat",
            min_ram_gb=12.0,
            good_ram_gb=18.0,
            min_vram_gb=8.0,
            good_vram_gb=12.0,
        )
        system = {
            "memory": {"total_gb": 20.0},
            "gpus": [
                {"name": "Intel(R) UHD Graphics", "memory_gb": 16.0, "kind": "integrated"},
            ],
        }

        fit, runtime, _reason = _estimate_model_fit(candidate, system)

        self.assertEqual(fit, "tight")
        self.assertEqual(runtime, "CPU")

    def test_live_model_size_drives_full_gpu_and_hybrid_fit_thresholds(self) -> None:
        candidate = _recommended_model_from_ollama_metadata(
            OllamaModelInfo(
                name="qwen3.8:27b",
                size_bytes=17_741_872_154,
                supports_images=True,
                supports_reasoning=True,
            )
        )

        fit, runtime, reason = _estimate_model_fit(
            candidate,
            {
                "memory": {"total_gb": 31.1},
                "gpus": [
                    {
                        "name": "RTX 4080 Laptop GPU",
                        "memory_gb": 12.0,
                        "kind": "dedicated",
                    }
                ],
            },
        )

        self.assertEqual(candidate.title, "Installed vision and reasoning model")
        self.assertEqual(candidate.model_size_gb, 16.5)
        self.assertEqual(candidate.min_ram_gb, 23.0)
        self.assertEqual(candidate.good_ram_gb, 31.0)
        self.assertEqual(candidate.min_vram_gb, 18.0)
        self.assertEqual(candidate.good_vram_gb, 20.0)
        self.assertEqual(fit, "tight")
        self.assertEqual(runtime, "Hybrid")
        self.assertIn("mixed CPU/GPU offload", reason)

    def test_fit_estimation_uses_partial_dedicated_gpu_for_hybrid_runtime(self) -> None:
        candidate = RecommendedModel(
            name="local:8b",
            title="Local model",
            use_case="chat",
            atlas_role="chat",
            min_ram_gb=12.0,
            good_ram_gb=18.0,
            min_vram_gb=8.0,
            good_vram_gb=12.0,
        )

        fit, runtime, reason = _estimate_model_fit(
            candidate,
            {
                "memory": {"total_gb": 14.0},
                "gpus": [
                    {"name": "Dedicated GPU", "memory_gb": 4.0, "kind": "dedicated"},
                ],
            },
        )

        self.assertEqual(fit, "tight")
        self.assertEqual(runtime, "Hybrid")
        self.assertIn("below the full-model estimate", reason)
