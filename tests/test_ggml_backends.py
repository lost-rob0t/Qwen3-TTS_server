import re
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = (ROOT / "scripts" / "qwen3-tts").read_text(encoding="utf-8")
HOME_MANAGER_MODULE = (ROOT / "nix" / "home-manager-module.nix").read_text(encoding="utf-8")
NIXOS_MODULE = (ROOT / "nix" / "nixos-module.nix").read_text(encoding="utf-8")

PINNED_TAG = "71ad93d591a2811f35db77e27c02acba091c9e9b"
TALKER_SHA256 = "d54dbaf10591421fa764ed630d764efa717ae40cd959bd48c66d4eb1af226426"
CODEC_SHA256 = "1883beeed99348fc35e23dd225e9082f93f6f8c109330a33d935baa8acdbfd94"
MODEL_COMMIT = "b7ee2e8c7459c3bea99da23e3d178125a7d1713c"


class GgmlBackendSelectionTest(unittest.TestCase):
    def test_vulkan_is_the_default_backend(self):
        self.assertIn('BACKEND="${QWEN_BACKEND:-vulkan}"', SCRIPT)

    def test_rocm_path_is_fully_removed(self):
        self.assertNotIn("rocm", SCRIPT.lower())
        self.assertNotIn("/dev/kfd", SCRIPT)
        self.assertNotIn("HSA_OVERRIDE", SCRIPT)

    def test_backend_enum_is_vulkan_cuda_cpu(self):
        self.assertIn("vulkan|cuda|cpu) return 0 ;;", SCRIPT)
        self.assertIn("expected vulkan, cuda, or cpu", SCRIPT)

    def test_images_are_pinned_to_the_verified_upstream_commit(self):
        self.assertIn("ghcr.io/serveurpersocom/qwentts.cpp", SCRIPT)
        self.assertIn(PINNED_TAG, SCRIPT)

    def test_vulkan_runs_on_render_nodes_only(self):
        device_args = SCRIPT[SCRIPT.index("device_args()") : SCRIPT.index("environment_args()")]
        vulkan_branch = device_args[: device_args.index("cuda)")]
        self.assertIn("--device=/dev/dri", vulkan_branch)
        self.assertNotIn("/dev/kfd", device_args)
        self.assertIn("renderD128", vulkan_branch)
        self.assertNotIn("--gpus=all", vulkan_branch)

    def test_pull_validates_backend_before_touching_the_registry(self):
        pull_case = re.search(r"^\s+pull\)\n(.*?);;", SCRIPT, re.MULTILINE | re.DOTALL)
        self.assertIsNotNone(pull_case)
        self.assertLess(pull_case.group(1).index("require_backend"), pull_case.group(1).index("ensure_image"))


class GgmlModelProvisioningTest(unittest.TestCase):
    def test_default_models_are_the_low_latency_variants(self):
        self.assertIn("qwen-talker-0.6b-base-Q8_0.gguf", SCRIPT)
        self.assertIn("qwen-tokenizer-12hz-Q8_0.gguf", SCRIPT)

    def test_model_urls_are_commit_pinned(self):
        self.assertIn(MODEL_COMMIT, SCRIPT)
        self.assertIn("https://huggingface.co/Serveurperso/Qwen3-TTS-GGUF/resolve/", SCRIPT)

    def test_model_downloads_verify_sha256(self):
        self.assertIn(TALKER_SHA256, SCRIPT)
        self.assertIn(CODEC_SHA256, SCRIPT)
        self.assertIn("sha256sum", SCRIPT)

    def test_models_land_in_data_dir_models(self):
        self.assertIn("/models", SCRIPT)
        self.assertIn('"$DATA_DIR/models:/models"', SCRIPT)


class GgmlServerContractTest(unittest.TestCase):
    def test_health_probe_uses_the_openai_server_health_route(self):
        self.assertIn("/health", SCRIPT)
        self.assertNotIn("/synthesize_speech/", SCRIPT)

    def test_container_ports_are_owner_private_by_default(self):
        self.assertIn('BIND_ADDRESS="${QWEN_BIND_ADDRESS:-127.0.0.1}"', SCRIPT)

    def test_ggml_vulkan_escape_hatch_envs_pass_through(self):
        for variable in (
            "GGML_VK_ALLOW_GRAPHICS_QUEUE",
            "GGML_VK_ASYNC_USE_TRANSFER_QUEUE",
            "GGML_VK_SERIALIZE_SUBMISSIONS",
            "RADV_PERFTEST",
        ):
            with self.subTest(variable=variable):
                self.assertIn(variable, SCRIPT)

    def test_doctor_no_longer_probes_torch(self):
        doctor = SCRIPT[SCRIPT.index("doctor()") : SCRIPT.index("install_user_service()")]
        self.assertNotIn("torch", doctor)

    def test_doctor_warns_that_live_gpu_validation_is_separate(self):
        doctor = SCRIPT[SCRIPT.index("doctor()") : SCRIPT.index("install_user_service()")]
        self.assertIn("bounded", doctor)


class GgmlNixModuleTest(unittest.TestCase):
    def test_home_manager_backend_enum_matches_script(self):
        self.assertIn('types.enum [ "vulkan" "cuda" "cpu" ]', HOME_MANAGER_MODULE)
        self.assertIn('default = "vulkan";', HOME_MANAGER_MODULE)

    def test_nixos_backend_enum_matches_script(self):
        self.assertIn('types.enum [ "vulkan" "cuda" "cpu" ]', NIXOS_MODULE)
        self.assertIn('default = "vulkan";', NIXOS_MODULE)


if __name__ == "__main__":
    unittest.main()
