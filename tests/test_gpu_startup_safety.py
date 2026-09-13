import re
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = (ROOT / "scripts" / "qwen3-tts").read_text(encoding="utf-8")
HOME_MANAGER_MODULE = (ROOT / "nix" / "home-manager-module.nix").read_text(encoding="utf-8")
NIXOS_MODULE = (ROOT / "nix" / "nixos-module.nix").read_text(encoding="utf-8")


class GpuStartupSafetyTest(unittest.TestCase):
    def test_modules_require_explicit_autostart_opt_in(self):
        for name, content in (
            ("home-manager", HOME_MANAGER_MODULE),
            ("nixos", NIXOS_MODULE),
        ):
            with self.subTest(module=name):
                self.assertIn("autoStart = mkOption", content)
                self.assertIn("default = false;", content)
                self.assertIn("lib.optionals cfg.autoStart", content)
                self.assertNotIn('Restart = "always";', content)
                self.assertIn('Restart = "on-failure";', content)

    def test_detached_container_is_not_boot_persistent_by_default(self):
        self.assertIn('RESTART_POLICY="${QWEN_RESTART_POLICY:-no}"', SCRIPT)
        self.assertNotIn("--restart=unless-stopped", SCRIPT)

    def test_install_user_leaves_service_disabled(self):
        active_enable = re.compile(
            r"^\s*systemctl --user enable --now qwen3-tts\.service\s*$",
            re.MULTILINE,
        )
        self.assertIsNone(active_enable.search(SCRIPT))
        self.assertIn(
            "systemctl --user disable --now qwen3-tts.service",
            SCRIPT,
        )
        self.assertNotIn("'Restart=always'", SCRIPT)
        self.assertIn("'Restart=on-failure'", SCRIPT)


if __name__ == "__main__":
    unittest.main()
