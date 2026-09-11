{
  description = "Qwen3-TTS server with AMD ROCm and Zara service tooling";

  inputs.nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";

  outputs = { self, nixpkgs }:
    let
      systems = [ "x86_64-linux" "aarch64-linux" ];
      forAllSystems = nixpkgs.lib.genAttrs systems;
    in {
      packages = forAllSystems (system:
        let
          pkgs = nixpkgs.legacyPackages.${system};
          qwen3-tts = pkgs.writeShellApplication {
            name = "qwen3-tts";
            runtimeInputs = with pkgs; [
              bash
              coreutils
              curl
              docker-client
              ffmpeg
              findutils
              gnugrep
              jq
              nix
              python3
              systemd
              util-linux
            ];
            text = ''
              export QWEN3_TTS_SOURCE=${self}
              export QWEN3_TTS_EXECUTABLE="$0"
              exec ${pkgs.bash}/bin/bash ${self}/scripts/qwen3-tts "$@"
            '';
          };
        in {
          inherit qwen3-tts;
          default = qwen3-tts;
        });

      apps = forAllSystems (system: {
        qwen3-tts = {
          type = "app";
          program = "${self.packages.${system}.qwen3-tts}/bin/qwen3-tts";
        };
        default = self.apps.${system}.qwen3-tts;
      });

      devShells = forAllSystems (system:
        let pkgs = nixpkgs.legacyPackages.${system};
        in {
          default = pkgs.mkShell {
            packages = with pkgs; [ ffmpeg python3 shellcheck ];
          };
        });

      checks = forAllSystems (system:
        let pkgs = nixpkgs.legacyPackages.${system};
        in {
          scripts = pkgs.runCommand "qwen3-tts-script-checks" {
            nativeBuildInputs = with pkgs; [ ffmpeg python3 shellcheck ];
          } ''
            export PYTHONPYCACHEPREFIX="$TMPDIR/pycache"
            shellcheck ${self}/scripts/qwen3-tts ${self}/build.sh ${self}/start.sh
            python -m py_compile ${self}/server.py ${self}/scripts/voice_manager.py
            python -m unittest discover -s ${self}/tests -v
            touch "$out"
          '';
        });

      nixosModules.default = import ./nix/nixos-module.nix { inherit self; };
      nixosModules.qwen3-tts = self.nixosModules.default;
      homeManagerModules.default = import ./nix/home-manager-module.nix { inherit self; };
      homeManagerModules.qwen3-tts = self.homeManagerModules.default;
    };
}
