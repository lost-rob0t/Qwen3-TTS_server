{ self }:
{ config, lib, pkgs, ... }:

let
  inherit (lib) mkEnableOption mkIf mkMerge mkOption types;
  cfg = config.services.qwen3-tts;
in {
  options.services.qwen3-tts = {
    enable = mkEnableOption "the Qwen3-TTS service";

    autoStart = mkOption {
      type = types.bool;
      default = false;
      description = "Start Qwen3-TTS automatically at boot. Disabled by default because GPU backend failures can destabilize the host.";
    };

    package = mkOption {
      type = types.package;
      default = self.packages.${pkgs.system}.qwen3-tts;
      description = "Qwen3-TTS management package.";
    };

    backend = mkOption {
      type = types.enum [ "rocm" "cuda" "cpu" ];
      default = "rocm";
      description = "Acceleration backend.";
    };

    port = mkOption {
      type = types.port;
      default = 7860;
      description = "Host port exposed by the TTS service.";
    };

    bindAddress = mkOption {
      type = types.str;
      default = "127.0.0.1";
      description = "Host address to bind. The API is unauthenticated, so loopback is the safe default.";
    };

    dataDir = mkOption {
      type = types.path;
      default = "/var/lib/qwen3-tts";
      description = "Persistent voices, outputs, and model cache.";
    };

    user = mkOption {
      type = types.nullOr types.str;
      default = null;
      description = "Existing user to run as, or null to run as root.";
    };

    environment = mkOption {
      type = types.attrsOf types.str;
      default = {};
      example = { HSA_OVERRIDE_GFX_VERSION = "10.3.0"; };
      description = "Extra server/container environment variables.";
    };
  };

  config = mkIf cfg.enable (mkMerge [
    {
      virtualisation.docker.enable = true;

      systemd.tmpfiles.rules = [
        "d ${toString cfg.dataDir} 0750 ${if cfg.user == null then "root" else cfg.user} ${if cfg.user == null then "root" else "users"} -"
      ];

      systemd.services.qwen3-tts = {
        description = "Qwen3-TTS service";
        wantedBy = lib.optionals cfg.autoStart [ "multi-user.target" ];
        after = [ "docker.service" "network-online.target" ];
        wants = [ "docker.service" "network-online.target" ];
        unitConfig = {
          StartLimitIntervalSec = 60;
          StartLimitBurst = 2;
        };
        environment = cfg.environment // {
          QWEN_BACKEND = cfg.backend;
          QWEN_PORT = toString cfg.port;
          QWEN_BIND_ADDRESS = cfg.bindAddress;
          QWEN3_TTS_DATA_DIR = toString cfg.dataDir;
        };
        serviceConfig = {
          ExecStart = "${cfg.package}/bin/qwen3-tts serve ${cfg.backend}";
          ExecStop = "${cfg.package}/bin/qwen3-tts stop";
          Restart = "on-failure";
          RestartSec = 10;
          TimeoutStartSec = "infinity";
        } // lib.optionalAttrs (cfg.user != null) {
          User = cfg.user;
          SupplementaryGroups = [ "docker" ];
        };
      };
    }
  ]);
}
