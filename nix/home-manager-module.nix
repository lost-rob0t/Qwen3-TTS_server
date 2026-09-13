{ self }:
{ config, lib, pkgs, ... }:

let
  inherit (lib) mkEnableOption mkIf mkOption types;
  cfg = config.services.qwen3-tts;
in {
  options.services.qwen3-tts = {
    enable = mkEnableOption "the Qwen3-TTS user service";

    autoStart = mkOption {
      type = types.bool;
      default = false;
      description = "Start Qwen3-TTS automatically with the user session. Disabled by default because GPU backend failures can destabilize the desktop session.";
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

    dataDir = mkOption {
      type = types.str;
      default = "${config.home.homeDirectory}/.local/share/qwen3-tts";
      description = "Persistent voices, outputs, and model cache.";
    };

    environment = mkOption {
      type = types.attrsOf types.str;
      default = {};
      description = "Extra server/container environment variables.";
    };
  };

  config = mkIf cfg.enable {
    home.packages = [ cfg.package ];

    systemd.user.services.qwen3-tts = {
      Unit = {
        Description = "Qwen3-TTS service";
        After = [ "network-online.target" ];
        Wants = [ "network-online.target" ];
        StartLimitIntervalSec = 60;
        StartLimitBurst = 2;
      };
      Service = {
        Environment = lib.mapAttrsToList (name: value: "${name}=${value}") (cfg.environment // {
          QWEN_BACKEND = cfg.backend;
          QWEN_PORT = toString cfg.port;
          QWEN_BIND_ADDRESS = "127.0.0.1";
          QWEN3_TTS_DATA_DIR = cfg.dataDir;
        });
        ExecStart = "${cfg.package}/bin/qwen3-tts serve ${cfg.backend}";
        ExecStop = "${cfg.package}/bin/qwen3-tts stop";
        Restart = "on-failure";
        RestartSec = 10;
        TimeoutStartSec = "infinity";
      };
      Install.WantedBy = lib.optionals cfg.autoStart [ "default.target" ];
    };
  };
}
