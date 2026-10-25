"""Fixed-voice AI host entry point. Import and --check-config never load models."""
from __future__ import annotations

import argparse
import os
from pathlib import Path

from vtuber_profile import (SERVICE_VERSION, ASSET_KEYS, load_voice, prepare_profile,
                            finish_profile, preload_reference, check_frontend_assets, validate_loaded_binding)


def main():
    parser = argparse.ArgumentParser(description="GPT-SoVITS AI VTuber adapter")
    parser.add_argument("--profile", default="vtuber.local.json")
    parser.add_argument("--tts-config", default="GPT_SoVITS/configs/tts_infer.yaml")
    parser.add_argument("--host", choices=("127.0.0.1", "::1"), default="127.0.0.1")
    parser.add_argument("--port", type=int, default=9881)
    parser.add_argument("--token-env", default="VTUBER_TTS_TOKEN")
    parser.add_argument("--check-config", action="store_true")
    parser.add_argument("--version", action="version", version=SERVICE_VERSION)
    args = parser.parse_args()
    voice = load_voice(args.profile)
    if not 1024 <= args.port <= 65535:
        parser.error("Port must be in 1024..65535")
    if args.check_config:
        print("Voice configuration structure is valid; no models, reference audio or devices were loaded.")
        return
    token = os.environ.get(args.token_env, "")
    if len(token) < 24 or not token.isascii() or any(c.isspace() for c in token):
        parser.error("Token environment variable must contain at least 24 non-whitespace ASCII characters")
    root = Path(__file__).resolve().parent
    config_file = Path(args.tts_config).resolve()
    # Refuse upstream's missing-file/default-path fallback before importing TTS.
    import yaml
    data = yaml.safe_load(config_file.read_text(encoding="utf-8"))
    if not isinstance(data, dict) or not isinstance(data.get("custom"), dict):
        parser.error("TTS YAML requires an explicit custom section")
    for key in ASSET_KEYS:
        value = data["custom"].get(key)
        if not isinstance(value, str) or not value or not (root / value).exists():
            parser.error(f"Missing configured model asset: {key}")
    check_frontend_assets(data["custom"].get("version"), voice, root)
    os.environ["bert_path"] = str((root / data["custom"]["bert_base_path"]).resolve())
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    from tools.portable_runtime import activate
    activate(change_cwd=True)
    from GPT_SoVITS.TTS_infer_pack.TTS import TTS, TTS_Config
    from GPT_SoVITS.TTS_infer_pack.streaming import validate_streaming_options
    from GPT_SoVITS.TTS_infer_pack.text_segmentation_method import splits
    from process_ckpt import get_sovits_version_from_path_fast
    config = TTS_Config(data)
    local_config = root / ".local/vtuber-runtime/tts_infer.yaml"
    local_config.parent.mkdir(parents=True, exist_ok=True)
    config.configs_path = str(local_config)
    validate_streaming_options({**voice["options"], "prompt_text": voice["prompt_text"], "media_type": "raw"},
                               config.version, api_mode=True)
    _, model_version, is_lora = get_sovits_version_from_path_fast(config.vits_weights_path)
    extras = {"lora_base": root / config.default_configs[model_version]["vits_weights_path"]} if is_lora else {}
    bundle = prepare_profile(config, voice, root, extras)
    engine = TTS(config)
    validate_loaded_binding(bundle, engine, root)
    preload_reference(engine, bundle, splits)
    profile = finish_profile(bundle, engine, root)
    from vtuber_service import create_app
    import uvicorn
    print("profile_fingerprint=" + profile["fingerprint"])
    uvicorn.run(create_app(engine, profile, token), host=args.host, port=args.port, workers=1, access_log=False)


if __name__ == "__main__":
    main()
