"""Fixed voice configuration and content identity, without importing ML packages."""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

SERVICE_VERSION = "0.1.0"
UPSTREAM_REVISION = "b4570b876f5c086819ba84e3215472733c6a7b70"
ASSET_KEYS = ("t2s_weights_path", "vits_weights_path", "bert_base_path", "cnhuhbert_base_path")
LEXICON_FILES = ("opencpop-strict.txt", "cmudict.rep", "cmudict-fast.rep", "engdict-hot.rep",
                 "engdict_cache.pickle", "namedict_cache.pickle")
DEFAULT_OPTIONS = {
    "streaming_mode": 1, "gpt_streaming": False, "batch_size": 1,
    "text_split_method": "cut0", "parallel_infer": False, "split_bucket": False,
    "fragment_interval": 0.0, "speed_factor": 1.0, "seed": -1,
    "top_k": 15, "top_p": 1.0, "temperature": 1.0, "repetition_penalty": 1.35,
    "min_chunk_length": 16, "overlap_length": 2, "streaming_window_mode": "fixed",
    "streaming_chunk_seconds": 2.0, "streaming_lookahead_seconds": 0.12,
    "streaming_overlap_seconds": 0.04, "gpt_streaming_lookahead_seconds": 0.5,
    "use_cuda_graph": True, "use_flash_attention": True, "super_sampling": False,
}


def digest(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True,
                                    separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def stamp(path):
    st = Path(path).stat()
    return st.st_size, st.st_mtime_ns, st.st_ctime_ns, st.st_ino


def file_hash(path):
    before = stamp(path)
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    if before != stamp(path):
        raise ValueError("Asset changed while computing its fingerprint")
    return h.hexdigest()


def load_voice(path):
    path = Path(path).resolve()
    value = json.loads(path.read_text(encoding="utf-8"))
    required = {"reference_audio", "prompt_text", "prompt_lang", "text_lang"}
    if not isinstance(value, dict) or required - value.keys() or value.keys() - (required | {"options"}):
        raise ValueError("Voice profile has missing or unknown fields")
    for name in required:
        if not isinstance(value[name], str) or not value[name].strip() or len(value[name]) > 2048:
            raise ValueError(f"Invalid profile field: {name}")
    raw_options = value.get("options", {})
    allowed = DEFAULT_OPTIONS.keys() | {"sample_steps", "cfg_rate"}
    if not isinstance(raw_options, dict) or raw_options.keys() - allowed:
        raise ValueError("Unsupported voice option")
    options = {**DEFAULT_OPTIONS, **raw_options}
    if type(options["streaming_mode"]) is not int or options["streaming_mode"] not in (1, 2, 3):
        raise ValueError("streaming_mode must be integer 1, 2 or 3")
    for name in ("gpt_streaming", "parallel_infer", "split_bucket", "use_cuda_graph", "use_flash_attention", "super_sampling"):
        if type(options[name]) is not bool:
            raise ValueError(f"{name} must be boolean")
    if options["parallel_infer"] or options["split_bucket"] or options["super_sampling"]:
        raise ValueError("This service requires parallel_infer, split_bucket and super_sampling=false")
    if type(options["batch_size"]) is not int or options["batch_size"] != 1 or options["text_split_method"] != "cut0":
        raise ValueError("Approved speech units require batch_size=1 and cut0")
    if options["streaming_mode"] in (2, 3) and not options["gpt_streaming"]:
        raise ValueError("Modes 2/3 require gpt_streaming=true in this service")
    if options["gpt_streaming"] and options["streaming_mode"] not in (2, 3):
        raise ValueError("gpt_streaming requires mode 2/3")
    if options["streaming_window_mode"] not in ("fixed", "hybrid"):
        raise ValueError("Invalid streaming window mode")
    for name, low, high in (("top_k", 1, 100), ("seed", -1, 2**32-1),
                            ("min_chunk_length", 1, 128), ("overlap_length", 1, 32), ("sample_steps", 1, 100)):
        if name in options and (type(options[name]) is not int or not low <= options[name] <= high):
            raise ValueError(f"Invalid integer option: {name}")
    for name, low, high in (("top_p", .001, 1), ("temperature", .01, 2), ("repetition_penalty", .1, 10),
                            ("speed_factor", .5, 2), ("fragment_interval", 0, 1), ("cfg_rate", 0, 10),
                            ("streaming_chunk_seconds", .2, 5), ("streaming_lookahead_seconds", 0, .5),
                            ("streaming_overlap_seconds", 0, .2), ("gpt_streaming_lookahead_seconds", .16, 2)):
        if name in options and (type(options[name]) not in (int, float) or not math.isfinite(options[name])
                                or not low <= options[name] <= high):
            raise ValueError(f"Invalid numeric option: {name}")
    return {**value, "options": options,
            "reference_audio": str((path.parent / value["reference_audio"]).resolve())}


def _asset_files(path):
    path = Path(path).resolve()
    if path.is_file():
        return [("file", path)]
    if not path.is_dir():
        raise ValueError("Required model asset does not exist")
    files = [(p.relative_to(path).as_posix(), p) for p in sorted(path.rglob("*"))
             if p.is_file() and p.suffix != ".pyc" and not any(part.startswith(".") or part == "__pycache__"
                                                             for part in p.relative_to(path).parts)]
    if not files:
        raise ValueError("Model asset directory is empty")
    return files


def uses_g2pw(version, voice):
    return version != "v1" and any(lang in {"zh", "all_zh", "auto", "auto_yue"}
                                  for lang in (voice["prompt_lang"], voice["text_lang"]))


def check_frontend_assets(version, voice, root):
    if not uses_g2pw(version, voice):
        return
    directory = Path(root) / "GPT_SoVITS/text/G2PWModel"
    required = ("config.py", "POLYPHONIC_CHARS.txt", "MONOPHONIC_CHARS.txt",
                "bopomofo_to_pinyin_wo_tune_dict.json", "char_bopomofo_dict.json")
    if (not directory.is_dir() or not all((directory / name).is_file() for name in required)
            or not any((directory / name).is_file() for name in ("g2pW.onnx", "g2pw.onnx"))):
        raise ValueError("Local G2PW assets are missing; this entry point will not download them")


def prepare_profile(config, voice, root, extra_assets=None):
    """Hash assets before model loading; snapshot the small reference audio."""
    root = Path(root).resolve()
    version = str(config.version)
    assets = {key: root / getattr(config, key) for key in ASSET_KEYS}
    assets.update(extra_assets or {})
    # These shipped dictionaries also affect v1 and Chinese/English mixed text.
    for name in LEXICON_FILES:
        assets[name] = root / "GPT_SoVITS/text" / name
    if uses_g2pw(version, voice):
        check_frontend_assets(version, voice, root)
        assets["g2pw"] = root / "GPT_SoVITS/text/G2PWModel"
        for name in ("g2pw/polyphonic.rep", "g2pw/polyphonic-fix.rep"):
            assets[name] = root / "GPT_SoVITS/text" / name
        for name in ("g2pw/polyphonic.pickle", "g2pw/polyphonic.md5"):
            path = root / "GPT_SoVITS/text" / name
            if path.is_file():
                assets[name] = path
    if version in {"v2Pro", "v2ProPlus"}:
        assets["speaker_encoder"] = root / "GPT_SoVITS/pretrained_models/sv/pretrained_eres2netv2w24s4ep4.ckpt"
    if version == "v3":
        assets["vocoder"] = root / "GPT_SoVITS/pretrained_models/models--nvidia--bigvgan_v2_24khz_100band_256x"
    elif version in {"v4", "v5turbo", "v5dev"}:
        series = "v5" if version.startswith("v5") else "v4"
        assets["vocoder"] = root / f"GPT_SoVITS/pretrained_models/gsv-{series}-pretrained/vocoder.pth"
    records, stamps = {}, {}
    for name, path in assets.items():
        records[name] = {}
        for relative, file in _asset_files(path):
            initial = stamp(file)
            records[name][relative] = file_hash(file)
            if stamp(file) != initial:
                raise ValueError("Model asset changed during fingerprint construction")
            stamps[str(file.resolve())] = initial
    reference = Path(voice["reference_audio"])
    if not reference.is_file() or not 0 < reference.stat().st_size <= 32 * 1024 * 1024:
        raise ValueError("Reference audio must be a local file of at most 32 MiB")
    before = stamp(reference)
    content = reference.read_bytes()
    if before != stamp(reference):
        raise ValueError("Reference audio changed while being snapshotted")
    ref_hash = hashlib.sha256(content).hexdigest()
    directory = root / ".local/vtuber-reference"
    directory.mkdir(parents=True, exist_ok=True)
    snapshot = directory / (ref_hash + reference.suffix.lower())
    try:
        with snapshot.open("xb") as stream:
            stream.write(content)
    except FileExistsError:
        if file_hash(snapshot) != ref_hash:
            raise ValueError("Existing reference snapshot is corrupt")
    records["reference_audio"] = ref_hash
    source_files = [root / name for name in ("vtuber_profile.py", "vtuber_service.py",
                                            "api_vtuber.py", "vtuber_engine_support.py")]
    source_files.extend((root / "GPT_SoVITS/TTS_infer_pack").glob("*.py"))
    source_files.extend((root / "GPT_SoVITS/text").rglob("*.py"))
    records["frontend_sources"] = {p.relative_to(root).as_posix(): file_hash(p) for p in sorted(source_files)}
    return {"assets": records, "stamps": stamps, "voice": voice, "reference_snapshot": str(snapshot),
            "configured_version": version,
            "configured_paths": {key: str((root / getattr(config, key)).resolve()) for key in ASSET_KEYS}}


def preload_reference(pipeline, bundle, terminators):
    """Move reference and prompt feature extraction to startup, without synthesis."""
    voice = bundle["voice"]
    snapshot = bundle["reference_snapshot"]
    expected = bundle["assets"]["reference_audio"]
    if file_hash(snapshot) != expected:
        raise ValueError("Reference snapshot changed before loading")
    pipeline.set_ref_audio(snapshot)
    prompt = voice["prompt_text"].strip("\n")
    if prompt[-1] not in terminators:
        prompt += "." if voice["prompt_lang"] == "en" else "。"
    phones, features, normalized = pipeline.text_preprocessor.segment_and_extract_feature_for_text(
        prompt, voice["prompt_lang"], pipeline.configs.version)
    if phones is None or not len(phones):
        raise ValueError("Reference prompt produced no phonemes")
    pipeline.prompt_cache.update(prompt_text=prompt, prompt_lang=voice["prompt_lang"],
                                 phones=phones, bert_features=features, norm_text=normalized)
    if voice["options"]["gpt_streaming"]:
        pipeline._streaming_ref_model = id(pipeline.vits_model)
    if file_hash(snapshot) != expected:
        raise ValueError("Reference snapshot changed while loading")


def validate_loaded_binding(bundle, pipeline, root):
    """Reject model fallback/version drift before invoking the language frontend."""
    root = Path(root).resolve()
    config = pipeline.configs
    for key, expected in bundle["configured_paths"].items():
        if str((root / getattr(config, key)).resolve()) != expected:
            raise ValueError("The engine silently changed a configured model asset")
    if str(config.version) != bundle["configured_version"]:
        raise ValueError("Configured model version differs from the loaded weights")


def finish_profile(bundle, pipeline, root):
    """Bind the public fingerprint to the actually selected loaded configuration."""
    root = Path(root).resolve()
    config = pipeline.configs
    validate_loaded_binding(bundle, pipeline, root)
    for file, before in bundle["stamps"].items():
        if stamp(file) != before:
            raise ValueError("Model assets changed during startup; restart with stable files")
    voice = bundle["voice"]
    if voice["text_lang"] not in config.languages or voice["prompt_lang"] not in config.languages:
        raise ValueError("Configured voice language is unsupported by the loaded model")
    options = voice["options"]
    if options["gpt_streaming"] and str(config.version) not in {"v2ProPlus", "v5turbo", "v5dev"}:
        raise ValueError("The loaded model does not support semantic streaming")
    if uses_g2pw(str(config.version), voice):
        # The frontend may create these derived caches during prompt preloading.
        for name in ("g2pw/polyphonic.pickle", "g2pw/polyphonic.md5"):
            path = root / "GPT_SoVITS/text" / name
            if name not in bundle["assets"] and path.is_file():
                bundle["assets"][name] = {"file": file_hash(path)}
    metadata = {"service_version": SERVICE_VERSION, "upstream_revision": UPSTREAM_REVISION,
                "assets": bundle["assets"], "model_version": str(config.version),
                "device": str(config.device), "is_half": bool(config.is_half),
                "prompt_text": voice["prompt_text"], "prompt_lang": voice["prompt_lang"],
                "text_lang": voice["text_lang"], "options": options}
    mode = options["streaming_mode"]
    params = {**options, "media_type": "raw", "streaming_mode": mode in (2, 3),
              "return_fragment": mode == 1, "fixed_length_chunk": mode == 3,
              "ref_audio_path": bundle["reference_snapshot"], "aux_ref_audio_paths": [],
              "prompt_text": voice["prompt_text"], "prompt_lang": voice["prompt_lang"],
              "text_lang": voice["text_lang"], "_strict_errors": True}
    return {"fingerprint": digest(metadata), "inference_params": params,
            "streaming_options": options, "model_version": str(config.version)}
