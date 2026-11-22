# Executor Tools

Low-level transformation and conversion utilities used by the benchmark agent.

For full parameter specs, behavior descriptions, and I/O schemas of each tool, see [`utils/resources/tools.yaml`](../../../utils/resources/tools.yaml). That file also controls which tools are available to the agent for a given modality. Adding a new entry there registers the tool in the pipeline.

To add your own tool: implement it in the appropriate `*_tools.py` file, then add a corresponding entry in `utils/resources/tools.yaml`.

---

## Image (`image_tools.py`)

| Function | Description |
|---|---|
| `apply_image_degradation` | Apply one or more degradation ops to an image (entry point) |
| `degrade_spatial` | Spatial degradations: blur, downscale, rotation, etc. |
| `degrade_photometric` | Photometric degradations: brightness, contrast, color shift |
| `degrade_compression` | Compression artifacts (JPEG quality reduction) |
| `degrade_noise` | Add pixel-level noise |
| `degrade_occlusion` | Occlude regions of the image |
| `image2text` | Generate a text description of an image via VLM |
| `image2objects` | Detect objects and return bounding boxes with category labels |
| `ocr` | Extract text from an image via OCR |
| `mask_or_crop_to_bbox` | Crop or mask an image to a bounding box region |
| `bbox_to_point` | Convert a bounding box to a representative point |

---

## Audio (`audio_tools.py`)

| Function | Description |
|---|---|
| `text2speech` | Synthesize speech from text (TTS) |
| `overlay_audio` | Overlay retained speech segments on a timeline, preserving full utterances |
| `speech2text` | Transcribe audio to text (ASR) |
| `add_environmental_noise` | Mix environmental noise into an audio clip at a given SNR |
| `add_background_music` | Schedule English vocal music as background interference; render after speech assembly |
| `merge_dialog` | Concatenate a list of dialog audio segments into one clip |

### Deferred audio generation

During benchmark generation, **`text2speech`, `add_environmental_noise` and `add_background_music` do not synthesize audio**. Instead they record their call parameters (speech script and background settings) into the sample's `_pure_tool_messages`. This keeps the pipeline fast and avoids requiring a GPU or audio library at generation time.

`overlay_audio` is also deferred: Generator writes `kind: tts` with `segments` (`id`, `label`, `text`), then optionally `kind: overlay_audio` with named `outputs` selecting segment IDs. A label only identifies a person for consistent voices. Standalone segments can be referenced directly, and composition can select/reuse subsets for several independent outputs. Audio schema fields use `str|list`; `media:audio:<segment-or-output-id>` is replaced by a runtime-assigned path. Environmental noise is a separate final step on selected assembled outputs. Legacy dialog/tracks jobs keep their old sequential behavior.

For MiniMax, use `python synthesize_audio.py --input-cache-dir <query-cache>` or `python -m utils.audio_processing.minimax_batch --eval-json <manifest>`, with `--dry-run` to inspect requests without API calls. Both default to the shared `utils/resources/minimax_voices.json`; `--config` overrides it. MiniMax credentials and voice mappings are delivery configuration, not generated script. See the [shared parameters and voice catalogs](../../utils/resources/minimax_audio.md) and [example config](../../utils/resources/minimax_config.example.json). An independent mixer is available as `python -m utils.audio_processing.overlay_batch --eval-json <manifest>`.

After the pipeline finishes, use the two batch scripts in `utils/audio_processing/` to perform the actual audio synthesis:

**Step 1: TTS** (`utils/audio_processing/xtts_batch.py`):
- Model: [XTTS v2](https://huggingface.co/coqui/XTTS-v2), multilingual, multi-speaker TTS via [Coqui TTS](https://github.com/idiap/coqui-ai-TTS). Pass `--model-path` to a local XTTS v2 checkpoint.
- Reads `_transcript` (dialog turns) and `_tts_output_path` from each record, synthesizes a merged wav and writes it to `_tts_output_path`.

**Step 2: Noise** (`utils/audio_processing/add_noise.py`):
- Dataset: [TUT Urban Acoustic Scenes 2016](https://zenodo.org/record/165995), labeled environmental noise clips (residential, city center, park, etc.). Pass `--noise-root` to the local dataset root (must contain `meta.txt`).
- Reads `_tts_output_path` (input wav) and `_add_environmental_noise_args` from each record, mixes at a target SNR, and writes the result to `_final_audio_path`.

```bash
# Step 1: TTS  (generates wav files to paths already recorded in the sample)
python utils/audio_processing/xtts_batch.py \
  --eval-json  cache/{topic_id}/evaluation_with_audio_settings.json \
  --model-path /path/to/xtts-v2

# Step 2: add noise (only for samples that have _add_environmental_noise_args)
python utils/audio_processing/add_noise.py \
  --eval-json  cache/{topic_id}/evaluation_with_audio_settings.json \
  --noise-root /path/to/TUT
```

Audio file paths (`_tts_output_path`, `_final_audio_path`) are determined at benchmark generation time by the stubs and recorded in the JSON. The batch scripts write to those exact paths.

---

## Vocal background music

The two layers follow the existing deferred audio tools:

- Agent description: `utils/resources/tools.yaml`; structured spec: `run_pure_tools.py`.
- Deferred interface: `implementations/audio_tools.py::add_background_music`. It reserves a path and returns `status: pending`, without accessing the music library or creating audio.
- Actual renderer and standalone/batch CLI: `utils/audio_processing/add_music.py`.

Enable `add_background_music` alongside `text2speech` and `overlay_audio`, then put
`background_music: {"intensity":"medium"}` on selected named outputs. Alternatively
use `background_music: {"speech_to_music_db":0}`. Only one level parameter is allowed.
The speech-to-music ratio ranges from -5 to 0 dB and defaults to 0 dB. The
low/medium/high settings correspond to 0/-2.5/-5 dB; negative values make the
song louder than active foreground speech. The measurement includes accompaniment;
it is not a separated vocal ratio.

The backend selects English lyric-bearing windows, repeats with crossfades if
necessary, preserves the foreground duration/rate/channels, and applies no ducking.
It uses one float mix and one peak attenuation for music and optional TUT noise.
It writes `<output>.music.json` with the source, seed, window, gain and source hashes;
delivery receipts also detect changed music or metadata files. Selection is seeded
per item/output; changing intensity reuses the same speech and song window.

The library defaults to `../datasets/data/JamendoLyrics` relative to the repository.
Override it with `MUSIC_ROOT` or `--music-root`. It must contain the downloaded
`metadata.jsonl` and its referenced audio. MP3 decoding uses `soundfile`/libsndfile
with MPEG support; outputs use the existing WAV/FLAC writer.

```bash
# One existing speech recording; no API calls
python -m utils.audio_processing.add_music --audio-path speech.wav --output-path speech_music.wav --intensity medium

# Independent pass after TTS/assembly
python -m utils.audio_processing.add_music --eval-json evaluation_with_audio_settings.json

# Integrated delivery; --dry-run also checks the music library without API calls
python synthesize_audio.py --input-cache-dir cache/user_queries/user_query_XX --music-root /path/to/JamendoLyrics
```

Music can coexist with `noise` when both tools are enabled. The music pass then
renders both backgrounds; the separate noise pass skips these outputs.

## Adding a Custom Tool

There are two kinds of tools in the pipeline. Choose the right one before starting:

| | LLM Tool | Non-LLM Tool |
|:---|:---|:---|
| Execution | `_run_llm_step()` in `transformation_tools_rollback.py` | `build_pure_tool_registry()` in `run_pure_tools.py` |
| Implementation | No Python code needed | Python function required |
| When to use | Free-form LLM transformation (rewrite, construct, reason) | Deterministic or non-LLM backend (TTS, search, image ops) |

> Non-LLM tools are referred to as *pure tools* in the codebase.

---

### Adding an LLM Tool

LLM steps require no Python implementation. The agent plans them automatically at grounding time using the description you provide.

**Step 1: Register it in `utils/resources/tools.yaml`**:

```yaml
- name: my_llm_tool
  category: scenario_alignment
  description: "What this LLM step does."
  modalities: [text]           # controls when the agent may select this tool
  io:
    inputs: [text]
    outputs: [text]
  params:
    - guidance: "string, instruction for the LLM"
  typical_uses:
    - "Describe when to use this step."
```

The executor will call `_run_llm_step()` with the `guidance` written by the grounding agent at sample time.

---

### Adding a Non-LLM Tool

Non-LLM tools require three steps: implement the function, register the tool, and wire up the spec.

**Step 1: Implement the function** in the appropriate `*_tools.py` file (or a new file):

```python
def my_tool(arg1: str, arg2: int = 1) -> dict:
    # ... your logic ...
    return {"result_field": ...}
```

The function must return a plain `dict`. If it produces a file (audio, image), include the output path in the dict so the pipeline can record it.

**Step 2: Register it in `utils/resources/tools.yaml`**:

Key fields to set: `name`, `modalities` (controls which task types the agent may select this tool for), `description`, `params`, and `typical_uses`. See the existing entries for `category`, `io`, and `behavior`. The unused legacy `memory` configuration is omitted; scripts and artifacts are retained by the executor and media export code.

**Step 3: Wire up the backend in `tools/executor_tools/run_pure_tools.py`**:

Add a `_build_spec_my_tool()` function that defines `param_schema` (the structured parameter spec the LLM planner sees), `return_schema`, and sets `backend=my_tool`. Then register it in `build_pure_tool_registry`:

```python
elif name == "my_tool":
    spec = _build_spec_my_tool(t)
```

---

## Note on partial implementations

This release includes implementations for the tools required by the example user_queries. 
Some tools (e.g. `speech2text`, `ocr`, `image2objects`) are provided as stubs and will be open-sourced in a future release.

---

## Web (`web_tools.py`)

| Function | Description |
|---|---|
| `web_search` | Run a web search query and return summarized results |
