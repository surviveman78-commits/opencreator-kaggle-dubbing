# OpenCreator Personal Burmese Dubbing MVP

This is a Kaggle-friendly personal-use MVP inspired by OpenCreator's staged creator workflow, progress events, retryable stages, and versioned local artifacts. It is a standalone Python/Gradio implementation rather than a direct Electron port of the OpenCreator repository.

## What it does

- Accepts a local video upload or a public video URL.
- Extracts audio and transcribes Chinese speech with `faster-whisper`.
- Extracts a transcript first, then translates numbered transcript batches with Gemini instead of making one API call per segment.
- Generates Burmese speech locally with F5 Myanmar TTS on the Kaggle GPU; optional male/female reference recordings can guide the two voices. Microsoft Edge voices remain available only as a fallback code path.
- Aligns each generated segment to the original timestamps.
- Renders Burmese ASS subtitles.
- Applies a draggable/resizable user-defined blur box over burned-in original subtitles.
- Supports `.ttf`/`.otf` font upload, color pickers, font size, outline, and audio settings.
- Provides a Settings accordion with session-only Gemini/Groq API key fields.
- Mixes Burmese audio with a reduced original track and exports MP4.
- Saves job artifacts under `opencreator_runtime/job-*/` for review and reruns.

## Kaggle setup

In Kaggle Notebook Settings, select a GPU accelerator such as T4/P100 and enable Internet. Kaggle already provides the NVIDIA driver/CUDA runtime for the selected accelerator; the project does not install a separate CUDA toolkit. `pipeline.py` checks `torch.cuda.is_available()` and then `nvidia-smi` as a fallback. If a GPU is available, faster-whisper uses CUDA/float16; otherwise it uses CPU/int8.

In a Kaggle notebook cell:

The copy/paste-ready GitHub clone, GPU check, FFmpeg install, Python dependency install, and launch cells are in [`cell.txt`](cell.txt). The launcher runs `custom_ui.py`, which downloads/loads Whisper and F5 TTS before the web page opens. It also starts a Cloudflare Quick Tunnel and prints the external `https://*.trycloudflare.com` URL. Do not use the Flask internal `127.0.0.1:7860` or `172.x.x.x:7860` addresses, and do not launch the old `app.py` entrypoint for this custom UI.

This package is **custom-ui-public-link-v2**. Confirm the correct GitHub copy with `VERSION.txt` and `PUBLIC_LINK_SETUP.md`, or search `custom_ui.py` for `start_public_tunnel`, `trycloudflare.com`, and `PUBLIC LINK (open this URL)`.

The main interface is a custom Flask HTML/CSS/JavaScript UI, not the Gradio layout. It provides the video preview, Liquid Glass drag/resize blur box, session Gemini key field, male/female reference audio fields, progress bar, and final MP4 download.

Subtitle font behavior is explicit: when no font is uploaded, the renderer and preview use **Noto Sans Myanmar**. When a `.ttf` or `.otf` file is uploaded, it is loaded into the preview and becomes selectable in the font dropdown; the selected font is sent to the final render.

Copy the files into `/kaggle/working/opencreator-kaggle-dubbing/`, then run:

```python
%cd /kaggle/working/opencreator-kaggle-dubbing
!python app.py
```

The app uses `share=True`, so Gradio prints a temporary public URL. It only works while the Kaggle session remains alive. Do not share the link for personal-use data.

After uploading a video, click **Preview Box**. The app creates a short looping video preview, not a static image. Drag the red rectangle over the original subtitle and resize it from the bottom-right handle. There are no visible X/Y/Width/Height controls; the editor keeps the coordinates as hidden state. Open **Settings** to paste a Gemini or Groq key for this session; keys are kept only in the running Python process. Use the color pickers to select subtitle and outline colors.

## Translation architecture and API keys

The default `gemini / batch transcript` mode sends numbered transcript batches to Gemini, which is much faster than running a local 7B model once for every subtitle segment. The UI also exposes `groq`, `auto`, and `local` provider choices for experimentation.

API keys are intentionally not required in the notebook cells. Open the running app's **Open Settings** section, paste the Gemini key, and click **Save keys**. The key is stored only in the running Python process for that Kaggle session and is not written to the repository.

The audio path is timestamp-preserving: the source video is converted to mono 16 kHz audio, Whisper creates timed Chinese segments, Gemini returns one Burmese line per segment, local F5 TTS creates voice clips, and CUDA/FFmpeg delays and mixes each clip at its original segment start time before rendering the final MP4. The renderer does not downscale the original video; it keeps its original resolution and aspect ratio.

```python
# API keys are entered in the UI, not in this file.
```

`auto` tries Gemini first, then Groq. If neither key is present, the pipeline stops with a clear error. A local translation adapter is intentionally left as the next extension because Burmese quality varies significantly by model.

## Important limitation in this first code drop

The pipeline performs lightweight pitch-based routing for the requested two-voice workflow: lower-pitched segments use Microsoft `my-MM-ThihaNeural`, and higher-pitched segments use `my-MM-NilarNeural`. This is a heuristic rather than full speaker diarization, so ambiguous or overlapping speech may still need review.

The blur box is draggable in the preview and currently applies one fixed rectangle to the whole video. A later revision can add per-scene/keyframe boxes for videos where subtitle position changes.

Edge TTS is an online service, not an offline local model. Internet access is required and voice availability/rate limits can change.

## Files

- `pipeline.py` — staged workflow, state artifacts, FFmpeg rendering, TTS, translation adapters.
- `app.py` — temporary Gradio UI with upload/link input, blur-box settings, font settings, preview, output download.
- `kaggle_setup.py` — optional Python bootstrap script for Kaggle.
- `cell.txt` — copy/paste-ready Kaggle notebook cells for extraction, package installation, secrets, and launch.
- `requirements.txt` — Python dependencies.
