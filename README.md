# OpenCreator Personal Burmese Dubbing MVP

This is a Kaggle-friendly personal-use MVP inspired by OpenCreator's staged creator workflow, progress events, retryable stages, and versioned local artifacts. It is a standalone Python/Gradio implementation rather than a direct Electron port of the OpenCreator repository.

## What it does

- Accepts a local video upload or a public video URL.
- Extracts audio and transcribes Chinese speech with `faster-whisper`.
- Translates each segment to Burmese through Gemini or Groq when configured.
- Generates Burmese speech through Microsoft Edge voices `my-MM-ThihaNeural` (သီဟ) and `my-MM-NilarNeural` (နီလာ).
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

The copy/paste-ready GitHub clone, GPU check, FFmpeg install, Python dependency install, and launch cells are in [`cell.txt`](cell.txt). Replace its `REPO_URL` placeholder with the public GitHub repository URL.

Copy the files into `/kaggle/working/opencreator-kaggle-dubbing/`, then run:

```python
%cd /kaggle/working/opencreator-kaggle-dubbing
!python app.py
```

The app uses `share=True`, so Gradio prints a temporary public URL. It only works while the Kaggle session remains alive. Do not share the link for personal-use data.

After uploading a video, click **Preview Box**. The app creates a short looping video preview, not a static image. Drag the red rectangle over the original subtitle and resize it from the bottom-right handle. There are no visible X/Y/Width/Height controls; the editor keeps the coordinates as hidden state. Open **Settings** to paste a Gemini or Groq key for this session; keys are kept only in the running Python process. Use the color pickers to select subtitle and outline colors.

## Translation API keys

API keys are intentionally not required in the notebook cells. Open the running app's **Open Settings** section and paste a Gemini or Groq key. The key is stored only in the running Python process for that Kaggle session and is not written to the repository. The app will use the selected provider (`auto`, `gemini`, or `groq`) for translation.

```python
# API keys are entered in the UI, not in this file.
```

`auto` tries Gemini first, then Groq. If neither key is present, the pipeline stops with a clear error. A local translation adapter is intentionally left as the next extension because Burmese quality varies significantly by model.

## Important limitation in this first code drop

The pipeline has the voice-routing hook for male/female segments and speaker mappings, but the first UI does not yet run automatic gender diarization. Transcription segments default to the male voice unless a segment's `gender` or a speaker map is populated. The next implementation step should add a diarization/gender review panel and allow editing the generated `segments.json` before TTS.

The blur box is draggable in the preview and currently applies one fixed rectangle to the whole video. A later revision can add per-scene/keyframe boxes for videos where subtitle position changes.

Edge TTS is an online service, not an offline local model. Internet access is required and voice availability/rate limits can change.

## Files

- `pipeline.py` — staged workflow, state artifacts, FFmpeg rendering, TTS, translation adapters.
- `app.py` — temporary Gradio UI with upload/link input, blur-box settings, font settings, preview, output download.
- `kaggle_setup.py` — optional Python bootstrap script for Kaggle.
- `cell.txt` — copy/paste-ready Kaggle notebook cells for extraction, package installation, secrets, and launch.
- `requirements.txt` — Python dependencies.
