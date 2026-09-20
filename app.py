from __future__ import annotations

import base64
import html
import json
import os
import tempfile
from pathlib import Path

import gradio as gr

from pipeline import DubbingConfig, DubbingPipeline, PipelineError

RUNTIME = Path(os.getenv("OPENCREATOR_RUNTIME", "./opencreator_runtime"))

BLUR_EDITOR_JS = r"""
() => {
  const attach = () => {
    const editor = document.querySelector('#oc-blur-editor');
    const video = document.querySelector('#oc-blur-video');
    const box = document.querySelector('#oc-blur-box');
    if (!editor || !video || !box || box.dataset.bound === '1') return;
    box.dataset.bound = '1';
    box.style.display = 'block';
    box.style.zIndex = '50';
    box.style.pointerEvents = 'auto';
    let mode = null, sx = 0, sy = 0, ox = 0, oy = 0, ow = 0, oh = 0;
    const sourceW = Number(editor.dataset.sourceWidth || video.videoWidth || 720);
    const sourceH = Number(editor.dataset.sourceHeight || video.videoHeight || 1280);
    const placeFromSource = () => {
      const scaleX = video.clientWidth / Math.max(1, sourceW);
      const scaleY = video.clientHeight / Math.max(1, sourceH);
      box.style.left = (Number(editor.dataset.sourceX || 0) * scaleX) + 'px';
      box.style.top = (Number(editor.dataset.sourceY || sourceH * .76) * scaleY) + 'px';
      box.style.width = (Number(editor.dataset.sourceBoxW || sourceW * .84) * scaleX) + 'px';
      box.style.height = (Number(editor.dataset.sourceBoxH || sourceH * .14) * scaleY) + 'px';
    };
    const setNumber = (id, value) => {
      const root = document.querySelector(`#${id}`);
      const input = root && root.querySelector('input');
      if (!input) return;
      const setter = Object.getOwnPropertyDescriptor(HTMLInputElement.prototype, 'value')?.set;
      if (setter) setter.call(input, String(Math.round(value))); else input.value = String(Math.round(value));
      input.dispatchEvent(new Event('input', {bubbles: true}));
      input.dispatchEvent(new Event('change', {bubbles: true}));
    };
    const sync = () => {
      const sx = Number(editor.dataset.sourceWidth || video.videoWidth) / Math.max(1, video.clientWidth);
      const sy = Number(editor.dataset.sourceHeight || video.videoHeight) / Math.max(1, video.clientHeight);
      const coords = {x: parseFloat(box.style.left)*sx, y: parseFloat(box.style.top)*sy, w: parseFloat(box.style.width)*sx, h: parseFloat(box.style.height)*sy};
      const hidden = document.querySelector('#blur-coordinates textarea, #blur-coordinates input');
      if (hidden) { hidden.value = JSON.stringify(coords); hidden.dispatchEvent(new Event('input', {bubbles:true})); hidden.dispatchEvent(new Event('change', {bubbles:true})); }
    };
    const point = e => { const r = editor.getBoundingClientRect(); return {x: e.clientX-r.left, y: e.clientY-r.top}; };
    box.addEventListener('pointerdown', e => {
      mode = e.target.dataset.handle === 'resize' ? 'resize' : 'move';
      const p = point(e); sx=p.x; sy=p.y; ox=parseFloat(box.style.left); oy=parseFloat(box.style.top); ow=parseFloat(box.style.width); oh=parseFloat(box.style.height);
      box.setPointerCapture(e.pointerId); e.preventDefault();
    });
    box.addEventListener('pointermove', e => {
      if (!mode) return;
      const p=point(e), dx=p.x-sx, dy=p.y-sy;
      if (mode === 'move') {
        box.style.left = Math.max(0, Math.min(ox+dx, video.clientWidth-ow))+'px';
        box.style.top = Math.max(0, Math.min(oy+dy, video.clientHeight-oh))+'px';
      } else {
        box.style.width = Math.max(24, Math.min(video.clientWidth-ox, ow+dx))+'px';
        box.style.height = Math.max(24, Math.min(video.clientHeight-oy, oh+dy))+'px';
      }
      sync();
    });
    box.addEventListener('pointerup', () => { mode=null; sync(); });
    video.addEventListener('loadedmetadata', () => { placeFromSource(); sync(); });
    video.addEventListener('timeupdate', sync);
    if (video.videoWidth) placeFromSource();
    sync();
  };
  new MutationObserver(attach).observe(document.body, {childList:true, subtree:true});
  attach();
}
"""


def get_video_path(file_value):
    if not file_value:
        return None
    if isinstance(file_value, str):
        return file_value
    if isinstance(file_value, dict):
        return file_value.get("path") or file_value.get("name")
    return getattr(file_value, "name", None) or getattr(file_value, "path", None)


def hex_to_ass(value: str, fallback: str) -> str:
    value = (value or "").strip().lstrip("#")
    if len(value) != 6:
        return fallback
    try:
        int(value, 16)
    except ValueError:
        return fallback
    rr, gg, bb = value[0:2], value[2:4], value[4:6]
    return f"&H00{bb}{gg}{rr}".upper()


def draggable_preview_html(video_file: Path, source_w: int, source_h: int, x: int, y: int, w: int, h: int) -> str:
    raw = video_file.read_bytes()
    encoded = base64.b64encode(raw).decode("ascii")
    scale = min(1.0, 720 / max(1, source_w))
    x, y, w, h = int(x * scale), int(y * scale), int(w * scale), int(h * scale)
    return f"""
    <div id="oc-blur-editor" data-source-width="{source_w}" data-source-height="{source_h}" data-source-x="{int(x / max(1, min(1.0, 720 / max(1, source_w))))}" data-source-y="{int(y / max(1, min(1.0, 720 / max(1, source_w))))}" data-source-box-w="{int(w / max(1, min(1.0, 720 / max(1, source_w))))}" data-source-box-h="{int(h / max(1, min(1.0, 720 / max(1, source_w))))}" style="position:relative;isolation:isolate;display:block;width:100%;max-width:720px;background:#111;user-select:none;touch-action:none;overflow:hidden;border-radius:18px">
      <video id="oc-blur-video" src="data:video/mp4;base64,{encoded}" controls autoplay muted loop playsinline style="position:relative;z-index:0!important;display:block;width:100%;height:auto;pointer-events:auto"></video>
      <div id="oc-blur-box" style="position:absolute!important;z-index:99999!important;display:block!important;left:{x}px;top:{y}px;width:{w}px;height:{h}px;border:2px solid rgba(255,255,255,.92);outline:2px solid rgba(93,169,255,.75);background:linear-gradient(135deg,rgba(255,255,255,.34),rgba(112,190,255,.22) 45%,rgba(255,255,255,.10));backdrop-filter:blur(14px) saturate(190%);-webkit-backdrop-filter:blur(14px) saturate(190%);box-shadow:inset 0 1px 0 rgba(255,255,255,.8),inset 0 -1px 0 rgba(255,255,255,.25),0 0 0 1px rgba(35,120,255,.45),0 0 24px rgba(80,170,255,.75);box-sizing:border-box;cursor:move;pointer-events:auto!important;touch-action:none;transform:translateZ(100px)">
        <span data-handle="resize" style="position:absolute;right:-12px;bottom:-12px;width:34px;height:34px;background:linear-gradient(135deg,#fff,#8ed0ff 55%,#3b8dff);border:3px solid rgba(255,255,255,.95);border-radius:50%;box-shadow:0 2px 12px rgba(0,80,180,.8);cursor:nwse-resize;touch-action:none"></span>
        <span style="position:absolute;left:10px;top:10px;background:rgba(20,45,90,.48);backdrop-filter:blur(8px);-webkit-backdrop-filter:blur(8px);color:white;border:1px solid rgba(255,255,255,.65);padding:5px 10px;border-radius:999px;font-size:13px;font-weight:700;white-space:nowrap;letter-spacing:.3px;pointer-events:none">LIQUID GLASS • DRAG / RESIZE</span>
      </div>
    </div>
    """


def frame_preview(file_value, blur_coordinates, blur_strength):
    path = get_video_path(file_value)
    if not path or not Path(path).exists():
        return "<div>Upload a video first.</div>", "Upload a video, then drag the red box over the original subtitle."
    tmp = Path(tempfile.mkstemp(suffix=".mp4")[1])
    try:
        pipeline = DubbingPipeline(RUNTIME)
        pipeline._require("ffmpeg")
        probe = pipeline.probe(Path(path))
        pw, ph = probe["width"] or 720, probe["height"] or 1280
        coords = json.loads(blur_coordinates or "{}") if isinstance(blur_coordinates, str) else {}
        x = int(coords.get("x", pw * 0.08)); y = int(coords.get("y", ph * 0.76)); w = int(coords.get("w", pw * 0.84)); h = int(coords.get("h", ph * 0.14))
        pipeline._run(["ffmpeg", "-y", "-t", "12", "-i", path, "-vf", "scale=720:-2", "-an", "-c:v", "libx264", "-preset", "ultrafast", "-crf", "30", "-movflags", "+faststart", str(tmp)])
        return draggable_preview_html(tmp, pw, ph, x, y, w, h), f"Video preview ready. Drag the red box and resize from the bottom-right handle. Blur strength: {int(blur_strength)}"
    except Exception as exc:
        return f"<div>Preview error: {html.escape(str(exc))}</div>", "Preview failed"
    finally:
        tmp.unlink(missing_ok=True)


def save_api_keys(gemini_key, groq_key):
    if gemini_key:
        os.environ["GEMINI_API_KEY"] = gemini_key.strip()
    if groq_key:
        os.environ["GROQ_API_KEY"] = groq_key.strip()
    saved = []
    if os.getenv("GEMINI_API_KEY"):
        saved.append("Gemini")
    if os.getenv("GROQ_API_KEY"):
        saved.append("Groq")
    return "Session keys ready: " + (", ".join(saved) if saved else "none")


def run_dubbing(file_value, url, output_ratio, voice_mode, subtitle_enabled, blur_enabled, blur_coordinates, blur_strength, font_file, font_name, font_size, font_color, outline_color, outline_width, keep_original, original_volume, provider, whisper_model, male_reference_audio, female_reference_audio, male_reference_text, female_reference_text, progress=gr.Progress()):
    source = get_video_path(file_value) or (url or "").strip()
    if not source:
        raise gr.Error("Upload a video or paste a public video URL.")
    coords = json.loads(blur_coordinates or "{}") if isinstance(blur_coordinates, str) else {}
    font_path = get_video_path(font_file) if font_file else ""
    default_voice = "my-MM-NilarNeural" if str(voice_mode).lower().startswith("female") else "my-MM-ThihaNeural"
    config = DubbingConfig(
        output_ratio=output_ratio, subtitle_enabled=subtitle_enabled, blur_enabled=blur_enabled,
        blur_x=int(coords.get("x", 0)), blur_y=int(coords.get("y", 0)), blur_w=int(coords.get("w", 0)), blur_h=int(coords.get("h", 0)), blur_strength=int(blur_strength),
        font_path=font_path or "", font_name=font_name or "Noto Sans Myanmar", font_size=int(font_size),
        primary_color=hex_to_ass(font_color, "&H00FFFFFF"), outline_color=hex_to_ass(outline_color, "&H00000000"),
        outline_width=int(outline_width), default_voice=default_voice, male_voice="my-MM-ThihaNeural",
        female_voice="my-MM-NilarNeural", keep_original_audio=keep_original, original_audio_volume=float(original_volume),
        translation_provider=("gemini" if str(provider).lower().startswith("gemini") else ("auto" if str(provider).lower().startswith("auto") else str(provider).lower())), speaker_voice_map={},
        tts_backend="local_f5", male_reference_audio=get_video_path(male_reference_audio) or "", female_reference_audio=get_video_path(female_reference_audio) or "",
        male_reference_text=male_reference_text or "", female_reference_text=female_reference_text or "",
    )
    messages = []
    def on_progress(stage, percent, message):
        messages.append(f"[{percent:>3}%] {stage}: {message}")
        progress(percent / 100, desc=message)
    pipeline = DubbingPipeline(RUNTIME, progress=on_progress)
    try:
        result = pipeline.run(source, config, whisper_model=whisper_model)
    except PipelineError as exc:
        raise gr.Error(str(exc))
    except Exception as exc:
        raise gr.Error(f"Unexpected error: {exc}")
    return result["output"], result["audio"], result["subtitle"], result["segments"], "\n".join(messages)


def build_ui():
    css = ".oc-title{text-align:center}.oc-note{border:1px solid #d9d9e3;border-radius:12px;padding:12px}.oc-drag-help{font-size:13px;color:#555}"
    with gr.Blocks(title="OpenCreator Personal Burmese Dubbing", css=css, js=BLUR_EDITOR_JS) as demo:
        gr.Markdown("# OpenCreator Personal Burmese Dubbing", elem_classes="oc-title")
        gr.Markdown("Temporary personal-use tool: Chinese video → Burmese dubbing with သီဟ/နီလာ, draggable subtitle blur box, and custom subtitle fonts.", elem_classes="oc-note")
        with gr.Row():
            with gr.Column(scale=1):
                video = gr.File(label="1. Upload video", file_types=[".mp4", ".mov", ".mkv", ".webm"], type="filepath")
                url = gr.Textbox(label="Or paste public video URL", placeholder="https://...")
                output_ratio = gr.State("original")
                voice_mode = gr.State("auto")
                provider = gr.State("gemini")
                whisper_model = gr.State("small")
            with gr.Column(scale=1):
                blur_editor = gr.HTML("<div>Upload a video, then click Preview Box.</div>", label="2. Video preview — drag and resize the blur box")
                preview_note = gr.Markdown("Preview is a short sample only; final output keeps the full video duration. Drag the Liquid Glass box and resize it.")
                blur_strength = gr.Slider(1, 50, value=18, step=1, label="Blur strength")
                preview_btn = gr.Button("Preview Box")
        blur_coordinates = gr.Textbox(value="", visible=False, elem_id="blur-coordinates")
        with gr.Accordion("⚙ Open Settings", open=False, elem_id="settings-panel"):
            gr.Markdown("### Gemini API (this Kaggle session only)")
            with gr.Row():
                gemini_key = gr.Textbox(label="Gemini API key", type="password", placeholder="Paste Gemini key")
                groq_key = gr.State("")
                save_keys = gr.Button("Save keys")
            key_status = gr.Markdown("No cloud keys saved in this session.")
            gr.Markdown("Keys are stored only in this running Python process. They are not written to the repository.")
            gr.Markdown("### Subtitle and font")
            with gr.Row():
                subtitle_enabled = gr.Checkbox(value=True, label="Add Burmese subtitles")
                blur_enabled = gr.Checkbox(value=True, label="Blur original subtitle")
                keep_original = gr.Checkbox(value=True, label="Keep original BGM/SFX softly")
            with gr.Row():
                font_file = gr.File(label="Custom .ttf/.otf font", file_types=[".ttf", ".otf"], type="filepath")
                font_name = gr.Textbox(value="Noto Sans Myanmar", label="Font family name")
                font_size = gr.Slider(18, 110, value=42, step=1, label="Font size")
            with gr.Row():
                font_color = gr.ColorPicker(value="#FFFFFF", label="Subtitle color")
                outline_color = gr.ColorPicker(value="#000000", label="Outline color")
                outline_width = gr.Slider(0, 12, value=3, step=1, label="Outline width")
                original_volume = gr.Slider(0, 0.5, value=0.15, step=0.01, label="Original audio volume")
            gr.Markdown("### Optional local F5 voice references")
            with gr.Row():
                male_reference_audio = gr.File(label="Male reference audio", file_types=[".wav", ".mp3", ".m4a"], type="filepath")
                female_reference_audio = gr.File(label="Female reference audio", file_types=[".wav", ".mp3", ".m4a"], type="filepath")
            with gr.Row():
                male_reference_text = gr.Textbox(label="Male reference transcript", placeholder="Exact Burmese words in the male reference audio")
                female_reference_text = gr.Textbox(label="Female reference transcript", placeholder="Exact Burmese words in the female reference audio")
        run_btn = gr.Button("3. Start Burmese Dubbing", variant="primary")
        with gr.Row():
            output_video = gr.Video(label="Burmese dubbed video")
            output_audio = gr.Audio(label="Dubbed audio", visible=False)
        with gr.Row():
            output_subtitle = gr.File(label="Burmese ASS subtitle", visible=False)
            output_segments = gr.File(label="Segments JSON / review data", visible=False)
        logs = gr.Textbox(label="Workflow log", lines=12, visible=False)
        preview_btn.click(frame_preview, [video, blur_coordinates, blur_strength], [blur_editor, preview_note])
        save_keys.click(save_api_keys, [gemini_key, groq_key], [key_status])
        run_btn.click(run_dubbing, [video, url, output_ratio, voice_mode, subtitle_enabled, blur_enabled, blur_coordinates, blur_strength, font_file, font_name, font_size, font_color, outline_color, outline_width, keep_original, original_volume, provider, whisper_model, male_reference_audio, female_reference_audio, male_reference_text, female_reference_text], [output_video, output_audio, output_subtitle, output_segments, logs])
    return demo


if __name__ == "__main__":
    build_ui().launch(share=True, server_name="0.0.0.0", show_error=True)
