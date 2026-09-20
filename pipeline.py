from __future__ import annotations

import asyncio
import json
import os
import re
import shutil
import subprocess
import time
import uuid
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Callable, Optional


@dataclass
class Segment:
    start: float
    end: float
    source_text: str
    translated_text: str = ""
    speaker: str = "speaker_1"
    gender: str = "unknown"
    voice: str = "my-MM-ThihaNeural"
    audio_path: str = ""


@dataclass
class DubbingConfig:
    source_language: str = "zh"
    target_language: str = "my"
    output_ratio: str = "original"
    subtitle_enabled: bool = True
    blur_enabled: bool = True
    blur_x: int = 0
    blur_y: int = 0
    blur_w: int = 0
    blur_h: int = 0
    blur_strength: int = 18
    blur_enabled_if_box_empty: bool = False
    font_path: str = ""
    font_name: str = "Noto Sans Myanmar"
    font_size: int = 42
    primary_color: str = "&H00FFFFFF"
    outline_color: str = "&H00000000"
    outline_width: int = 3
    shadow: int = 1
    margin_v: int = 12
    subtitle_alignment: int = 2
    male_voice: str = "my-MM-ThihaNeural"
    female_voice: str = "my-MM-NilarNeural"
    default_voice: str = "my-MM-ThihaNeural"
    voice_rate: str = "+0%"
    voice_pitch: str = "+0Hz"
    tts_backend: str = "local_f5"
    male_reference_audio: str = ""
    female_reference_audio: str = ""
    male_reference_text: str = ""
    female_reference_text: str = ""
    keep_original_audio: bool = True
    original_audio_volume: float = 0.15
    max_duration_seconds: int = 900
    translation_provider: str = "gemini"
    local_translation_model: str = "Qwen/Qwen2.5-7B-Instruct"
    speaker_voice_map: dict[str, str] = field(default_factory=dict)


class PipelineError(RuntimeError):
    pass


class DubbingPipeline:
    """OpenCreator-inspired staged pipeline for a temporary personal Kaggle session."""

    def __init__(self, work_root: str | Path = "./opencreator_runtime", progress: Optional[Callable] = None):
        self.root = Path(work_root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.progress = progress or (lambda stage, percent, message: None)
        self._local_tokenizer = None
        self._local_model = None
        self._local_tts = None
        self._whisper_models = {}

    def _emit(self, stage: str, percent: int, message: str) -> None:
        self.progress(stage, percent, message)

    def _run(self, args: list[str], cwd: Optional[Path] = None) -> str:
        try:
            proc = subprocess.run(args, cwd=cwd, check=True, text=True, capture_output=True)
            return proc.stdout.strip()
        except FileNotFoundError as exc:
            raise PipelineError(f"Required executable not found: {args[0]}") from exc
        except subprocess.CalledProcessError as exc:
            detail = (exc.stderr or exc.stdout or "").strip()[-2000:]
            raise PipelineError(f"Command failed: {' '.join(args[:4])}\n{detail}") from exc

    def _require(self, name: str) -> None:
        if shutil.which(name) is None:
            raise PipelineError(f"{name} is required. Install it in Kaggle before running the app.")

    def new_job(self) -> Path:
        job = self.root / f"job-{time.strftime('%Y%m%d-%H%M%S')}-{uuid.uuid4().hex[:6]}"
        for name in ("audio", "tts", "renders", "fonts", "meta"):
            (job / name).mkdir(parents=True, exist_ok=True)
        return job

    def probe(self, video_path: Path) -> dict:
        self._require("ffprobe")
        raw = self._run([
            "ffprobe", "-v", "error", "-show_entries",
            "format=duration:stream=width,height,codec_name", "-of", "json", str(video_path)
        ])
        data = json.loads(raw)
        fmt = data.get("format", {})
        streams = data.get("streams", [])
        video = next((s for s in streams if "width" in s), {})
        return {
            "duration": float(fmt.get("duration", 0) or 0),
            "width": int(video.get("width", 0) or 0),
            "height": int(video.get("height", 0) or 0),
        }

    def acquire_input(self, source: str | Path, job: Path) -> Path:
        self._emit("input", 5, "Preparing video input")
        source = str(source)
        target = job / "input.mp4"
        if Path(source).exists():
            shutil.copy2(source, target)
        elif re.match(r"^https?://", source):
            self._require("yt-dlp")
            self._run(["yt-dlp", "--no-playlist", "-f", "bv*+ba/b", "--merge-output-format", "mp4", "-o", str(target), source])
        else:
            raise PipelineError("Input must be a local video path or an http(s) URL.")
        if not target.exists() or target.stat().st_size == 0:
            raise PipelineError("Input video was not created.")
        meta = self.probe(target)
        if meta["duration"] > 0 and meta["duration"] > 900:
            raise PipelineError("This personal-use MVP limits videos to 15 minutes.")
        return target

    def extract_audio(self, video: Path, job: Path) -> Path:
        self._emit("audio", 12, "Extracting 16 kHz mono audio")
        self._require("ffmpeg")
        out = job / "audio" / "source.wav"
        self._run(["ffmpeg", "-y", "-i", str(video), "-vn", "-ac", "1", "-ar", "16000", "-c:a", "pcm_s16le", str(out)])
        return out

    def cuda_available(self) -> bool:
        try:
            import torch
            return bool(torch.cuda.is_available())
        except Exception:
            try:
                result = subprocess.run(["nvidia-smi", "-L"], capture_output=True, text=True)
                return result.returncode == 0 and bool(result.stdout.strip())
            except OSError:
                return False

    def nvenc_available(self) -> bool:
        try:
            result = subprocess.run(["ffmpeg", "-hide_banner", "-encoders"], capture_output=True, text=True)
            return result.returncode == 0 and "h264_nvenc" in result.stdout
        except OSError:
            return False

    def transcribe(self, audio: Path, job: Path, model_name: str = "small") -> list[Segment]:
        self._emit("transcribing", 22, f"Transcribing Chinese audio with faster-whisper ({model_name})")
        try:
            from faster_whisper import WhisperModel
        except ImportError as exc:
            raise PipelineError("faster-whisper is not installed. Run the Kaggle setup cell first.") from exc
        device = "cuda" if self.cuda_available() else "cpu"
        compute = "float16" if device == "cuda" else "int8"
        self._emit("transcribing", 23, f"Whisper runtime: {device} / {compute}")
        cache_key = f"{model_name}:{device}:{compute}"
        model = self._whisper_models.get(cache_key)
        if model is None:
            model = WhisperModel(model_name, device=device, compute_type=compute)
            self._whisper_models[cache_key] = model
        pieces, _info = model.transcribe(str(audio), language="zh", vad_filter=True, word_timestamps=False)
        segments = [Segment(float(s.start), float(s.end), s.text.strip()) for s in pieces if s.text.strip()]
        if not segments:
            raise PipelineError("No speech was detected in the video.")
        self.assign_voice_by_pitch(audio, segments)
        self.save_segments(job, segments)
        return segments

    def assign_voice_by_pitch(self, audio: Path, segments: list[Segment]) -> None:
        """Fast heuristic routing: lower pitch -> Thiha, higher pitch -> Nilar.

        This is intentionally lightweight and does not claim speaker diarization; it is
        useful for the requested two-voice workflow and falls back to Thiha when pitch
        cannot be estimated.
        """
        try:
            import numpy as np
            import soundfile as sf
            samples, sample_rate = sf.read(str(audio), dtype="float32")
            if samples.ndim > 1:
                samples = samples.mean(axis=1)
            for segment in segments:
                start = max(0, int(segment.start * sample_rate))
                end = min(len(samples), int(segment.end * sample_rate))
                clip = samples[start:end]
                if len(clip) < sample_rate // 5:
                    continue
                clip = clip[: min(len(clip), sample_rate * 2)]
                clip = clip - float(np.mean(clip))
                if float(np.max(np.abs(clip))) < 0.01:
                    continue
                frame = clip[: min(len(clip), 4096)]
                min_lag = max(1, int(sample_rate / 350))
                max_lag = min(len(frame) - 1, int(sample_rate / 70))
                if max_lag <= min_lag:
                    continue
                corr = np.correlate(frame, frame, mode="full")[len(frame) - 1:]
                lag = min_lag + int(np.argmax(corr[min_lag:max_lag]))
                pitch = sample_rate / max(1, lag)
                segment.gender = "female" if pitch >= 165 else "male"
                segment.voice = "my-MM-NilarNeural" if segment.gender == "female" else "my-MM-ThihaNeural"
        except Exception:
            return

    def preload_models(self, whisper_model: str = "small") -> None:
        """Download/load local models before the UI starts."""
        try:
            from faster_whisper import WhisperModel
        except ImportError as exc:
            raise PipelineError("faster-whisper is not installed.") from exc
        device = "cuda" if self.cuda_available() else "cpu"
        compute = "float16" if device == "cuda" else "int8"
        self._emit("preload", 20, f"Loading Whisper {whisper_model} ({device}/{compute})")
        self._whisper_models[f"{whisper_model}:{device}:{compute}"] = WhisperModel(whisper_model, device=device, compute_type=compute)
        self._emit("preload", 65, "Loading Burmese F5 TTS model")
        self._load_local_tts()
        self._emit("preload", 100, "Models ready; UI can start")

    def _load_local_translator(self, config: DubbingConfig):
        if self._local_model is not None and self._local_tokenizer is not None:
            return self._local_tokenizer, self._local_model
        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
        except ImportError as exc:
            raise PipelineError("Local LLM packages are missing. Install transformers, accelerate, and bitsandbytes from requirements.txt.") from exc
        model_id = config.local_translation_model or os.getenv("LOCAL_TRANSLATION_MODEL", "Qwen/Qwen2.5-7B-Instruct")
        self._emit("translating", 38, f"Loading local translation LLM: {model_id}")
        tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
        kwargs = {"device_map": "auto", "trust_remote_code": True}
        if self.cuda_available():
            kwargs["quantization_config"] = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16, bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True)
        else:
            kwargs["torch_dtype"] = torch.float32
        try:
            model = AutoModelForCausalLM.from_pretrained(model_id, **kwargs)
        except Exception as exc:
            raise PipelineError(f"Could not load local LLM {model_id}. Confirm Kaggle Internet/GPU is enabled and the model can fit in memory: {exc}") from exc
        model.eval()
        self._local_tokenizer, self._local_model = tokenizer, model
        return tokenizer, model

    def translate_local(self, text: str, config: DubbingConfig) -> str:
        import torch
        tokenizer, model = self._load_local_translator(config)
        messages = [
            {"role": "system", "content": "You are a professional Chinese-to-Burmese subtitle translator. Return only natural spoken Burmese. Preserve names and meaning. Keep the translation concise enough to fit the original timestamp."},
            {"role": "user", "content": text},
        ]
        prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer(prompt, return_tensors="pt")
        try:
            device = next(model.parameters()).device
            inputs = {key: value.to(device) for key, value in inputs.items()}
            with torch.inference_mode():
                output = model.generate(**inputs, max_new_tokens=256, do_sample=False, temperature=0.0, pad_token_id=tokenizer.eos_token_id)
            generated = output[0][inputs["input_ids"].shape[-1]:]
            result = tokenizer.decode(generated, skip_special_tokens=True).strip()
            if not result:
                raise PipelineError("Local LLM returned an empty Burmese translation.")
            return result
        except PipelineError:
            raise
        except Exception as exc:
            raise PipelineError(f"Local LLM generation failed: {exc}") from exc

    def translate_segment(self, text: str, config: DubbingConfig) -> str:
        provider = (config.translation_provider or "auto").lower()
        local_error = None
        if provider in ("local", "auto"):
            try:
                return self.translate_local(text, config)
            except Exception as exc:
                local_error = exc
                if provider == "local":
                    raise
        if provider in ("auto", "gemini") and os.getenv("GEMINI_API_KEY"):
            try:
                from google import genai
                client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])
                prompt = (
                    "Translate this Chinese dialogue into natural spoken Burmese. "
                    "Keep it concise enough for the same speaking duration. Return only Burmese text.\n\n" + text
                )
                response = client.models.generate_content(model=os.getenv("GEMINI_MODEL", "gemini-2.5-flash"), contents=prompt)
                if response.text:
                    return response.text.strip()
            except Exception:
                if provider == "gemini":
                    raise
        if provider in ("auto", "groq") and os.getenv("GROQ_API_KEY"):
            try:
                import requests
                payload = {
                    "model": os.getenv("GROQ_TRANSLATION_MODEL", "llama-3.1-8b-instant"),
                    "messages": [{"role": "user", "content": "Translate to natural spoken Burmese; return only the translation:\n" + text}],
                    "temperature": 0.2,
                }
                result = requests.post("https://api.groq.com/openai/v1/chat/completions", headers={"Authorization": f"Bearer {os.environ['GROQ_API_KEY']}"}, json=payload, timeout=90)
                result.raise_for_status()
                return result.json()["choices"][0]["message"]["content"].strip()
            except Exception:
                if provider == "groq":
                    raise
        if local_error is not None and provider == "auto":
            raise PipelineError(f"Local LLM translation failed before cloud fallback: {local_error}") from local_error
        raise PipelineError("No translation backend available. Choose Local LLM or set a Gemini/Groq key in Settings.")

    def translate_gemini_batch(self, segments: list[Segment], job: Path, config: DubbingConfig) -> list[Segment]:
        """Translate the transcript in a small number of Gemini requests, not one request per segment."""
        if not os.getenv("GEMINI_API_KEY"):
            raise PipelineError("Gemini API key is required. Open Settings, paste the key, and click Save keys.")
        try:
            from google import genai
        except ImportError as exc:
            raise PipelineError("google-genai is not installed. Run the Kaggle requirements cell first.") from exc
        client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])
        model_name = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
        batch_size = 40
        for offset in range(0, len(segments), batch_size):
            batch = segments[offset:offset + batch_size]
            numbered = "\n".join(f"{i + 1}. {seg.source_text}" for i, seg in enumerate(batch))
            prompt = (
                "Translate each numbered Chinese subtitle line into natural spoken Burmese. "
                "Keep exactly the same number and order of lines. Return ONLY a valid JSON array "
                "of strings, with no markdown fences or explanations. Preserve names and meaning; "
                "keep each line concise for its original timestamp.\n\n" + numbered
            )
            self._emit("translating", 40 + int(10 * min(offset + len(batch), len(segments)) / len(segments)), f"Gemini translating transcript batch {offset + 1}-{offset + len(batch)} / {len(segments)}")
            try:
                response = client.models.generate_content(model=model_name, contents=prompt)
                raw = (response.text or "").strip()
                raw = re.sub(r"^```(?:json)?\s*|\s*```$", "", raw, flags=re.IGNORECASE).strip()
                translated = json.loads(raw)
                if not isinstance(translated, list) or len(translated) != len(batch):
                    raise ValueError(f"Gemini returned {len(translated) if isinstance(translated, list) else 'non-list'} lines for {len(batch)} input lines")
                for segment, text in zip(batch, translated):
                    segment.translated_text = str(text).strip()
                    if not segment.translated_text:
                        raise ValueError("Gemini returned an empty translation line")
            except Exception as exc:
                raise PipelineError(f"Gemini batch translation failed: {exc}") from exc
        self.save_segments(job, segments)
        (job / "meta" / "transcript.txt").write_text("\n".join(s.source_text for s in segments), encoding="utf-8")
        return segments

    def translate(self, segments: list[Segment], job: Path, config: DubbingConfig) -> list[Segment]:
        self._emit("translating", 38, "Translating Chinese dialogue into Burmese")
        provider = (config.translation_provider or "gemini").lower()
        if provider == "gemini":
            return self.translate_gemini_batch(segments, job, config)
        for index, segment in enumerate(segments):
            segment.translated_text = self.translate_segment(segment.source_text, config)
            self._emit("translating", 38 + int(12 * (index + 1) / len(segments)), f"Translated {index + 1}/{len(segments)} segments")
        self.save_segments(job, segments)
        return segments

    def choose_voice(self, segment: Segment, config: DubbingConfig) -> str:
        if segment.speaker in config.speaker_voice_map:
            return config.speaker_voice_map[segment.speaker]
        if segment.gender.lower() in ("female", "f"):
            return config.female_voice
        if segment.gender.lower() in ("male", "m"):
            return config.male_voice
        return config.default_voice

    async def _edge_tts_one(self, text: str, voice: str, rate: str, pitch: str, output: Path) -> None:
        import edge_tts
        communicate = edge_tts.Communicate(text, voice, rate=rate, pitch=pitch)
        await communicate.save(str(output))

    def _load_local_tts(self):
        if self._local_tts is not None:
            return self._local_tts
        try:
            from f5_myanmar_tts import MyanmarTTS
        except ImportError as exc:
            raise PipelineError("Local Burmese TTS is not installed. Install f5-myanmar-tts from requirements.txt.") from exc
        self._emit("tts", 52, "Loading local F5 Myanmar TTS on GPU")
        try:
            self._local_tts = MyanmarTTS()
        except Exception as exc:
            raise PipelineError(f"Could not load local F5 Myanmar TTS: {exc}") from exc
        return self._local_tts

    def _local_tts_one(self, tts, text: str, output: Path, reference_audio: str = "", reference_text: str = "") -> None:
        kwargs = {"text": text, "output_file": str(output)}
        if reference_audio:
            kwargs["ref_audio"] = reference_audio
            if reference_text:
                kwargs["ref_text"] = reference_text
        tts.speak(**kwargs)

    def _make_original_voice_references(self, segments: list[Segment], job: Path, config: DubbingConfig) -> None:
        """Extract short voice samples from the original video audio automatically."""
        source_audio = job / "audio" / "source.wav"
        if not source_audio.exists():
            return
        for gender, field_name in (("male", "male_reference_audio"), ("female", "female_reference_audio")):
            if getattr(config, field_name):
                continue
            candidate = next((s for s in segments if s.gender == gender and s.end - s.start >= 0.8), None)
            if candidate is None:
                continue
            out = job / "audio" / f"original-{gender}-reference.wav"
            start = max(0.0, candidate.start)
            duration = min(6.0, max(1.2, candidate.end - candidate.start))
            try:
                self._run(["ffmpeg", "-y", "-ss", str(start), "-t", str(duration), "-i", str(source_audio), "-ar", "24000", "-ac", "1", str(out)])
                setattr(config, field_name, str(out))
            except PipelineError:
                continue

    def make_tts(self, segments: list[Segment], job: Path, config: DubbingConfig) -> list[Segment]:
        self._emit("tts", 52, "Generating Burmese voice segments with Thiha/Nilar")
        if config.tts_backend == "local_f5":
            tts = self._load_local_tts()
            self._make_original_voice_references(segments, job, config)
            for index, segment in enumerate(segments):
                segment.voice = self.choose_voice(segment, config)
                if not segment.translated_text:
                    raise PipelineError(f"Segment {index + 1} has no translated text.")
                is_female = segment.voice == config.female_voice
                reference_audio = config.female_reference_audio if is_female else config.male_reference_audio
                reference_text = config.female_reference_text if is_female else config.male_reference_text
                out = job / "tts" / f"segment-{index:04d}.wav"
                self._local_tts_one(tts, segment.translated_text, out, reference_audio, reference_text)
                segment.audio_path = str(out)
                self._emit("tts", 52 + int(18 * (index + 1) / len(segments)), f"Generated local voice {index + 1}/{len(segments)}")
            self.save_segments(job, segments)
            return segments
        try:
            import edge_tts  # noqa: F401
        except ImportError as exc:
            raise PipelineError("edge-tts is not installed. Run the Kaggle setup cell first.") from exc
        for index, segment in enumerate(segments):
            segment.voice = self.choose_voice(segment, config)
            if not segment.translated_text:
                raise PipelineError(f"Segment {index + 1} has no translated text.")
            out = job / "tts" / f"segment-{index:04d}.mp3"
            asyncio.run(self._edge_tts_one(segment.translated_text, segment.voice, config.voice_rate, config.voice_pitch, out))
            segment.audio_path = str(out)
            self._emit("tts", 52 + int(18 * (index + 1) / len(segments)), f"Generated voice {index + 1}/{len(segments)} ({segment.voice})")
        self.save_segments(job, segments)
        return segments

    def save_segments(self, job: Path, segments: list[Segment]) -> None:
        (job / "meta" / "segments.json").write_text(json.dumps([asdict(s) for s in segments], ensure_ascii=False, indent=2), encoding="utf-8")

    def load_segments(self, job: Path) -> list[Segment]:
        data = json.loads((job / "meta" / "segments.json").read_text(encoding="utf-8"))
        return [Segment(**item) for item in data]

    def write_ass(self, segments: list[Segment], job: Path, config: DubbingConfig) -> Path:
        ass = job / "meta" / "burmese.ass"
        font_name = config.font_name.replace(",", "")
        header = f"""[Script Info]\nScriptType: v4.00+\nPlayResX: 1080\nPlayResY: 1920\nScaledBorderAndShadow: yes\n\n[V4+ Styles]\nFormat: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding\nStyle: Default,{font_name},{config.font_size},{config.primary_color},&H000000FF,{config.outline_color},&H80000000,0,0,0,0,100,100,0,0,1,{config.outline_width},{config.shadow},{config.subtitle_alignment},30,30,{config.margin_v},1\n\n[Events]\nFormat: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text\n"""
        def stamp(seconds: float) -> str:
            h = int(seconds // 3600)
            m = int((seconds % 3600) // 60)
            s = seconds % 60
            return f"{h}:{m:02d}:{s:05.2f}"
        lines = [header]
        for segment in segments:
            text = segment.translated_text.replace("{", "\\{").replace("}", "\\}").replace("\n", "\\N")
            lines.append(f"Dialogue: 0,{stamp(segment.start)},{stamp(segment.end)},Default,,0,0,0,,{text}\n")
        ass.write_text("".join(lines), encoding="utf-8")
        return ass

    def build_dubbed_audio(self, video: Path, segments: list[Segment], job: Path, config: DubbingConfig) -> Path:
        self._emit("mixing", 78, "Aligning Burmese voice segments and mixing audio")
        self._require("ffmpeg")
        dubbed = job / "renders" / "dubbed-audio.m4a"
        if self.cuda_available():
            try:
                import numpy as np
                import soundfile as sf
                import torch
                sample_rate = 48000
                duration = max(self.probe(video)["duration"], max((s.end for s in segments), default=0.0))
                mix = torch.zeros(int(duration * sample_rate) + sample_rate, dtype=torch.float32, device="cuda")
                self._emit("mixing", 79, f"CUDA audio assembly on {torch.cuda.get_device_name(0)}")
                for index, segment in enumerate(segments):
                    wav = job / "audio" / f"tts-{index:04d}.wav"
                    self._run(["ffmpeg", "-y", "-i", segment.audio_path, "-ar", str(sample_rate), "-ac", "1", "-c:a", "pcm_s16le", str(wav)])
                    samples, rate = sf.read(str(wav), dtype="float32")
                    if samples.ndim > 1:
                        samples = samples.mean(axis=1)
                    clip = torch.from_numpy(np.asarray(samples, dtype=np.float32)).to("cuda")
                    start = max(0, int(round(segment.start * sample_rate)))
                    end = min(mix.numel(), start + clip.numel())
                    if end > start:
                        mix[start:end] += clip[:end - start]
                    self._emit("mixing", 79 + int(8 * (index + 1) / len(segments)), f"CUDA mixed voice {index + 1}/{len(segments)}")
                mix = torch.clamp(mix, -1.0, 1.0).detach().cpu().numpy()
                mixed_wav = job / "audio" / "cuda-mixed.wav"
                sf.write(str(mixed_wav), mix, sample_rate, subtype="PCM_16")
                self._run(["ffmpeg", "-y", "-i", str(mixed_wav), "-c:a", "aac", "-b:a", "192k", str(dubbed)])
                return dubbed
            except Exception as exc:
                self._emit("mixing", 78, f"CUDA audio path unavailable; using FFmpeg fallback: {exc}")
        inputs: list[str] = []
        filters: list[str] = []
        for i, segment in enumerate(segments):
            inputs += ["-i", segment.audio_path]
            delay = max(0, int(segment.start * 1000))
            filters.append(f"[{i}:a]adelay={delay}:all=1[a{i}]")
        labels = "".join(f"[a{i}]" for i in range(len(segments)))
        filters.append(f"{labels}amix=inputs={len(segments)}:duration=longest:dropout_transition=0:normalize=0[dub]")
        self._run(["ffmpeg", "-y", *inputs, "-filter_complex", ";".join(filters), "-map", "[dub]", "-c:a", "aac", "-b:a", "192k", str(dubbed)])
        return dubbed

    def render(self, video: Path, dubbed_audio: Path, ass: Path, job: Path, config: DubbingConfig) -> Path:
        self._emit("rendering", 90, "Rendering blur box, Burmese subtitles, and final MP4")
        self._require("ffmpeg")
        output = job / "renders" / "burmese-dubbed.mp4"
        meta = self.probe(video)
        vf: list[str] = []
        if config.output_ratio == "9:16":
            vf.append("crop=ih*9/16:ih:(iw-ih*9/16)/2:0")
        elif config.output_ratio == "16:9":
            vf.append("crop=iw:iw*9/16:0:(ih-iw*9/16)/2")
        if config.blur_enabled and config.blur_w > 0 and config.blur_h > 0:
            x, y, w, h = config.blur_x, config.blur_y, config.blur_w, config.blur_h
            vf.append(f"split=2[base][blursrc];[blursrc]crop={w}:{h}:{x}:{y},boxblur={config.blur_strength}:1[blur];[base][blur]overlay={x}:{y}")
        if config.subtitle_enabled:
            font_dir = str(Path(config.font_path).parent) if config.font_path else "."
            vf.append(f"subtitles={ass}:fontsdir={font_dir}")
        video_filter = ",".join(vf) if vf else "null"
        cmd = ["ffmpeg", "-y", "-i", str(video), "-i", str(dubbed_audio)]
        if config.keep_original_audio:
            cmd += ["-filter_complex", f"[0:a]volume={config.original_audio_volume}[orig];[1:a]volume=1[dub];[orig][dub]amix=inputs=2:duration=longest:dropout_transition=2[aout]", "-map", "0:v", "-map", "[aout]"]
        else:
            cmd += ["-map", "0:v", "-map", "1:a"]
        if self.nvenc_available():
            cmd += ["-vf", video_filter, "-c:v", "h264_nvenc", "-preset", "p4", "-cq", "21", "-c:a", "aac", "-b:a", "192k", "-movflags", "+faststart", str(output)]
        else:
            cmd += ["-vf", video_filter, "-c:v", "libx264", "-preset", "veryfast", "-crf", "21", "-c:a", "aac", "-b:a", "192k", "-movflags", "+faststart", str(output)]
        self._run(cmd)
        self._emit("done", 100, f"Completed: {output.name}")
        return output

    def run(self, source: str | Path, config: DubbingConfig, whisper_model: str = "small") -> dict:
        job = self.new_job()
        meta_path = job / "meta" / "job.json"
        meta_path.write_text(json.dumps({"job": str(job), "config": asdict(config)}, ensure_ascii=False, indent=2), encoding="utf-8")
        video = self.acquire_input(source, job)
        audio = self.extract_audio(video, job)
        segments = self.transcribe(audio, job, whisper_model)
        segments = self.translate(segments, job, config)
        segments = self.make_tts(segments, job, config)
        dubbed_audio = self.build_dubbed_audio(video, segments, job, config)
        ass = self.write_ass(segments, job, config)
        output = self.render(video, dubbed_audio, ass, job, config)
        return {"job_dir": str(job), "output": str(output), "audio": str(dubbed_audio), "subtitle": str(ass), "segments": str(job / "meta" / "segments.json")}
