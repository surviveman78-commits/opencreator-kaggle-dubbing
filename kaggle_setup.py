from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parent

subprocess.run(["bash", "-lc", "apt-get update -qq && apt-get install -y -qq ffmpeg fonts-noto-core fonts-noto-extra"], check=True)
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "-r", str(ROOT / "requirements.txt")], check=True)

print("Setup complete. Optional translation environment variables:")
print("  GEMINI_API_KEY=...  (and optionally GEMINI_MODEL=gemini-2.5-flash)")
print("  GROQ_API_KEY=...    (and optionally GROQ_TRANSLATION_MODEL=llama-3.1-8b-instant)")
print("Run: python app.py")
