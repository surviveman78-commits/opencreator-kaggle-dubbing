# Kaggle Public Link Setup

Run the project from the cloned repository with `python custom_ui.py`. Do not use `app.py` for the custom interface.

The launcher first preloads Whisper and F5 Myanmar TTS. It then starts Flask on `0.0.0.0:7860`, downloads `cloudflared` if needed, and prints:

```text
PUBLIC LINK (open this URL): https://<random-name>.trycloudflare.com
```

Open the `https://...trycloudflare.com` URL. The addresses printed by Flask, such as `127.0.0.1:7860` or `172.x.x.x:7860`, are internal notebook addresses and are not public links.

To verify that the correct version was cloned, run:

```python
from pathlib import Path
p = Path('/kaggle/working/opencreator-kaggle-dubbing/custom_ui.py').read_text()
assert 'start_public_tunnel' in p
assert 'trycloudflare.com' in p
assert 'PUBLIC LINK (open this URL)' in p
print('PUBLIC-LINK V2 OK')
```

If these assertions fail, Kaggle is using an old clone. Delete the old folder and clone the repository again, or upload the contents of this ZIP as a new GitHub commit.
