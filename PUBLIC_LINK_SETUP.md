# Kaggle Public Link: Gradio Share

Run `app.py`, not `custom_ui.py`:

```python
%cd /kaggle/working/opencreator-kaggle-dubbing
!python app.py
```

The launcher uses `share=True` and prints a temporary public URL similar to:

```text
https://random-name.gradio.live
```

Open the Gradio URL while the Kaggle process is running. If the browser is refreshed, the latest completed input/output artifacts are restored from `opencreator_runtime/job-*/`.

To verify the correct version:

```python
from pathlib import Path
p = Path('/kaggle/working/opencreator-kaggle-dubbing/app.py').read_text()
assert 'launch(share=True' in p
assert 'restore_last_result' in p
print('GRADIO SHARE V4 OK')
```

Do not use `127.0.0.1:7860`, `172.x.x.x:7860`, or the old Cloudflare Quick Tunnel launcher for this Kaggle workflow.
