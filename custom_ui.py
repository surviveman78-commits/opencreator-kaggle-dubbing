from __future__ import annotations

import json
import os
import threading
import uuid
import re
import stat
import subprocess
import time
import urllib.request
from pathlib import Path

from flask import Flask, jsonify, render_template_string, request, send_from_directory

from pipeline import DubbingConfig, DubbingPipeline, PipelineError

ROOT = Path(os.getenv('OPENCREATOR_RUNTIME', './opencreator_runtime'))
UPLOADS = ROOT / 'uploads'
UPLOADS.mkdir(parents=True, exist_ok=True)
PIPE = DubbingPipeline(ROOT)
JOBS: dict[str, dict] = {}
LOCK = threading.Lock()

HTML = r'''<!doctype html><html><head><meta name="viewport" content="width=device-width,initial-scale=1"><title>OpenCreator Burmese Dubbing</title><style>
:root{color-scheme:dark;--bg:#0b0d12;--panel:#151922;--line:#2b3444;--accent:#68b7ff}*{box-sizing:border-box}body{margin:0;background:radial-gradient(circle at 15% 0,#18263c,#0b0d12 46%);font-family:Inter,system-ui,sans-serif;color:#f7f9fc}.wrap{max-width:1000px;margin:auto;padding:28px 18px}.hero{padding:28px;border:1px solid #34435a;border-radius:28px;background:linear-gradient(135deg,#182334cc,#11151dcc);box-shadow:0 18px 60px #0007}.hero h1{margin:0 0 8px;font-size:clamp(28px,5vw,48px)}.muted{color:#aab6c8}.grid{display:grid;grid-template-columns:1fr 1fr;gap:16px;margin-top:18px}.card{background:#121720e8;border:1px solid var(--line);border-radius:20px;padding:18px}.wide{grid-column:1/-1}label{display:block;color:#b8c4d5;font-size:13px;margin:12px 0 7px}input[type=file],input[type=password],input[type=text]{width:100%;padding:12px;border-radius:12px;border:1px solid #384457;background:#0d1118;color:#fff}button{border:0;border-radius:14px;padding:13px 18px;background:linear-gradient(135deg,#62baff,#5d7dff);color:#07101d;font-weight:800;cursor:pointer;margin-top:14px}button:disabled{opacity:.5;cursor:not-allowed}.preview{position:relative;max-width:720px;margin:14px auto;overflow:hidden;border-radius:20px;background:#000;isolation:isolate}.preview video{display:block;width:100%;max-height:70vh;object-fit:contain;position:relative;z-index:0}.subtitlePreview{position:absolute;z-index:7;left:10%;top:77%;width:80%;text-align:center;color:#fff;font-size:42px;font-family:'Noto Sans Myanmar',sans-serif;font-weight:600;line-height:1.18;pointer-events:none;text-shadow:3px 3px 0 #000,-3px -3px 0 #000,3px -3px 0 #000,-3px 3px 0 #000}.glass{position:absolute;z-index:5;left:8%;top:76%;width:84%;height:14%;min-height:55px;border:2px solid #fff;outline:2px solid #6aaeff;background:linear-gradient(135deg,#ffffff55,#78c5ff33,#ffffff12);backdrop-filter:blur(14px) saturate(190%);box-shadow:inset 0 1px #fff9,0 0 24px #4ca7ffb8;touch-action:none;cursor:move}.glass span{position:absolute;left:10px;top:10px;padding:6px 11px;border-radius:999px;background:#18315dcc;border:1px solid #ffffffaa;font-size:12px;font-weight:800;pointer-events:none}.handle{position:absolute;right:-13px;bottom:-13px;width:34px;height:34px;border-radius:50%;background:linear-gradient(135deg,#fff,#5ba8ff);border:3px solid #fff;box-shadow:0 2px 12px #125aaacc;cursor:nwse-resize}.settings{margin-top:14px;padding:14px;border:1px solid #3b4d68;border-radius:16px;background:#101722}.bar{height:10px;background:#202838;border-radius:99px;overflow:hidden;margin-top:14px}.fill{height:100%;width:0;background:linear-gradient(90deg,#5fd2ff,#7d75ff);transition:width .25s}.status{white-space:pre-wrap;color:#b9c7da;min-height:28px;margin-top:10px}.download{display:inline-block;margin-top:14px;color:#08111c;background:#7dd2ff;padding:13px 18px;border-radius:14px;font-weight:800;text-decoration:none}.hidden{display:none}@media(max-width:720px){.grid{grid-template-columns:1fr}.wide{grid-column:auto}.wrap{padding:14px}.hero{padding:20px}}
	</style></head><body><main class="wrap"><section class="hero"><h1>OpenCreator</h1><div class="muted">Chinese video → Burmese dubbing · Local GPU TTS · Gemini batch translation</div><div class="grid"><div class="card wide"><label>Video</label><input id="video" type="file" accept="video/*"><button id="upload">Load video</button><button id="previewBtn" disabled>Preview Box</button><div id="previewHost"><div class="muted">Upload a video to open the preview editor.</div></div></div><div class="card"><label>Blur strength</label><input id="blur" type="range" min="1" max="50" value="18"><button id="settingsBtn">⚙ Open Settings</button><div id="settings" class="settings hidden"><label>Gemini API key (session only)</label><input id="key" type="password" placeholder="Paste Gemini API key"><button id="saveKey">Save Gemini key</button><div id="keyStatus" class="muted">Not saved</div><label>Custom font (.ttf/.otf)</label><input id="font" type="file" accept=".ttf,.otf"><label>Font to use</label><select id="fontSelect"><option value="Noto Sans Myanmar">Noto Sans Myanmar (default)</option></select><label>Font size</label><input id="fontSize" type="range" min="18" max="110" value="42"><label>Subtitle color</label><input id="fontColor" type="color" value="#ffffff"><label>Outline color</label><input id="outlineColor" type="color" value="#000000"><label>Outline width</label><input id="outlineWidth" type="range" min="0" max="12" value="3"><div class="muted" style="margin-top:8px">Male/female voice references are extracted automatically from the original video audio.</div></div><button id="run" disabled>Start dubbing</button></div><div class="card"><div class="muted">Models are downloaded before this UI opens. Original resolution/aspect ratio is preserved.</div><div class="bar"><div id="fill" class="fill"></div></div><div id="status" class="status">Ready</div><div id="output" class="hidden"><video id="outputVideo" controls playsinline style="width:100%;border-radius:14px;margin-top:14px"></video><a id="download" class="download">Download dubbed MP4</a></div></div></div></section></main><script>
let fileName='', previewMeta={w:1280,h:720}, coords={x:0,y:0,w:0,h:0}; const $=id=>document.getElementById(id); function setStatus(t,p){$('status').textContent=t;$('fill').style.width=p+'%'} function updateSubtitlePreview(){let e=$('subtitlePreview');if(!e)return;e.style.fontSize=$('fontSize').value+'px';e.style.color=$('fontColor').value;e.style.fontFamily="'"+$('fontSelect').value+"', sans-serif";let o=$('outlineColor').value;e.style.textShadow='3px 3px 0 '+o+',-3px -3px 0 '+o+',3px -3px 0 '+o+',-3px 3px 0 '+o}['fontSize','fontColor','outlineColor','fontSelect'].forEach(id=>document.addEventListener('input',e=>{if(e.target.id===id)updateSubtitlePreview()}));
	let loaded=false;$('settingsBtn').onclick=()=>{$('settings').classList.toggle('hidden')};$('upload').onclick=async()=>{let f=$('video').files[0];if(!f)return setStatus('Choose a video first',0);let fd=new FormData();fd.append('video',f);setStatus('Uploading…',3);let r=await fetch('/api/upload',{method:'POST',body:fd});let d=await r.json();if(!r.ok)return setStatus(d.error,0);fileName=d.name;previewMeta={w:d.coordsW,h:d.coordsH};coords=d.coords;$('previewHost').innerHTML=`<div class="preview"><video id="pv" src="/media/${d.name}" controls playsinline></video><div id="glass" class="glass"><span>LIQUID GLASS · DRAG / RESIZE</span><i class="handle"></i></div><div id="subtitlePreview" class="subtitlePreview">မြန်မာစာ နမူနာစာတန်း</div></div>`;bind();$('previewBtn').disabled=false;$('run').disabled=false;loaded=true;setStatus('Video ready. Preview Box is active; set the fixed blur area.',5)};$('previewBtn').onclick=()=>{if(loaded){$('previewHost').scrollIntoView({behavior:'smooth',block:'center'});setStatus('Drag the Liquid Glass box and resize its corner handle.',6)}};
function bind(){let b=$('glass'),v=$('pv'),mode=null,sx=0,sy=0,ox=0,oy=0,ow=0,oh=0;function place(){let sx=v.clientWidth/previewMeta.w,sy=v.clientHeight/previewMeta.h;b.style.left=coords.x*sx+'px';b.style.top=coords.y*sy+'px';b.style.width=coords.w*sx+'px';b.style.height=coords.h*sy+'px'}function sync(){let sx=previewMeta.w/v.clientWidth,sy=previewMeta.h/v.clientHeight;coords={x:parseFloat(b.style.left)*sx,y:parseFloat(b.style.top)*sy,w:parseFloat(b.style.width)*sx,h:parseFloat(b.style.height)*sy}}b.onpointerdown=e=>{mode=e.target.classList.contains('handle')?'resize':'move';sx=e.clientX;sy=e.clientY;ox=b.offsetLeft;oy=b.offsetTop;ow=b.offsetWidth;oh=b.offsetHeight;b.setPointerCapture(e.pointerId);e.preventDefault()};b.onpointermove=e=>{if(!mode)return;let dx=e.clientX-sx,dy=e.clientY-sy;if(mode==='move'){b.style.left=Math.max(0,Math.min(ox+dx,v.clientWidth-ow))+'px';b.style.top=Math.max(0,Math.min(oy+dy,v.clientHeight-oh))+'px'}else{b.style.width=Math.max(40,Math.min(v.clientWidth-ox,ow+dx))+'px';b.style.height=Math.max(40,Math.min(v.clientHeight-oy,oh+dy))+'px'}};b.onpointerup=()=>{mode=null;sync()};v.onloadedmetadata=()=>{previewMeta.w=v.videoWidth;previewMeta.h=v.videoHeight;place()};}
	$('font').onchange=async()=>{let f=$('font').files[0];if(!f)return;let family='UploadedFont';try{let face=new FontFace(family,`url(${URL.createObjectURL(f)})`);await face.load();document.fonts.add(face)}catch(e){}let old=[...$('fontSelect').options].find(o=>o.value===family);if(!old){let o=document.createElement('option');o.value=family;o.textContent=f.name;$('fontSelect').appendChild(o)}$('fontSelect').value=family;updateSubtitlePreview()};let savedKey=sessionStorage.getItem('opencreator_gemini_key')||'';if(savedKey){$('key').value=savedKey;$('keyStatus').textContent='Saved for this browser session'}$('saveKey').onclick=()=>{savedKey=$('key').value.trim();sessionStorage.setItem('opencreator_gemini_key',savedKey);$('keyStatus').textContent=savedKey?'Saved for this browser session':'Not saved'};$('run').onclick=async()=>{let fd=new FormData();fd.append('name',fileName);fd.append('gemini_key',$('key').value||savedKey);fd.append('coords',JSON.stringify(coords));fd.append('blur_strength',$('blur').value);fd.append('font_name',$('fontSelect').value);fd.append('font_size',$('fontSize').value);fd.append('font_color',$('fontColor').value);fd.append('outline_color',$('outlineColor').value);fd.append('outline_width',$('outlineWidth').value);if($('font').files[0])fd.append('font',$('font').files[0]);$('previewHost').classList.add('hidden');$('previewBtn').classList.add('hidden');$('run').disabled=true;setStatus('Starting…',8);let r=await fetch('/api/run',{method:'POST',body:fd});let d=await r.json();if(!r.ok){setStatus(d.error,0);$('run').disabled=false;$('previewHost').classList.remove('hidden');return}let timer=setInterval(async()=>{let s=await (await fetch('/api/status/'+d.job)).json();setStatus(s.message,s.percent);if(s.done){clearInterval(timer);$('run').disabled=false;if(s.error)return;let a=$('download');a.href='/download/'+d.job;let ov=$('outputVideo');ov.src='/output/'+d.job;$('output').classList.remove('hidden');a.classList.remove('hidden')}},700)};
</script></body></html>'''


def _ass_color(value, fallback):
    value = (value or '').strip().lstrip('#')
    if len(value) != 6:
        return fallback
    return f'&H00{value[4:6]}{value[2:4]}{value[0:2]}'.upper()

def _run(job_id, name, key, coords, blur, font_path, font_name, font_size, font_color, outline_color, outline_width):
    try:
        os.environ['GEMINI_API_KEY'] = key
        src = UPLOADS / name
        config = DubbingConfig(output_ratio='original', blur_enabled=True, blur_x=int(coords.get('x', 0)), blur_y=int(coords.get('y', 0)), blur_w=int(coords.get('w', 0)), blur_h=int(coords.get('h', 0)), blur_strength=int(blur), translation_provider='gemini', tts_backend='local_f5', font_path=font_path or '', font_name=font_name or 'Noto Sans Myanmar', font_size=int(font_size or 42), primary_color=_ass_color(font_color, '&H00FFFFFF'), outline_color=_ass_color(outline_color, '&H00000000'), outline_width=int(outline_width or 3))
        def progress(stage, percent, message):
            with LOCK: JOBS[job_id].update(percent=int(percent), message=message)
        PIPE.progress = progress
        result = PIPE.run(src, config, whisper_model='small')
        with LOCK: JOBS[job_id].update(done=True, percent=100, message='Completed', output=result['output'])
    except Exception as exc:
        with LOCK: JOBS[job_id].update(done=True, percent=0, message=str(exc), error=str(exc))

app = Flask(__name__)

@app.get('/')
def index(): return render_template_string(HTML)

@app.post('/api/upload')
def upload():
    f=request.files.get('video')
    if not f: return jsonify(error='Video file is required'),400
    name=uuid.uuid4().hex+'.mp4'; path=UPLOADS/name; f.save(path)
    meta=PIPE.probe(path); w=meta['width'] or 1280; h=meta['height'] or 720
    return jsonify(name=name,coords={'x':int(w*.08),'y':int(h*.76),'w':int(w*.84),'h':int(h*.14)},coordsW=w,coordsH=h)

@app.post('/api/run')
def run():
    name=request.form.get('name',''); coords=json.loads(request.form.get('coords','{}')); job=uuid.uuid4().hex
    font_path=''
    if request.files.get('font'):
        font_path=str(UPLOADS/(job+'-font'+Path(request.files['font'].filename).suffix)); request.files['font'].save(font_path)
    with LOCK: JOBS[job]={'percent':1,'message':'Queued','done':False,'output':'','error':''}
    threading.Thread(target=_run,args=(job,name,request.form.get('gemini_key',''),coords,request.form.get('blur_strength','18'),font_path,request.form.get('font_name','Noto Sans Myanmar'),request.form.get('font_size','42'),request.form.get('font_color','#ffffff'),request.form.get('outline_color','#000000'),request.form.get('outline_width','3')),daemon=True).start()
    return jsonify(job=job)

@app.get('/api/status/<job>')
def status(job): return jsonify(JOBS.get(job, {'percent':0,'message':'Unknown job','done':True,'error':'Unknown job'}))

@app.get('/media/<name>')
def media(name): return send_from_directory(UPLOADS,name)

@app.get('/download/<job>')
def download(job):
    out=JOBS.get(job,{}).get('output'); return send_from_directory(Path(out).parent,Path(out).name,as_attachment=True) if out else ('Not ready',404)

@app.get('/output/<job>')
def output(job):
    out=JOBS.get(job,{}).get('output'); return send_from_directory(Path(out).parent,Path(out).name) if out else ('Not ready',404)

def start_public_tunnel(port: int):
    if os.getenv('PUBLIC_LINK', 'cloudflared').lower() in ('0', 'false', 'off', 'none'):
        print(f'Public tunnel disabled. Internal server: http://127.0.0.1:{port}')
        return None
    binary = Path(os.getenv('CLOUDFLARED_BIN', str(ROOT / 'cloudflared')))
    if not binary.exists():
        print('Downloading cloudflared tunnel client…')
        urllib.request.urlretrieve('https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64', binary)
        binary.chmod(binary.stat().st_mode | stat.S_IXUSR)
    proc = subprocess.Popen([str(binary), 'tunnel', '--no-autoupdate', '--url', f'http://127.0.0.1:{port}'], stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
    pattern = re.compile(r'https://[-a-z0-9]+\.trycloudflare\.com')
    deadline = time.time() + 45
    while time.time() < deadline:
        line = proc.stdout.readline() if proc.stdout else ''
        match = pattern.search(line)
        if match:
            print('\nPUBLIC LINK (open this URL):', match.group(0), '\n', flush=True)
            return proc
    print('Cloudflared started, but the public URL was not detected. Check the tunnel log above.', flush=True)
    return proc

if __name__=='__main__':
    if os.getenv('OPENCREATOR_SKIP_PRELOAD') == '1':
        print('Preview mode: skipping model preload.')
    else:
        print('Preloading local models before UI starts…')
        PIPE.preload_models('small')
        print('Models ready. Starting custom UI.')
    port = int(os.getenv('PORT','7860'))
    threading.Thread(target=lambda: app.run(host='0.0.0.0', port=port, debug=False, use_reloader=False), daemon=True).start()
    time.sleep(1)
    start_public_tunnel(port)
    while True:
        time.sleep(60)
