# Musik Splitter v2026 FINAL - auto ffmpeg fallback
from flask import Flask, request, render_template, jsonify, send_from_directory, send_file
import os, subprocess, sys, uuid, glob, threading, re, logging, io, zipfile, shutil
import librosa
import numpy as np
logging.getLogger('werkzeug').setLevel(logging.ERROR)
os.environ.setdefault("OMP_NUM_THREADS","4")
def _ffmpeg():
    f=shutil.which("ffmpeg")
    if f: return f
    try:
        import imageio_ffmpeg
        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:
        return "ffmpeg"
FFMPEG=_ffmpeg()
print(f"[FFMPEG] using: {FFMPEG}", flush=True)
app = Flask(__name__, template_folder='.')
app.config['MAX_CONTENT_LENGTH']=100*1024*1024
os.makedirs("uploads",exist_ok=True); os.makedirs("output",exist_ok=True)
progress={}; done={}; error_msg={}; last_log={}; busy=False; busy_lock=threading.Lock()
CHORDS=['C','C#','D','D#','E','F','F#','G','G#','A','A#','B']
MAJOR=np.array([1,0,1,0,1,1,0,1,0,1,0,1],dtype=float)
MINOR=np.array([1,0,1,1,0,1,0,1,1,0,1,0],dtype=float)
def detect_bpm_chord(path):
    try:
        y,sr=librosa.load(path,sr=22050,mono=True,duration=30)
        tempo,_=librosa.beat.beat_track(y=y,sr=sr)
        bpm=float(tempo[0]) if hasattr(tempo,'__len__') else float(tempo)
        chroma=librosa.feature.chroma_cqt(y=y,sr=sr); avg=chroma.mean(axis=1)
        best=-1; bc="-"
        for i,root in enumerate(CHORDS):
            for kind,t in [('maj',MAJOR),('m',MINOR)]:
                tt=np.roll(t,i); s=float(np.dot(avg,tt)/(np.linalg.norm(avg)*np.linalg.norm(tt)+1e-6))
                if s>best: best=s; bc=f"{root} {'Mayor' if kind=='maj' else 'Minor'}"
        return round(bpm),bc
    except Exception as e: print("BPM error:",e,flush=True); return 0,"-"
def _real_error(all_out,code):
    lines=[l.strip() for l in (all_out or "").replace("\r","\n").split("\n") if l.strip()]
    bad=[l for l in lines if any(k in l for k in ["Traceback","Error","Exception","Failed","No such file","Permission","Memory","Killed","not found","cannot","no channel"])]
    bad=[l for l in bad if "model is a bag" not in l and "progress bars" not in l and "Separating track" not in l]
    if bad: return "Demucs gagal: "+bad[-1][-220:]
    if code!=0: return f"Proses berhenti (code {code}). Kemungkinan file ke-potong / RAM habis / server ke-restart. Coba upload ulang."
    return "Demucs gagal, coba lagu MP3 lain."
@app.route("/")
def index(): return render_template("index.html")
@app.route("/original/<uid>")
def original(uid):
    p=f"uploads/{uid}_clean.wav"
    if os.path.exists(p): return send_from_directory("uploads",f"{uid}_clean.wav")
    return jsonify({"error":"original not found"}),404
@app.route("/health")
def health(): return jsonify({"ok":True,"busy":busy,"ffmpeg":FFMPEG})
@app.route("/separate", methods=["POST"])
def separate():
    global busy
    with busy_lock:
        if busy: return jsonify({"error":"Masih ada lagu yang diproses, tunggu sampe kelar dulu ya"}),429
        busy=True
    try:
        if "audio" not in request.files: busy=False; return jsonify({"error":"File nggak ke-upload, coba lagi"}),400
        f=request.files["audio"]
        if not f.filename: busy=False; return jsonify({"error":"Pilih lagu dulu"}),400
        ext=os.path.splitext(f.filename)[1].lower()
        if ext not in [".mp3",".wav",".m4a",".flac",".ogg",".aac"]: busy=False; return jsonify({"error":f"Format {ext} belum didukung. Pake MP3 / WAV / M4A aja"}),400
        uid=str(uuid.uuid4())[:8]
        raw_path=f"uploads/{uid}_input{ext}"; f.save(raw_path)
        if os.path.getsize(raw_path)<1000: busy=False; return jsonify({"error":"File-nya rusak / 0KB. Coba upload ulang"}),400
        in_path=f"uploads/{uid}_clean.wav"
        conv=subprocess.run([FFMPEG,"-y","-v","error","-i",raw_path,"-ac","2","-ar","44100",in_path],capture_output=True,text=True)
        if conv.returncode!=0 or not os.path.exists(in_path) or os.path.getsize(in_path)<1000:
            busy=False; print(f"[CONVERT FAIL] {uid} {conv.stderr[-500:]}",flush=True)
            return jsonify({"error":"File lagu ini rusak / bukan audio beneran (no channel). Coba download ulang lagunya, atau kirim via Files / Google Drive, jangan dari Safari."}),400
        print(f"[SEPARATE] {uid} clean.wav {os.path.getsize(in_path)} bytes",flush=True)
        bpm,chord=detect_bpm_chord(in_path)
        progress[uid]=2; done[uid]=False; error_msg[uid]=""; last_log[uid]=""
        out_dir=f"output/{uid}"
        cmd=[sys.executable,"-m","demucs","-n","htdemucs","--shifts","0","--overlap","0.1","-o",out_dir,in_path]
        def run():
            global busy
            try:
                p=subprocess.Popen(cmd,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,bufsize=1,errors="ignore")
                buf=""; all_out=""; last_pct=0
                while True:
                    ch=p.stdout.read(1)
                    if not ch: break
                    buf+=ch; all_out+=ch
                    if ch in ("\r","\n"):
                        last_log[uid]=buf[-500:]
                        m=re.findall(r"(\d+)%",buf)
                        if m:
                            try: last_pct=max(last_pct,int(m[-1])); progress[uid]=max(progress[uid],min(95,last_pct))
                            except: pass
                        buf=""
                p.wait(); last_log[uid]=all_out[-2000:]
                print(f"[DONE] {uid} code={p.returncode}",flush=True)
                if p.returncode==0:
                    base=glob.glob(f"output/{uid}/htdemucs/*")
                    ok=False
                    if base: ok=all(os.path.exists(os.path.join(base[0],f"{n}.wav")) and os.path.getsize(os.path.join(base[0],f"{n}.wav"))>1000 for n in ["vocals","drums","bass","other"])
                    if ok: progress[uid]=100; done[uid]=True
                    else: error_msg[uid]="Demucs kelar tapi file stem nggak lengkap. Coba lagi."
                else: error_msg[uid]=_real_error(all_out,p.returncode)
            except Exception as e:
                print(f"[ERROR] {uid} {e}",flush=True); error_msg[uid]=str(e)
            finally:
                with busy_lock: busy=False
        threading.Thread(target=run,daemon=True).start()
        return jsonify({"uid":uid,"bpm":bpm,"chord":chord})
    except Exception as e:
        with busy_lock: busy=False
        return jsonify({"error":str(e)}),500
@app.route("/progress/<uid>")
def get_progress(uid): return jsonify({"percent":progress.get(uid,0),"done":done.get(uid,False),"error":error_msg.get(uid,"")})
@app.route("/result/<uid>")
def result(uid):
    base=glob.glob(f"output/{uid}/htdemucs/*")
    if not base: return jsonify({"ready":False}),202
    stems={}
    for n in ["vocals","drums","bass","other"]:
        if os.path.exists(os.path.join(base[0],f"{n}.wav")): stems[n]=f"/download/{uid}/{n}"
    return jsonify({"ready":True,"stems":stems,"original":f"/original/{uid}"})
def _to_mp3(base,name):
    src=os.path.join(base,f"{name}.wav"); out=os.path.join(base,f"{name}.mp3")
    if os.path.exists(out): return out
    r=subprocess.run([FFMPEG,"-y","-i",src,out],capture_output=True)
    return out if r.returncode==0 else src
@app.route("/download/<uid>/<name>")
def download(uid,name):
    fmt=request.args.get("format","wav"); base=glob.glob(f"output/{uid}/htdemucs/*")[0]
    if fmt=="mp3": return send_from_directory(base,os.path.basename(_to_mp3(base,name)))
    return send_from_directory(base,f"{name}.wav")
@app.route("/download-all/<uid>")
def download_all(uid):
    fmt=request.args.get("format","wav"); names=request.args.get("stems","vocals,drums,bass,other").split(","); base=glob.glob(f"output/{uid}/htdemucs/*")[0]; buf=io.BytesIO()
    with zipfile.ZipFile(buf,"w",zipfile.ZIP_DEFLATED) as z:
        for n in names:
            src=os.path.join(base,f"{n}.wav")
            if not os.path.exists(src): continue
            if fmt=="mp3": z.write(_to_mp3(base,n),f"{n}.mp3")
            else: z.write(src,f"{n}.wav")
    buf.seek(0); return send_file(buf,mimetype="application/zip",as_attachment=True,download_name=f"stems-{uid}.zip")
if __name__=="__main__": app.run(host='0.0.0.0',debug=False,port=int(os.environ.get('PORT',5000)))
