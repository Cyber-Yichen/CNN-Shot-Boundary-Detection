import os
import cv2
import time
import json
import random
import threading
import traceback
import xml.etree.ElementTree as ET
from dataclasses import dataclass, asdict
from typing import List, Dict, Optional, Tuple

from flask import Flask, request, redirect, url_for, render_template, jsonify, send_from_directory, abort

BUILD_ID = "dataset-tool-v1-8003"

BASE_DIR = os.path.abspath(os.path.dirname(__file__))

DEFAULTS = {
    "MOVIE_DIR": os.path.join(BASE_DIR, "movie"),
    "XML_DIR": os.path.join(BASE_DIR, "xml"),
    "FRAME_DIR": os.path.join(BASE_DIR, "p"),
    "OUT_CUT_DIR": os.path.join(BASE_DIR, "out_cut"),
    "OUT_NONCUT_DIR": os.path.join(BASE_DIR, "out_noncut"),
    "FPS": 24,
    "OUT_W": 1280,
    "OUT_H": 720,
}

VIDEO_EXTS = {".mp4", ".mov", ".avi", ".mkv", ".flv", ".wmv"}

app = Flask(__name__)

# -----------------------------
# Job state (in-memory)
# -----------------------------
@dataclass
class Job:
    job_id: str
    video_name: str
    mode: str  # "extract" | "cut" | "noncut" | "all"
    status: str  # "queued" | "running" | "done" | "error"
    progress: int
    total: int
    message: str
    started_at: Optional[float] = None
    finished_at: Optional[float] = None
    params: Optional[dict] = None
    outputs: Optional[dict] = None
    error_trace: Optional[str] = None

JOBS: Dict[str, Job] = {}
JOBS_LOCK = threading.Lock()

def _new_job_id() -> str:
    return f"job-{int(time.time()*1000)}-{random.randint(1000,9999)}"

def job_update(job_id: str, **kwargs):
    with JOBS_LOCK:
        j = JOBS.get(job_id)
        if not j:
            return
        for k, v in kwargs.items():
            setattr(j, k, v)

def job_get(job_id: str) -> Optional[Job]:
    with JOBS_LOCK:
        return JOBS.get(job_id)

def job_list() -> List[Job]:
    with JOBS_LOCK:
        return list(JOBS.values())

# -----------------------------
# Utilities
# -----------------------------
def ensure_dirs():
    for k in ["MOVIE_DIR", "XML_DIR", "FRAME_DIR", "OUT_CUT_DIR", "OUT_NONCUT_DIR"]:
        os.makedirs(DEFAULTS[k], exist_ok=True)

def list_videos(movie_dir: str) -> List[str]:
    if not os.path.isdir(movie_dir):
        return []
    out = []
    for fn in sorted(os.listdir(movie_dir)):
        p = os.path.join(movie_dir, fn)
        if os.path.isfile(p) and os.path.splitext(fn)[1].lower() in VIDEO_EXTS:
            out.append(fn)
    return out

def base_name(filename: str) -> str:
    return os.path.splitext(filename)[0]

def parse_timecode_like_0214_to_frame(tc: str, fps: int = 24) -> int:
    """
    Accept formats:
      - "02:14" meaning mm:ss (or ss:ff?) 你这里写的是 02:14 这种更像 mm:ss
      - "hh:mm:ss"
      - "mm:ss:ff" 也可能
    为了稳：按冒号数量判别：
      1个冒号 -> mm:ss
      2个冒号 -> hh:mm:ss 或 mm:ss:ff(如果最后段<fps倾向当ff)
    """
    parts = tc.strip().split(":")
    parts = [p.strip() for p in parts if p.strip() != ""]
    if len(parts) == 2:
        mm = int(parts[0])
        ss = int(parts[1])
        return (mm * 60 + ss) * fps
    if len(parts) == 3:
        a, b, c = parts
        ia, ib, ic = int(a), int(b), int(c)
        # heuristic: if c < fps, treat as frames, and a/b are mm:ss
        if ic < fps:
            mm, ss, ff = ia, ib, ic
            return (mm * 60 + ss) * fps + ff
        else:
            hh, mm, ss = ia, ib, ic
            return (hh * 3600 + mm * 60 + ss) * fps
    raise ValueError(f"Unsupported timecode format: {tc}")

def xml_parse_cut_frames_fcp7(xml_path: str, target_fps: int = 24) -> List[int]:
    """
    解析 DaVinci 导出的 FCP7 XML（常见结构：<xmeml><sequence>...<clipitem><start>...）
    我们取每个 clipitem 的 start（非0）作为剪辑点。
    如果xml里带 timebase/ntsc，会做时间换算到 target_fps。
    """
    tree = ET.parse(xml_path)
    root = tree.getroot()

    sequence = root.find(".//sequence")
    if sequence is None:
        return []

    # timeline fps
    timeline_fps = None
    rate_elem = sequence.find(".//rate")
    if rate_elem is not None:
        tb = rate_elem.find("timebase")
        ntsc = rate_elem.find("ntsc")
        if tb is not None and tb.text:
            try:
                timeline_fps = float(tb.text.strip())
            except:
                timeline_fps = None
        if timeline_fps and ntsc is not None and ntsc.text and ntsc.text.strip().upper() == "TRUE":
            timeline_fps = timeline_fps * 1000.0 / 1001.0

    if not timeline_fps or timeline_fps <= 0:
        timeline_fps = float(target_fps)

    cut_frames = []
    for clip in root.findall(".//video//track//clipitem"):
        st = clip.find("start")
        if st is None or not st.text:
            continue
        try:
            start_frame = int(st.text.strip())
        except:
            continue
        if start_frame != 0:
            cut_frames.append(start_frame)

    cut_frames = sorted(set(cut_frames))

    # convert to target fps scale if needed
    if abs(timeline_fps - target_fps) > 1e-3:
        adjusted = []
        for cf in cut_frames:
            sec = cf / timeline_fps
            adjusted.append(int(round(sec * target_fps)))
        cut_frames = sorted(set(adjusted))

    return [int(x) for x in cut_frames]

def extract_frames_to_dir(video_path: str, frame_dir: str, fps: int, job_id: str) -> int:
    """
    以目标fps抽帧保存为 jpg（你也可改 png）
    命名：VideoName_000000.jpg ...
    返回抽到的帧数（按目标fps）
    """
    os.makedirs(frame_dir, exist_ok=True)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    orig_fps = cap.get(cv2.CAP_PROP_FPS)
    if not orig_fps or orig_fps <= 0:
        orig_fps = float(fps)

    vname = base_name(os.path.basename(video_path))
    frame_count = 0
    saved = 0
    next_t = 0.0

    # We do not know total frames precisely at target fps; estimate using duration
    total_orig = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    duration_sec = total_orig / orig_fps if total_orig > 0 else None
    est_total = int(duration_sec * fps) if duration_sec else 0
    job_update(job_id, total=max(est_total, 1))

    ok, frame = cap.read()
    while ok:
        cur_t = frame_count / orig_fps
        if cur_t + 1e-9 >= next_t:
            out_path = os.path.join(frame_dir, f"{vname}_{saved:06d}.jpg")
            cv2.imwrite(out_path, frame)
            saved += 1
            next_t += 1.0 / fps
            job_update(job_id, progress=saved, message=f"Extracting frames... {saved}")
        frame_count += 1
        ok, frame = cap.read()

    cap.release()
    job_update(job_id, progress=saved, total=max(saved, 1), message=f"Extract done: {saved} frames")
    return saved

def write_segment_video_from_frames(
    frames_dir: str,
    video_base: str,
    start_frame: int,
    length: int,
    out_path: str,
    fps: int,
    out_w: int,
    out_h: int,
):
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(out_path, fourcc, fps, (out_w, out_h))

    for i in range(start_frame, start_frame + length):
        img_path = os.path.join(frames_dir, f"{video_base}_{i:06d}.jpg")
        img = cv2.imread(img_path)
        if img is None:
            continue
        img = cv2.resize(img, (out_w, out_h))
        writer.write(img)
    writer.release()

def cut_samples_from_cutpoints(
    video_base: str,
    frames_dir: str,
    total_frames: int,
    cut_frames: List[int],
    pre_frames: int,
    post_frames: int,
    out_dir: str,
    fps: int,
    out_w: int,
    out_h: int,
    job_id: str,
) -> List[str]:
    os.makedirs(out_dir, exist_ok=True)
    outputs = []
    valid = []
    for cf in cut_frames:
        if cf - pre_frames < 0:
            continue
        if cf + post_frames - 1 >= total_frames:
            continue
        valid.append(cf)

    job_update(job_id, total=max(len(valid), 1), progress=0)
    idx = 0
    for cf in valid:
        idx += 1
        start = cf - pre_frames
        length = pre_frames + post_frames
        out_path = os.path.join(out_dir, f"{video_base}_cut_{idx:03d}.mp4")
        write_segment_video_from_frames(frames_dir, video_base, start, length, out_path, fps, out_w, out_h)
        outputs.append(out_path)
        job_update(job_id, progress=idx, message=f"Cut sample {idx}/{len(valid)} @cut={cf}")
    return outputs

def noncut_random_segments(
    video_base: str,
    frames_dir: str,
    total_frames: int,
    cut_frames: List[int],
    seg_len: int,
    seg_count: int,
    start_id: int,
    out_dir: str,
    fps: int,
    out_w: int,
    out_h: int,
    job_id: str,
) -> List[str]:
    """
    从不含剪辑点的区间中随机挑选段落，每段 seg_len 帧，生成 seg_count 个 mp4
    输出命名：V{start_id:03d}.mp4, V{start_id+1:03d}...
    """
    os.makedirs(out_dir, exist_ok=True)
    cuts = sorted(set([c for c in cut_frames if 0 <= c < total_frames]))
    boundaries = [0] + cuts + [total_frames]  # cut frame itself belongs to next segment start by our definition
    intervals = []
    for i in range(len(boundaries) - 1):
        s = boundaries[i]
        e = boundaries[i + 1] - 1
        if e - s + 1 >= seg_len:
            intervals.append((s, e))

    if not intervals:
        return []

    # build list of all possible starts per interval (weighted)
    weights = []
    acc = 0
    meta = []
    for (s, e) in intervals:
        possible = (e - s + 1) - seg_len + 1
        if possible <= 0:
            continue
        acc += possible
        weights.append(acc)
        meta.append((s, possible))
    total_possible = acc
    if total_possible <= 0:
        return []

    outputs = []
    used = set()

    job_update(job_id, total=max(seg_count, 1), progress=0)
    made = 0
    tries = 0
    max_tries = seg_count * 200  # avoid infinite loop

    while made < seg_count and tries < max_tries:
        tries += 1
        r = random.randrange(total_possible)
        # find interval index
        idx = 0
        while idx < len(weights) and r >= weights[idx]:
            idx += 1
        if idx >= len(meta):
            idx = len(meta) - 1
        prev = weights[idx - 1] if idx > 0 else 0
        offset = r - prev

        s, possible = meta[idx]
        start = s + offset
        if start in used:
            continue

        # ensure segment doesn't include a cut frame
        # we require [start, start+seg_len-1] to not contain any cut
        end = start + seg_len - 1
        # fast check by scanning cuts (cuts count typically small)
        bad = False
        for c in cuts:
            if start <= c <= end:
                bad = True
                break
        if bad:
            continue

        used.add(start)
        out_name = f"V{start_id + made:03d}.mp4"
        out_path = os.path.join(out_dir, out_name)
        write_segment_video_from_frames(frames_dir, video_base, start, seg_len, out_path, fps, out_w, out_h)
        outputs.append(out_path)
        made += 1
        job_update(job_id, progress=made, message=f"Non-cut {made}/{seg_count} start={start}")

    return outputs

# -----------------------------
# Worker
# -----------------------------
def run_job(job_id: str, video_file: str, mode: str, params: dict):
    try:
        ensure_dirs()

        movie_dir = params.get("movie_dir", DEFAULTS["MOVIE_DIR"])
        xml_dir = params.get("xml_dir", DEFAULTS["XML_DIR"])
        frame_root = params.get("frame_dir", DEFAULTS["FRAME_DIR"])
        out_cut_dir = params.get("out_cut_dir", DEFAULTS["OUT_CUT_DIR"])
        out_noncut_dir = params.get("out_noncut_dir", DEFAULTS["OUT_NONCUT_DIR"])

        fps = int(params.get("fps", DEFAULTS["FPS"]))
        out_w = int(params.get("out_w", DEFAULTS["OUT_W"]))
        out_h = int(params.get("out_h", DEFAULTS["OUT_H"]))

        pre_frames = int(params.get("pre_frames", 10))
        post_frames = int(params.get("post_frames", 10))

        noncut_len = int(params.get("noncut_len", 20))
        noncut_count = int(params.get("noncut_count", 10))
        start_id = int(params.get("start_id", 1))

        video_path = os.path.join(movie_dir, video_file)
        vbase = base_name(video_file)
        xml_path = os.path.join(xml_dir, vbase + ".xml")

        if not os.path.isfile(video_path):
            raise RuntimeError(f"Video not found: {video_path}")
        if not os.path.isfile(xml_path) and mode in ("cut", "noncut", "all"):
            raise RuntimeError(f"XML not found: {xml_path} (需要与视频同名)")

        # each video has its own frame folder to avoid mixing
        frames_dir = os.path.join(frame_root, vbase)
        os.makedirs(frames_dir, exist_ok=True)

        job_update(job_id, status="running", started_at=time.time(), message="Starting...")

        outputs = {"frames_dir": frames_dir, "cut_videos": [], "noncut_videos": []}

        # Step A: extract frames if needed OR if frames folder empty
        need_extract = (mode in ("extract", "all", "cut", "noncut"))
        # if frames already exist, we can skip extraction unless forced
        force_extract = bool(params.get("force_extract", False))
        existing = [f for f in os.listdir(frames_dir) if f.lower().endswith(".jpg")]
        if need_extract and (force_extract or len(existing) == 0):
            job_update(job_id, message="Extracting frames...")
            total_frames = extract_frames_to_dir(video_path, frames_dir, fps, job_id)
        else:
            # estimate existing frames count by max index
            total_frames = len(existing)
            job_update(job_id, message=f"Frames exist, skip extract: {total_frames} frames")

        outputs["total_frames"] = total_frames

        cut_frames = []
        if mode in ("cut", "noncut", "all"):
            cut_frames = xml_parse_cut_frames_fcp7(xml_path, target_fps=fps)
            outputs["cut_frames"] = cut_frames

        # Step B: cut samples
        if mode in ("cut", "all"):
            job_update(job_id, message="Generating cut samples...")
            out_list = cut_samples_from_cutpoints(
                vbase, frames_dir, total_frames, cut_frames,
                pre_frames, post_frames,
                os.path.join(out_cut_dir, vbase),
                fps, out_w, out_h,
                job_id
            )
            outputs["cut_videos"] = out_list

        # Step C: noncut samples
        if mode in ("noncut", "all"):
            job_update(job_id, message="Generating non-cut samples...")
            out_list = noncut_random_segments(
                vbase, frames_dir, total_frames, cut_frames,
                noncut_len, noncut_count, start_id,
                os.path.join(out_noncut_dir, vbase),
                fps, out_w, out_h,
                job_id
            )
            outputs["noncut_videos"] = out_list

        job_update(job_id, status="done", finished_at=time.time(), message="Done", outputs=outputs, progress=1, total=1)

    except Exception as e:
        job_update(job_id, status="error", finished_at=time.time(), message=str(e), error_trace=traceback.format_exc())

# -----------------------------
# Routes
# -----------------------------
@app.route("/", methods=["GET"])
def index():
    ensure_dirs()
    movie_dir = request.args.get("movie_dir", DEFAULTS["MOVIE_DIR"])
    xml_dir = request.args.get("xml_dir", DEFAULTS["XML_DIR"])

    videos = list_videos(movie_dir)
    # map whether xml exists
    xml_exists = {}
    for v in videos:
        bn = base_name(v)
        xml_exists[v] = os.path.isfile(os.path.join(xml_dir, bn + ".xml"))

    jobs = sorted(job_list(), key=lambda j: j.started_at or 0, reverse=True)
    return render_template(
        "index.html",
        build_id=BUILD_ID,
        defaults=DEFAULTS,
        movie_dir=movie_dir,
        xml_dir=xml_dir,
        videos=videos,
        xml_exists=xml_exists,
        jobs=jobs
    )

@app.route("/start", methods=["POST"])
def start():
    ensure_dirs()
    video_file = request.form.get("video_file")
    mode = request.form.get("mode", "all")

    # params from form
    params = {
        "movie_dir": request.form.get("movie_dir", DEFAULTS["MOVIE_DIR"]),
        "xml_dir": request.form.get("xml_dir", DEFAULTS["XML_DIR"]),
        "frame_dir": request.form.get("frame_dir", DEFAULTS["FRAME_DIR"]),
        "out_cut_dir": request.form.get("out_cut_dir", DEFAULTS["OUT_CUT_DIR"]),
        "out_noncut_dir": request.form.get("out_noncut_dir", DEFAULTS["OUT_NONCUT_DIR"]),
        "fps": int(request.form.get("fps", DEFAULTS["FPS"])),
        "out_w": int(request.form.get("out_w", DEFAULTS["OUT_W"])),
        "out_h": int(request.form.get("out_h", DEFAULTS["OUT_H"])),
        "pre_frames": int(request.form.get("pre_frames", 10)),
        "post_frames": int(request.form.get("post_frames", 10)),
        "noncut_len": int(request.form.get("noncut_len", 20)),
        "noncut_count": int(request.form.get("noncut_count", 10)),
        "start_id": int(request.form.get("start_id", 1)),
        "force_extract": True if request.form.get("force_extract") == "on" else False,
    }

    if not video_file:
        return redirect(url_for("index"))

    job_id = _new_job_id()
    job = Job(
        job_id=job_id,
        video_name=video_file,
        mode=mode,
        status="queued",
        progress=0,
        total=1,
        message="Queued",
        params=params,
        outputs=None
    )
    with JOBS_LOCK:
        JOBS[job_id] = job

    t = threading.Thread(target=run_job, args=(job_id, video_file, mode, params), daemon=True)
    t.start()

    return redirect(url_for("job_page", job_id=job_id))

@app.route("/job/<job_id>", methods=["GET"])
def job_page(job_id):
    j = job_get(job_id)
    if not j:
        abort(404)
    return render_template("job.html", job=j, build_id=BUILD_ID)

@app.route("/api/job/<job_id>", methods=["GET"])
def api_job(job_id):
    j = job_get(job_id)
    if not j:
        return jsonify({"error": "not found"}), 404
    return jsonify(asdict(j))

@app.route("/download", methods=["GET"])
def download():
    """
    用于下载生成的视频或帧（可选）
    ?path=/absolute/or/relative
    安全起见：仅允许 BASE_DIR 内部路径
    """
    path = request.args.get("path", "")
    if not path:
        abort(400)

    # normalize
    if not os.path.isabs(path):
        path = os.path.join(BASE_DIR, path)
    path = os.path.abspath(path)

    if not path.startswith(BASE_DIR):
        abort(403)

    if not os.path.isfile(path):
        abort(404)

    d = os.path.dirname(path)
    fn = os.path.basename(path)
    return send_from_directory(d, fn, as_attachment=True)

def run_server():
    ensure_dirs()
    # IMPORTANT: 8003
    app.run(host="0.0.0.0", port=8003, debug=False, threaded=True)

if __name__ == "__main__":
    run_server()
