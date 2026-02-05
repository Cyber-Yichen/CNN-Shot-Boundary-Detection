import os
import glob
import cv2
import streamlit as st

MOVIE_DIR = os.path.join("dataset", "movie")
PHOTO_DIR = os.path.join("dataset", "movie", "photo")
VIDEO_EXTS = ("*.mp4", "*.mov", "*.mkv", "*.avi", "*.m4v", "*.webm")

st.set_page_config(page_title="Movie Frame Sampler", layout="centered")
st.title("🎬 Movie 抽帧并保存到服务器")

def list_videos(movie_dir: str):
    files = []
    for ext in VIDEO_EXTS:
        files.extend(glob.glob(os.path.join(movie_dir, ext)))
    return sorted(files)

def safe_mkdir(path: str):
    os.makedirs(path, exist_ok=True)

def get_video_meta(video_path: str):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        cap.release()
        return None, None, None
    fps = cap.get(cv2.CAP_PROP_FPS)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    if fps is None or fps <= 0:
        fps = 24.0
    duration = total / fps if fps > 0 else None
    return float(fps), int(total), float(duration) if duration is not None else None

def save_sampled_frames(
    video_path: str,
    interval_sec: float,
    out_dir: str,
    prefix: str,
    overwrite: bool,
    progress_cb=None
):
    safe_mkdir(out_dir)

    fps, total, duration = get_video_meta(video_path)
    if fps is None:
        raise RuntimeError("视频无法打开，可能路径/编码有问题。")

    step = max(1, int(round(fps * interval_sec)))
    indices = list(range(0, total, step))
    n = len(indices)
    if n == 0:
        return 0, fps, total, duration

    # 如果选择覆盖，先删掉同名前缀的旧帧
    if overwrite:
        old = glob.glob(os.path.join(out_dir, f"{prefix}_*.png"))
        for p in old:
            try:
                os.remove(p)
            except:
                pass

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        cap.release()
        raise RuntimeError("视频无法打开（VideoCapture failed）。")

    saved = 0
    for i, idx in enumerate(indices, start=1):
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ok, frame_bgr = cap.read()
        if not ok or frame_bgr is None:
            # 读不到就跳出（有些视频末尾会读失败）
            break

        out_name = f"{prefix}_{i:04d}.png"
        out_path = os.path.join(out_dir, out_name)

        # 直接保存 PNG（opencv 写入用 BGR 就行）
        ok_w = cv2.imwrite(out_path, frame_bgr)
        if ok_w:
            saved += 1

        if progress_cb:
            progress_cb(i, n, idx, fps)

    cap.release()
    return saved, fps, total, duration


videos = list_videos(MOVIE_DIR)
if not videos:
    st.error(f"没找到视频文件：请确认目录存在且有视频 -> {MOVIE_DIR}")
    st.stop()

video_path = st.selectbox("选择视频", videos, format_func=lambda p: os.path.basename(p))
video_base = os.path.splitext(os.path.basename(video_path))[0]

fps, total, duration = get_video_meta(video_path)
if fps is None:
    st.error("这个视频打不开（路径/编码可能有问题）。")
    st.stop()

st.caption(f"检测到：FPS={fps:.3f} | 总帧数={total} | 时长≈{duration:.1f} 秒")

interval_sec = st.slider("每隔几秒抽一帧", 0.1, 30.0, 1.0, 0.1)

col1, col2 = st.columns(2)
with col1:
    overwrite = st.checkbox("覆盖同名旧帧", value=True)
with col2:
    st.caption(f"输出目录：{PHOTO_DIR}")

run_btn = st.button("开始抽帧并保存 PNG", type="primary")

if run_btn:
    safe_mkdir(PHOTO_DIR)

    progress = st.progress(0)
    status = st.empty()

    def cb(i, n, frame_idx, fps_):
        pct = int(i * 100 / n)
        progress.progress(min(100, pct))
        status.write(f"进度：{i}/{n}  |  当前读取帧：{frame_idx}  |  时间≈{frame_idx / fps_:.2f}s")

    try:
        saved, fps2, total2, dur2 = save_sampled_frames(
            video_path=video_path,
            interval_sec=interval_sec,
            out_dir=PHOTO_DIR,
            prefix=video_base,
            overwrite=overwrite,
            progress_cb=cb
        )
        progress.progress(100)
        status.write(f"完成 ✅ 保存 {saved} 张 PNG 到：{PHOTO_DIR}（命名：{video_base}_0001.png ...）")
    except Exception as e:
        st.error(f"失败：{e}")
