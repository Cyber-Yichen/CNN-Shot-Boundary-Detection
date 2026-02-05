# app.py - Reports System (modular version)
import os
from io import BytesIO
from zipfile import ZipFile

from flask import Flask, render_template, send_file, abort

from rsys.config import REPORTS_DIR
from rsys.cache import refresh_cache, reports_data
from rsys.utils import safe_float, safe_int, seconds_to_hms
from rsys.parser import parse_zip  # for lazy-load single report if needed

# ---- optional site meta ----
ICP_TEXT = ""
BUILD_ID = ""
try:
    from rsys.site_meta import ICP_TEXT as _ICP_TEXT, BUILD_ID as _BUILD_ID
    ICP_TEXT = _ICP_TEXT
    BUILD_ID = _BUILD_ID
except Exception:
    pass


app = Flask(__name__)


@app.context_processor
def inject_site_meta():
    # used by _footer.html
    return {"ICP_TEXT": ICP_TEXT, "BUILD_ID": BUILD_ID}


def make_metrics_panel(final_kv: dict, title: str):
    # final_test sheet keys are lowercased in parser.parse_kv_sheet
    order = [
        ("threshold", "阈值"),
        ("precision", "精确率"),
        ("recall", "召回率"),
        ("f1", "F1"),
        ("acc", "准确率"),
        ("pr_auc_ap", "AP(PR-AUC)"),
        ("roc_auc", "AUC(ROC)"),
        ("tp", "TP"),
        ("fp", "FP"),
        ("tn", "TN"),
        ("fn", "FN"),
        ("pos_pred_rate", "正类预测率"),
    ]
    items = []
    for k, label in order:
        if k in final_kv and final_kv[k] is not None:
            items.append({"label": label, "value": final_kv[k]})
    return {"title": title, "items": items}


def make_env_panel(info: dict, title: str):
    # run_info keys are lowercased in parser.parse_kv_sheet
    keys = [
        ("time_start", "开始时间"),
        ("time", "记录时间"),
        ("platform", "平台"),
        ("processor", "CPU架构"),
        ("python_version", "Python版本"),
        ("torch_version", "Torch版本"),
        ("cuda_available", "CUDA可用"),
        ("cuda_device_count", "GPU数量"),
        ("cuda_device_name_0", "GPU0名称"),
        ("device_used", "使用设备"),
        ("epochs", "训练轮数"),
        ("train_seconds", "训练时长"),
        ("threshold_default", "默认阈值"),
        ("batch_size", "batch_size"),
        ("pos_weight_used", "pos_weight"),
        ("lr_init", "初始学习率"),
        ("optimizer", "优化器"),
        ("report_folder_name", "报告名"),
    ]
    items = []
    for k, label in keys:
        if k in info and info[k] is not None:
            v = info[k]
            if k == "train_seconds":
                v = seconds_to_hms(v) or v
            items.append({"key": k, "label": label, "value": v})
    return {"title": title, "items": items}


def build_test_timelines(per_video_table: dict):
    """
    per_video_table is produced by rsys.parser.sheet_to_table():
    {headers: [...], rows: [...], total_rows:..., shown_rows:...}

    We compute tp/fp/fn frame lists using gt_cuts and pred_cuts.
    """
    if not per_video_table or not per_video_table.get("headers") or not per_video_table.get("rows"):
        return []

    headers = [str(h).strip().lower() for h in per_video_table["headers"]]

    def find_idx(*names):
        for n in names:
            n = n.lower()
            if n in headers:
                return headers.index(n)
        return None

    # robust aliases
    i_vid = find_idx("vid", "video", "video_id", "name", "filename", "file")
    i_total = find_idx("total_frames", "n_frames", "frames", "total", "frame_count")
    i_gt = find_idx("gt_cuts", "gt", "gt_cut_frames", "gt_cut", "gt_frames", "gt_list", "gt_indices")
    i_pred = find_idx("pred_cuts", "pred", "pred_cut_frames", "pred_cut", "pred_frames", "pred_list", "pred_indices")
    i_tp = find_idx("tp")
    i_fp = find_idx("fp")
    i_fn = find_idx("fn")

    def parse_cut_list(s):
        if s is None:
            return []
        if isinstance(s, (int, float)):
            return [int(s)]
        s = str(s).strip()
        if not s or s.lower() == "none":
            return []
        s = s.replace("，", ",")
        out = []
        for p in [x.strip() for x in s.split(",") if x.strip()]:
            try:
                out.append(int(float(p)))
            except Exception:
                pass
        return out

    timelines = []
    for row in per_video_table["rows"]:
        def get(i):
            return row[i] if (i is not None and i < len(row)) else None

        vid = get(i_vid) or "unknown"
        total_frames = safe_int(get(i_total)) or 0

        gt_list = parse_cut_list(get(i_gt))
        pred_list = parse_cut_list(get(i_pred))

        gt_set = set(gt_list)
        pred_set = set(pred_list)

        tp_frames = sorted(gt_set & pred_set)
        fp_frames = sorted(pred_set - gt_set)
        fn_frames = sorted(gt_set - pred_set)

        timelines.append({
            "vid": str(vid),
            "total_frames": int(total_frames),
            "gt_frames": sorted(gt_set),
            "pred_frames": sorted(pred_set),
            "tp_frames": tp_frames,
            "fp_frames": fp_frames,
            "fn_frames": fn_frames,
            "tp": safe_int(get(i_tp)) if get(i_tp) is not None else len(tp_frames),
            "fp": safe_int(get(i_fp)) if get(i_fp) is not None else len(fp_frames),
            "fn": safe_int(get(i_fn)) if get(i_fn) is not None else len(fn_frames),
        })

    return timelines


@app.route("/")
def index():
    files, cache = refresh_cache(REPORTS_DIR)

    total_reports = len(files)
    total_models = sum(len(d.get("model_files", [])) for d in cache.values())

    total_epochs = 0
    total_time_seconds = 0.0

    reports_list = []
    for report_id, d in cache.items():
        train_info = d.get("train", {}).get("run_info", {})

        # epochs
        epochs = safe_int(train_info.get("epochs"))
        if epochs is None:
            epochs = len(d.get("train", {}).get("epoch_metrics", {}).get("epoch", []))
        epochs = epochs or 0
        total_epochs += epochs

        # train_seconds
        ts = d.get("train_seconds")
        if isinstance(ts, (int, float)):
            total_time_seconds += float(ts)

        # pos_weight_used
        pos_weight = train_info.get("pos_weight_used", None)
        if pos_weight is None:
            pos_weight = train_info.get("pos_weight", None)
        pos_weight_f = safe_float(pos_weight)

        reports_list.append({
            "id": report_id,
            "epochs": epochs,
            "weight": pos_weight_f,  # used by index.html sorting data-weight
            "weight_str": f"{pos_weight_f:.2f}" if pos_weight_f is not None else "-",
            "models": len(d.get("model_files", [])),
            "time_str": seconds_to_hms(ts),
        })

    total_time_str = seconds_to_hms(total_time_seconds)
    reports_list.sort(key=lambda x: x["id"])

    return render_template(
        "index.html",
        total_reports=total_reports,
        total_models=total_models,
        total_epochs=total_epochs,
        total_time_str=total_time_str,
        reports_list=reports_list,
    )


@app.route("/reports/<report_id>")
def report_detail(report_id):
    file_path = os.path.join(REPORTS_DIR, f"{report_id}.zip")
    if not os.path.exists(file_path):
        return abort(404)

    # lazy load if not cached
    if report_id not in reports_data:
        reports_data[report_id] = parse_zip(file_path)

    d = reports_data[report_id]

    train_info = d.get("train", {}).get("run_info", {})
    train_curves = d.get("train", {}).get("epoch_metrics", {})
    train_final = d.get("train", {}).get("final_test", {})

    test_info = d.get("test", {}).get("run_info", {})
    test_final = d.get("test", {}).get("final_test", {})

    test_dataset_summary = d.get("test", {}).get("dataset_summary")
    test_per_video = d.get("test", {}).get("per_video")
    test_classification_report = d.get("test", {}).get("classification_report")

    test_timelines = build_test_timelines(test_per_video)

    return render_template(
        "detail.html",
        report_id=report_id,

        # panels
        train_env_panel=make_env_panel(train_info, "训练环境与参数（train）"),
        test_env_panel=make_env_panel(test_info, "测试环境与参数（cut_test）"),
        train_metrics_panel=make_metrics_panel(train_final, "训练集最终指标（train / val 汇总）"),
        test_metrics_panel=make_metrics_panel(test_final, "测试集最终指标（cut_test）"),

        # charts
        train_curves=train_curves,

        # test timelines + sheet previews
        test_timelines=test_timelines,
        test_dataset_summary=test_dataset_summary,
        test_per_video=test_per_video,
        test_classification_report=test_classification_report,

        # downloads
        model_files=d.get("model_files", []),
        excel_files=d.get("excel_files", []),
    )


@app.route("/reports/<report_id>/download")
def download_report(report_id):
    file_path = os.path.join(REPORTS_DIR, f"{report_id}.zip")
    if not os.path.exists(file_path):
        return abort(404)
    return send_file(file_path, as_attachment=True, download_name=f"{report_id}.zip")


@app.route("/reports/<report_id>/file/<path:filename>")
def download_file(report_id, filename):
    file_path = os.path.join(REPORTS_DIR, f"{report_id}.zip")
    if not os.path.exists(file_path):
        return abort(404)

    if report_id not in reports_data:
        reports_data[report_id] = parse_zip(file_path)
    d = reports_data[report_id]

    # allow only listed files
    if filename not in d.get("model_files", []) and filename not in d.get("excel_files", []):
        return abort(404)

    zf = ZipFile(file_path)
    try:
        blob = zf.read(filename)
    finally:
        zf.close()

    return send_file(BytesIO(blob), as_attachment=True, download_name=os.path.basename(filename))


if __name__ == "__main__":
    # production: use gunicorn; this is for local debug
    app.run(host="0.0.0.0", port=8002, debug=False)
