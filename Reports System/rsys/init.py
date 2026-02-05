import os
from .parser import parse_zip

# 简单内存缓存（轻量服务器够用）
reports_data = {}


def refresh_cache(reports_dir):
    files = [f for f in os.listdir(reports_dir) if f.lower().endswith(".zip")]
    files.sort()

    for fname in files:
        report_id = os.path.splitext(fname)[0]
        file_path = os.path.join(reports_dir, fname)
        mod_time = os.path.getmtime(file_path)

        if report_id in reports_data:
            if reports_data[report_id].get("last_modified") != mod_time:
                reports_data[report_id] = parse_zip(file_path)
        else:
            reports_data[report_id] = parse_zip(file_path)

    # 清理已删除
    for rid in list(reports_data.keys()):
        if f"{rid}.zip" not in files:
            reports_data.pop(rid, None)

    return files, reports_data
