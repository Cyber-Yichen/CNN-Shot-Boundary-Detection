import os
from .parser import parse_zip

# 简单内存缓存（轻量服务器够用）
reports_data = {}


def refresh_cache(reports_dir: str):
    """
    扫描 reports_dir 下所有 .zip
    - 新文件：解析并加入缓存
    - 已存在但被修改：重新解析
    - 已删除：从缓存移除
    返回：(files, reports_data)
    """
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

    # 清理：文件被删了就移出缓存
    for rid in list(reports_data.keys()):
        if f"{rid}.zip" not in files:
            reports_data.pop(rid, None)

    return files, reports_data
