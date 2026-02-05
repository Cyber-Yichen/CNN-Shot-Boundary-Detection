import os

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
REPORTS_DIR = os.path.join(BASE_DIR, "reports")
os.makedirs(REPORTS_DIR, exist_ok=True)

# 表格预览行数上限（防止大表卡死）
SHEET_PREVIEW_LIMIT = 200
