def safe_float(x):
    try:
        return float(x)
    except Exception:
        return None


def safe_int(x):
    try:
        return int(float(x))
    except Exception:
        return None


def seconds_to_hms(seconds):
    if seconds is None:
        return None
    try:
        s = int(float(seconds))
    except Exception:
        return None

    h = s // 3600
    m = (s % 3600) // 60
    sec = s % 60

    if h > 0:
        return f"{h}小时{m}分钟{sec}秒"
    if m > 0:
        return f"{m}分钟{sec}秒"
    return f"{sec}秒"
