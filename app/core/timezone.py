from datetime import datetime, timedelta, timezone

CN = timezone(timedelta(hours=8))


def now_cn() -> datetime:
    return datetime.now(CN)
