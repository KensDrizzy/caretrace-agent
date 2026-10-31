"""Cross-process locks for the supported MySQL and local SQLite deployments.

MySQL uses connection-scoped advisory locks; SQLite uses sidecar flock files.
The MySQL lock connection is deliberately separate from service transactions.
"""
from contextlib import contextmanager
import hashlib
import time
from pathlib import Path

from sqlalchemy import text

from app.services.tool_errors import ToolExecutionError


@contextmanager
def file_lock(path: Path, timeout: float = 10):
    import fcntl

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a+") as handle:
        deadline = time.monotonic() + timeout
        while True:
            try:
                fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
                break
            except BlockingIOError:
                if time.monotonic() >= deadline:
                    raise ToolExecutionError("工具资源正在执行，请稍后重试", code="RESOURCE_BUSY")
                time.sleep(0.05)
        try:
            yield
        finally:
            fcntl.flock(handle, fcntl.LOCK_UN)


@contextmanager
def operation_lock(engine, resource: str, timeout: float = 10):
    name = hashlib.sha256(resource.encode()).hexdigest()
    if engine.dialect.name == "mysql":
        with engine.connect() as conn:
            acquired = conn.execute(text("SELECT GET_LOCK(:name, :timeout)"), {"name": name, "timeout": int(timeout)}).scalar()
            if acquired != 1:
                raise ToolExecutionError("工具资源正在执行，请稍后重试", code="RESOURCE_BUSY")
            try:
                yield
            finally:
                conn.execute(text("SELECT RELEASE_LOCK(:name)"), {"name": name})
    elif engine.dialect.name == "sqlite":
        database = engine.url.database
        if not database or database == ":memory:":
            # In-memory SQLite is for unit tests, not a multi-process deployment.
            base = Path("target/tool-locks/in-memory")
        else:
            base = Path(database).resolve().with_suffix(".tool-locks")
        with file_lock(base / name, timeout):
            yield
    else:
        raise ToolExecutionError("Unsupported tool database backend", code="CONFIGURATION", retryable=False)
