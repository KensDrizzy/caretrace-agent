from pathlib import Path

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles

from app.api.routes import router
from app.core.bootstrap import create_schema, seed_data
from app.core.config import Settings, get_settings
from app.core.database import SessionLocal
from app.services.tool_queue import get_tool_queue_worker


def _migrate_legacy_ledger(settings: Settings) -> None:
    active_path = settings.project_root / settings.excel_path
    legacy_path = active_path.with_name("mindbridge-risk-ledger.xlsx")
    if not legacy_path.exists():
        return
    if not active_path.exists():
        legacy_path.rename(active_path)
        print(f"Renamed legacy ledger {legacy_path} -> {active_path}")
        return
    backup_path = legacy_path.with_suffix(".xlsx.bak")
    counter = 1
    while backup_path.exists():
        backup_path = legacy_path.with_suffix(f".xlsx.bak{counter}")
        counter += 1
    legacy_path.rename(backup_path)
    print(f"Both ledgers existed; archived legacy to {backup_path}")


def create_app() -> FastAPI:
    app = FastAPI(title="CareTrace Python", version="0.1.0")

    @app.middleware("http")
    async def no_cache_frontend_assets(request, call_next):
        response = await call_next(request)
        path = request.url.path
        if path == "/" or path.endswith((".html", ".js", ".css")):
            response.headers["Cache-Control"] = "no-store"
        return response

# 启动时自执行
    @app.on_event("startup")
    def startup() -> None:
        settings = get_settings()
        _migrate_legacy_ledger(settings)
        create_schema()
        db = SessionLocal()
        try:
            seed_data(db)
        finally:
            db.close()
        worker = get_tool_queue_worker(settings)
        worker.start()
        app.state.tool_queue_worker = worker

    @app.on_event("shutdown")
    def shutdown() -> None:
        worker = getattr(app.state, "tool_queue_worker", None)
        if worker is not None:
            worker.stop()

    app.include_router(router)
    static_dir = Path(__file__).resolve().parent / "static"
    app.mount("/", StaticFiles(directory=static_dir, html=True), name="static")
    return app


app = create_app()
