from __future__ import annotations

from pathlib import Path

from app.core.config import Settings


def finetuned_model_status(settings: Settings) -> dict:
    return model_status(
        settings,
        name=settings.finetuned_model_name,
        model_dir=settings.finetuned_model_dir,
        model_file=settings.finetuned_model_file,
    )


def campus_model_status(settings: Settings) -> dict:
    return model_status(
        settings,
        name=settings.campus_model_name,
        model_dir=settings.campus_model_dir,
        model_file=settings.campus_model_file,
    )


def model_status(settings: Settings, *, name: str, model_dir: str, model_file: str) -> dict:
    root = settings.project_root
    resolved_dir = resolve_model_dir(settings, model_dir)
    gguf_path = resolved_dir / model_file
    modelfile_path = resolved_dir / "Modelfile"
    return {
        "name": name,
        "directory": str(resolved_dir.relative_to(root)) if resolved_dir.is_relative_to(root) else str(resolved_dir),
        "ggufFile": model_file,
        "ggufExists": gguf_path.exists(),
        "ggufSizeBytes": gguf_path.stat().st_size if gguf_path.exists() else 0,
        "modelfileExists": modelfile_path.exists(),
        "ollamaCreateCommand": f"scripts/create-finetuned-model.sh",
    }


def resolve_model_dir(settings: Settings, model_dir: str | None = None) -> Path:
    path = Path(model_dir or settings.finetuned_model_dir)
    return path if path.is_absolute() else settings.project_root / path
