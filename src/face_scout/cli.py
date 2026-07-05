"""Typer CLI entrypoint. Heavy imports are deferred so ``--help`` works anywhere."""

from __future__ import annotations

import typer

app = typer.Typer(help="face-scout — live webcam face tracking + identity", no_args_is_help=True)


@app.command()
def run(
    camera: int = 0,
    video: str = typer.Option(None, help="process a video file instead of the webcam"),
) -> None:
    """Run live face tracking + identity recognition."""
    from .app import run as _run
    from .capture import VideoFileSource
    from .config import Config

    source = VideoFileSource(video) if video else None
    _run(Config(camera_index=camera), source=source)


@app.command()
def enroll(name: str, camera: int = 0, shots: int = 5) -> None:
    """Enroll a known face into the gallery (NAME = person's label)."""
    from .config import Config
    from .enroll import enroll as _enroll

    _enroll(name, Config(camera_index=camera, enroll_shots=shots))


if __name__ == "__main__":
    app()
