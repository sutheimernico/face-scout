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


lips_app = typer.Typer(help="Phase 2 — lip reading (train/eval)", no_args_is_help=True)
app.add_typer(lips_app, name="lips")


@lips_app.command("train")
def lips_train(
    data: str = typer.Option(..., help="dataset directory of recorded utterances"),
    out: str = typer.Option("models/lips.joblib", help="output model path"),
    length: int = typer.Option(32, help="resampled sequence length"),
) -> None:
    """Train the lip-reading classifier from a dataset directory."""
    from .lips.train import train_model

    clf = train_model(data, out, length=length)
    typer.echo(f"trained on labels {clf.classes_} -> {out}")


@lips_app.command("eval")
def lips_eval(
    data: str = typer.Option(..., help="dataset directory of recorded utterances"),
    val_session: list[str] = typer.Option(..., help="session id(s) to hold out for validation"),
    length: int = typer.Option(32, help="resampled sequence length"),
) -> None:
    """Evaluate with a session-aware split; prints accuracy and a confusion matrix."""
    from .lips.train import evaluate_split

    report = evaluate_split(data, val_session, length=length)
    typer.echo(f"accuracy {report.accuracy:.3f} on {report.n} held-out samples")
    typer.echo("labels: " + ", ".join(report.labels))
    typer.echo("confusion (rows=true, cols=pred):")
    typer.echo(str(report.confusion))


@lips_app.command("record")
def lips_record(
    label: str = typer.Option(..., help="utterance label"),
    session: str = typer.Option(..., help="session id (use a fresh one per recording day)"),
    video: str = typer.Option(..., help="video clip of the single utterance to record"),
    data: str = typer.Option("lips_data", help="dataset directory"),
    index: int = typer.Option(0, help="sample index within (label, session)"),
) -> None:
    """Record one labeled utterance from a video clip into the dataset."""
    from .capture import VideoFileSource
    from .config import Config
    from .landmarker import MediaPipeLandmarker
    from .lips.record import record_utterance, save_utterance

    cfg = Config()
    source = VideoFileSource(video)
    landmarker = MediaPipeLandmarker(cfg.landmarker_model, num_faces=1)
    try:
        sequence = record_utterance(source, landmarker)
    finally:
        source.release()
        landmarker.close()
    path = save_utterance(data, label, session, sequence, index)
    typer.echo(f"saved '{label}' ({sequence.shape[0]} frames) -> {path}")


@lips_app.command("run")
def lips_run(
    model: str = typer.Option("models/lips.joblib", help="trained lip model path"),
    camera: int = 0,
    min_confidence: float = typer.Option(0.5, help="below this, show '?'"),
) -> None:
    """Live push-to-talk lip reading: hold SPACE while speaking, release to predict."""
    from .config import Config
    from .lips.live import run_live

    run_live(model, Config(camera_index=camera), min_confidence)


if __name__ == "__main__":
    app()
