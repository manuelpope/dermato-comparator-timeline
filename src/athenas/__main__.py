"""Command-line entry point for the Athenas trichology comparator.

Usage
-----
::

    # Use the bundled sample images, write PDF only (no PNGs on disk)
    uv run athenas --report report.pdf

    # Provide your own pair
    uv run athenas path/to/A.JPG path/to/B.JPG --report report.pdf

The pipeline runs entirely in memory; the only file produced is the PDF.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import click

from .config import PipelineConfig
from .pipeline import run_pipeline
from .report import make_report

log = logging.getLogger("athenas")


def _setup_logging(verbose: bool) -> None:
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
        stream=sys.stderr,
    )


@click.command(context_settings={"help_option_names": ["-h", "--help"]})
@click.argument(
    "baseline",
    type=click.Path(exists=True, dir_okay=False, readable=True, path_type=Path),
    required=False,
)
@click.argument(
    "followup",
    type=click.Path(exists=True, dir_okay=False, readable=True, path_type=Path),
    required=False,
)
@click.option(
    "--report",
    type=click.Path(dir_okay=False, path_type=Path),
    default="report.pdf",
    show_default=True,
    help="Path of the PDF report to write (the only file produced).",
)
@click.option(
    "--bg-method",
    type=click.Choice(["rembg", "grabcut"], case_sensitive=False),
    default=None,
    help="Background-removal backend. 'rembg' (default) is a deep-learning "
         "portrait matting model that preserves hair edges but downloads "
         "~200 MB on first run. 'grabcut' is the OpenCV offline fallback.",
)
@click.option(
    "--rembg-model",
    type=str,
    default=None,
    help="Rembg model name (e.g. 'birefnet-portrait', 'u2net', "
         "'isnet-general-use'). Ignored if --bg-method=grabcut.",
)
@click.option(
    "--rembg-alpha-threshold",
    type=click.IntRange(0, 255),
    default=None,
    help="Pixels with rembg alpha < threshold become background. "
         "Lower → more halo kept. Ignored if --bg-method=grabcut.",
)
@click.option(
    "-v", "--verbose", is_flag=True, help="Enable DEBUG-level logging.",
)
def main(
    baseline: Path | None,
    followup: Path | None,
    report: Path,
    bg_method: str | None,
    rembg_model: str | None,
    rembg_alpha_threshold: int | None,
    verbose: bool,
) -> None:
    """Athenas — visual comparator for trichology follow-up photos.

    Runs in memory; the only file produced is the PDF report (4 stages,
    2 panels per page, 2 pages total).
    """
    _setup_logging(verbose)

    project_root = Path(__file__).resolve().parents[2]
    if baseline is None:
        baseline = project_root / "data" / "baseline.JPG"
    if followup is None:
        followup = project_root / "data" / "followup.JPG"

    if not baseline.exists():
        raise click.ClickException(f"Baseline image not found: {baseline}")
    if not followup.exists():
        raise click.ClickException(f"Follow-up image not found: {followup}")

    overrides: dict[str, object] = {}
    if bg_method is not None:
        overrides["background_method"] = bg_method
    if rembg_model is not None:
        overrides["rembg_model"] = rembg_model
    if rembg_alpha_threshold is not None:
        overrides["rembg_alpha_threshold"] = rembg_alpha_threshold
    config = PipelineConfig(**overrides) if overrides else PipelineConfig()

    log.info("Baseline : %s", baseline)
    log.info("Follow-up: %s", followup)
    log.info("PDF      : %s", report)

    result = run_pipeline(baseline, followup, config=config)
    pdf_path = make_report(
        result,
        report,
        baseline_name=baseline.stem,
        followup_name=followup.stem,
    )
    click.echo(str(pdf_path), err=True)


if __name__ == "__main__":
    main()
