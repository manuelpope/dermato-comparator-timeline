# Athenas — Trichology Image Comparator

A simple computer-vision comparator that takes two trichology photographs
(*baseline* = A, *follow-up* = B) and produces a **4-figure report** — one
side-by-side A-vs-B comparison per processing stage.

> ⚠️ **Clinical disclaimer** — this is an MVP visual-aid tool. The figures
> are descriptors of the photographs, not a clinical diagnosis and **not** an
> automated follicle density count. A trained trichologist must always review
> the masks and figures before drawing any clinical conclusion.

---

## The report

The pipeline writes **exactly four PNGs** to the output directory — one per
stage, each a 1×2 side-by-side comparison (A on the left, B on the right,
both resized to the same display height so they are directly comparable,
panels glued together with minimal gap):

| # | File | Stage | What you see |
|---|------|-------|--------------|
| 1 | `01_original.png`  | Original              | Raw baseline and follow-up photos, untouched. |
| 2 | `02_no_background.png` | Sin fondo         | GrabCut removes the room/wall, leaving only the head. |
| 3 | `03_bw_enhanced.png`   | B&W alta definición | Grayscale + CLAHE local contrast + unsharp-mask sharpening — individual hair strands pop out. |
| 4 | `04_cluster.png`       | Cluster Lab k=3      | K-means in Lab space, painted with a 3-colour palette tuned for trichology. Title shows **% of visible scalp** for both photos so the thinning is measured at a glance. |

**Nothing else is produced** — no JSON, no ZIP, no metrics files, no
difference heatmaps. Just these four figures.

### Stage 4 palette and metric

The k-means cluster sorts pixels by Lab-L (luminance) and paints them with
a fixed palette:

| L-rank | Colour swatch          | Meaning                          |
|--------|------------------------|----------------------------------|
| Darkest | near-black             | Hair shafts                     |
| Mid     | **coral** (eye-catcher)| **Exposed scalp / skin** — the area of this colour IS the cue for bald zones |
| Lightest| off-white cream        | Background + highlights / bright reflections |

The figure title displays the **percentage of skin-coloured pixels** for
both A and B, so the change in thinning is read directly off the figure.
On the bundled sample pair:

```
4 · Cluster Lab (k=3) — scalp visible: A 17.7%  ·  B 18.1%
```

> **Caveat** — the metric is computed inside the *head bbox* returned by
> GrabCut, which includes the neck and a sliver of clothing. So the number
> is an **approximation** that mixes scalp with neck/forehead skin. It is
> reliable as a *relative* measure (A vs B comparison), not as an absolute
> scalp-percentage. To isolate vertex / hairline / occipital sub-regions
> you would need additional ROI segmentation, which is intentionally out
> of scope for MVP1.

---

## Pipeline

```
A.JPG + B.JPG
    ↓
Stage 1: original pair             → 01_original.png
    ↓
GrabCut (background mask) + head bbox
    ↓
Crop + resize to same display height (size normalisation)
    ↓
Stage 2: GrabCut background removal (normalised head)  → 02_no_background.png
    ↓
Stage 3: B&W + CLAHE + unsharp mask (normalised head)  → 03_bw_enhanced.png
    ↓
Stage 4: Lab-cluster (white-filled normalised head)      → 04_cluster.png
```

The GrabCut mask drives **two normalisations**:

1. **Background removal** — for stages 2 and 3 the background is zeroed.
2. **Head-size normalisation** — the bbox of the foreground + 5 % padding
   is cropped and resized to a fixed display height (800 px) so the two
   heads occupy the same vertical extent in every figure. Any difference
   you see is then due to hair coverage / colour, **not** to camera
   distance or zoom. Stage 1 is left untouched on purpose so the clinician
   can still audit the capture protocol (lighting, framing, angle).

For stage 4 the background is filled with **white**, not zeroed — otherwise
the k-means would mis-classify the BLACK background as the "darkest"
cluster and steal the "hair" slot, which would make the scalp-percentage
metric wrong.

### Pose-agnostic head-region detection

The GrabCut mask covers the whole foreground (head *and* torso), so the
raw bbox is too tall — the buzo leaks into the crop. `comparison.head_bbox`
strips the body out so the crop is always focused on the head:

* It computes the mask-width per row inside the foreground bbox and skips
  the top 20 % (where wispy hair strands make the silhouette sparse and
  would otherwise collapse the bbox to a 1-pixel strip).
* If a sharp narrowing appears within `head_cutoff_ratio` (default 0.55)
  of the bbox height, that's treated as the neck transition.
* Otherwise the head is cropped to the upper `head_cutoff_ratio` of the
  bbox — the head always sits in the top portion of the frame for any
  pose (upright, side-back, bent-forward).

Tune `head_cutoff_ratio` in `PipelineConfig` if your capture protocol
puts the head lower in the frame.

---

## Project layout

```
.
├── pyproject.toml          # uv / hatchling project metadata + CLI entry point
├── README.md
├── .python-version         # uv pin: 3.12
├── data/
│   ├── baseline.JPG        # bundled sample (the user's l1.JPG)
│   └── followup.JPG        # bundled sample (the user's l2.JPG)
├── src/athenas/            # importable package (src-layout)
│   ├── __init__.py
│   ├── __main__.py         # CLI entry point (`athenas` command)
│   ├── config.py           # PipelineConfig dataclass — all tunables
│   ├── io.py               # read_rgb, save_rgb, save_gray
│   ├── background.py       # GrabCut + apply_mask + clean_binary
│   ├── segmentation.py     # hair_structure_mask (darkness × gradient)
│   ├── enhance.py          # to_grayscale, clahe_enhance, unsharp_mask, enhance_bw
│   ├── cluster.py          # numpy_kmeans, cluster_lab, density_heatmap
│   ├── comparison.py       # make_comparison (side-by-side figure) + make_overlay
│   ├── pipeline.py         # orchestrator: run_pipeline()
│   └── report.py           # make_report — bundles the 4 PNGs into a single-page PDF
└── outputs/                # 4 generated PNGs + (optional) report.pdf (gitignored)
```

---

## Installation

The project uses [`uv`](https://docs.astral.sh/uv/) and a `src/`-layout
package. Python 3.12 is pinned via `.python-version`.

```bash
# Production dependencies and the `athenas` console script
uv sync

# Optional: dev tools (pytest, ruff)
uv sync --extra dev
```

---

## Usage

### Quickest path — bundled sample images

```bash
uv run athenas
```

Runs against `data/baseline.JPG` + `data/followup.JPG` (the `l1.JPG` /
`l2.JPG` the user supplied) and writes the four figures into `./outputs/`.

### Bring your own pair

```bash
uv run athenas path/to/baseline.JPG path/to/followup.JPG -o ./report
```

### CLI flags

| flag | effect |
|------|--------|
| `-o / --output DIR` | Output directory (default: `outputs`). |
| `--report PATH`     | Also bundle the 4 figures into a single-page minimalist PDF at `PATH`. |
| `-v / --verbose`   | Enable DEBUG-level logging. |
| `-h / --help`      | Show help and exit. |

### Minimalist PDF report

After the 4 PNGs are produced you can ask for a single-page PDF that
stacks the figures vertically with their labels, ready to print or share:

```bash
uv run athenas data/baseline.JPG data/followup.JPG \
              -o outputs \
              --report outputs/report.pdf
```

The PDF is A4 portrait, no metrics file is generated — the
scalp-visible percentages live in the stage-4 figure title and are
therefore self-contained on the printed page. Layout:

```
┌──────────────────────────────────┐
│ Athenas — Trichology Report      │
├──────────────────────────────────┤
│ 1 · Original                    │
│ 2 · Sin fondo (cabeza norm.)    │
│ 3 · B&W alta definición         │
│ 4 · Cluster Lab (k=3)           │
└──────────────────────────────────┘
```

You can also call `report.make_report(result, "report.pdf")` directly
from Python if you want to bundle an existing run into a PDF without
re-running the pipeline.

### As a library

```python
from pathlib import Path
from athenas.config import PipelineConfig
from athenas.pipeline import run_pipeline

result = run_pipeline(
    baseline_path="data/baseline.JPG",
    followup_path="data/followup.JPG",
    output_dir="report",
    config=PipelineConfig(),
)
print(result.original)        # Path to 01_original.png
print(result.no_background)   # Path to 02_no_background.png
print(result.bw_enhanced)     # Path to 03_bw_enhanced.png
print(result.cluster)         # Path to 04_cluster.png
```

---

## Key design decisions

* **Four figures, period.** The user explicitly asked for a simple report.
  No JSON, no metrics files, no ZIP, no clustering-as-overlay, no
  difference heatmaps. Each figure is a single side-by-side A vs B
  comparison.
* **Same display size, minimal gap.** `comparison.fit_to_canvas` forces
  both panels to the same height (default 800 px) and
  `gridspec_kw={"wspace": 0.01}` glues them together so the boundary is
  easy to scan across.
* **GrabCut for background removal.** Lightweight, dependency-free, gives a
  tight head silhouette in one call.
* **CLAHE + unsharp mask for the B&W view.** CLAHE boosts local contrast
  without over-saturating skin tones; unsharp-mask sharpening then makes
  individual hair strands visible.
* **Lab-cluster for stage 4.** K-means in Lab space sorted by ascending L
  so the darkest cluster = hair and the mid cluster = scalp. The **coral**
  palette colour used for "skin" is chosen to make exposed scalp POP
  visually — that is the direct cue for thinning. The metric (% scalp
  visible) is baked into the figure title.
* **Pure NumPy + OpenCV** — no PyTorch / TensorFlow / scikit-learn.
* **Src-layout package** — modern Python packaging recommendation.
* **Frozen config dataclass** — every threshold lives in
  `config.PipelineConfig` so you can re-run experiments without touching
  module code.
* **CLI via `click`** — discoverable, type-checked arguments; the script
  can also be imported as a library.
* **Headless matplotlib** — `matplotlib.use("Agg")` is set inside
  `comparison.py`, so the pipeline works on servers without a display.
* **Reproducible** — K-means seed and RNG seeds are fixed where used; the
  CLI installs the same package version everywhere through `uv.lock`.

---

## FastAPI service (MVP1)

The CLI above is unchanged. A new FastAPI surface ships next to it:

```
app/                                # new — FastAPI package
├── main.py                          # factory, CORS (open in dev), /health
├── core/config.py                   # pydantic-settings Settings
├── api/v1/
│   ├── router.py                    # aggregate v1 router
│   ├── trichology.py                # POST /v1/trichology/compare
│   └── dermatology.py               # POST /v1/dermatology/compare
└── services/
        ├── pdf.py                   # trichology + dermatology PDF builders
        ├── trichology/service.py    # async wrap of run_pipeline
        └── dermatology/
            ├── filters.py           # 4 visual filters + parallel_apply_filters
            └── service.py           # N-image orchestrator
```

### Endpoints

| Method | Path                                | Purpose                                                                            |
|--------|-------------------------------------|----------------------------------------------------------------------------------|
| `GET`  | `/health`                           | `{ "status": "ok", "version": "0.2.0" }`                                        |
| `POST` | `/v1/trichology/compare`            | 2 scalp photos → 3-page report (same layout as the CLI)                          |
| `POST` | `/v1/dermatology/compare`           | 1–3 lesion photos → 5-page report (1 page per filter × 4 + notes), A4 landscape   |
| `GET`  | `/docs`, `/redoc`, `/openapi.json`  | Auto-generated by FastAPI                                                        |

All responses are `application/pdf` — no JSON sidecar in MVP1.

### Run locally

```bash
# 1. Install — FastAPI deps are part of the default dependency group now
uv sync --extra dev

# 2. Boot the API (development mode, auto-reload)
uv run uvicorn app.main:app --reload --port 8000

# 3. Open the auto-generated docs in a browser
open http://127.0.0.1:8000/docs
```

### Run with Docker / Docker Compose

A multi-stage `Dockerfile` (slim Python 3.12 + `uv` for fast installs) and
a single-service `docker-compose.yml` ship next to the source. The image
runs as a **non-root** user (`uid=1000`), exposes port `8000`, and
**does not bundle `data/`** — mount your own folder at runtime if you
need sample photos for testing.

```bash
# Build + run detached
docker compose up --build -d

# Tail logs
docker compose logs -f api

# Healthcheck
curl -sf http://127.0.0.1:8000/health

# Open the auto-generated docs
open http://127.0.0.1:8000/docs

# Tear down
docker compose down
```

Environment overrides via `.env` (optional, not required to boot):

```env
ATHENAS_CORS_ALLOW_ORIGINS=["*"]
ATHENAS_MAX_UPLOAD_SIZE_MB=25
ATHENAS_TRICHOLOGY_BG_METHOD=grabcut
ATHENAS_TRICHOLOGY_DISPLAY_HEIGHT_PX=800
ATHENAS_TRICHOLOGY_CLAHE_CLIP_LIMIT=3.0
ATHENAS_DERM_TARGET_HEIGHT_PX=600
ATHENAS_DERM_CLAHE_CLIP_LIMIT=2.0
ATHENAS_DERM_SATURATION_BOOST=1.2
```

Run the bundled CLI from inside the running container:

```bash
docker compose run --rm api uv run athenas /data/sample_a.jpg /data/sample_b.jpg -o ./scratch
```

### Examples (API)

```bash
# Trichology: two scalp photos, GrabCut background removal (default)
curl -F baseline=@data/baseline.JPG -F followup=@data/followup.JPG \
  http://127.0.0.1:8000/v1/trichology/compare -o trichology.pdf

# Trichology: opt into the deep-learning rembg backend (≈200 MB download on first call)
curl -F baseline=@data/baseline.JPG -F followup=@data/followup.JPG \
  -F bg_method=rembg \
  http://127.0.0.1:8000/v1/trichology/compare -o trichology_rembg.pdf

# Dermatology: 2 lesion photos with explicit dates + patient id
curl -F photo_a=@d1.JPG -F date_a=2026-01-15 \
     -F photo_b=@d2.JPG -F date_b=2026-07-11 \
     -F patient_id=P-42 \
  http://127.0.0.1:8000/v1/dermatology/compare -o derm2.pdf

# Dermatology: 3 lesion photos (A, B, C) with dates
curl -F photo_a=@d1.JPG -F date_a=2026-01-15 \
     -F photo_b=@d2.JPG -F date_b=2026-07-11 \
     -F photo_c=@d3.JPG -F date_c=2026-09-20 \
  http://127.0.0.1:8000/v1/dermatology/compare -o derm3.pdf

# Dermatology: tune the filters via form fields
curl -F photo_a=@d1.JPG \
     -F target_height=800 \
     -F clahe_clip_limit=3.0 \
     -F saturation_boost=1.4 \
  http://127.0.0.1:8000/v1/dermatology/compare -o derm_tuned.pdf
```

### Configuration

All settings come from environment variables prefixed with `ATHENAS_`
or a local `.env` file:

| Env var                                  | Default         | Effect                                                      |
|------------------------------------------|-----------------|-------------------------------------------------------------|
| `ATHENAS_CORS_ALLOW_ORIGINS`             | `["*"]` (open)  | MVP1 is open; tighten in MVP2                               |
| `ATHENAS_MAX_UPLOAD_SIZE_MB`             | `25`            | per-file cap (advisory; FastAPI doesn't enforce it natively)|
| `ATHENAS_TRICHOLOGY_BG_METHOD`           | `"grabcut"`     | Default backend; rembg is per-request opt-in                |
| `ATHENAS_TRICHOLOGY_DISPLAY_HEIGHT_PX`   | `800`           | Panel height inside the PDF                                 |
| `ATHENAS_TRICHOLOGY_CLAHE_CLIP_LIMIT`    | `3.0`           | B&W filter aggressiveness                                   |
| `ATHENAS_DERM_TARGET_HEIGHT_PX`          | `600`           | Display height of normalised lesion                         |
| `ATHENAS_DERM_CLAHE_CLIP_LIMIT`          | `2.0`           | CLAHE / color-contrast pass                                 |
| `ATHENAS_DERM_SATURATION_BOOST`          | `1.2`           | HSV-S stretch factor                                        |

### Dermatology: the 4 filters

Every lesion image (1–3 photos) is processed by 4 visual filters, run
in parallel per image via `asyncio.gather` + `asyncio.to_thread`. The
PDF then renders **one page per filter** with the temporal series
side-by-side, so the clinician can see *A vs B vs C* under each specific
filter treatment:

1. **CLAHE on Lab-L** — luminance contrast boost, colour preserved (`a*`, `b*` untouched).
2. **B&W high-definition** — grayscale + CLAHE + unsharp mask (reuses `athenas.enhance.enhance_bw`).
3. **Color high-contrast** — Lab-L CLAHE **plus** HSV-S stretch (`× ATHENAS_DERM_SATURATION_BOOST`).
4. **Scale normalisation** — resize to a fixed height (`ATHENAS_DERM_TARGET_HEIGHT_PX`) so temporal shots compare 1:1.

The PDF is **always 5 pages** regardless of the photo count:

| Page | Content                                                            |
|------|--------------------------------------------------------------------|
| 1    | CLAHE (Lab-L) — A · B · C (temporal series under this filter)      |
| 2    | B&W alta definición — A · B · C                                    |
| 3    | Color con mejor contraste — A · B · C                              |
| 4    | Normalización de escala — A · B · C                                |
| 5    | Notas — clinical disclaimer + per-filter explanation + lesion list |

A4 landscape layout. Each photo is captioned with its number, filename
and date below the panel.

### Roadmap (ecosystem)

MVP1 ships two domains under one `/v1` prefix. The layout is designed to
host more diagnosis / advisory endpoints without restructuring:

* `v1/trichology/` — A vs B scalp comparison (shipped)
* `v1/dermatology/` — temporal lesion comparison (shipped)
* `v1/trichoscopy/` — close-up follicle imaging (planned)
* `v1/advisory/` — diet / routine recommendations (planned)

Auth, persistent storage and async job queues are deferred to MVP2.

---

## Tests

```bash
uv run pytest -v
```

41 tests:

* 26 library smoke tests cover the small, pure functions across every
  module (`io`, `background`, `enhance`, `segmentation`, `cluster`,
  `comparison`, `report`).
* 5 dermatology filter unit tests lock the filter contracts.
* 8 FastAPI smoke tests cover routing, validation, content types
  (`tests/test_api.py`).

---

## Limitations of this MVP and how to evolve it

1. **Fix the capture protocol** (camera, focal, distance, lighting, hair
   preparation) before adding algorithmic complexity.
2. **Validate against a ground truth** (manual follicle counts or
   trichoscopy images) before claiming any quantitative metric is
   clinically meaningful.
3. **Region-specific ROI** — for vertex vs hairline vs occipital, the
   `grabcut_rect_frac` rectangle and the cluster/segmentation thresholds
   in `PipelineConfig` need to be retuned.
4. **Physical scale** — without a ruler / marker in the scene we cannot
   convert pixel measurements to millimetres.
5. **Beyond MVP1** — `cluster.density_heatmap()` is already implemented
   and exported as a library function; it returns a turbo-colormap heatmap
   of local skin density and could replace or augment stage 4 once the
   visual protocol is fixed.

---

## License

MIT.# dermato-comparator-timeline
