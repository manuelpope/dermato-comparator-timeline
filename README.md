# Athenas — Trichology Image Comparator

A simple computer-vision comparator that takes two trichology photographs
(*baseline* = A, *follow-up* = B) and produces a **4-figure report** — one
side-by-side A-vs-B comparison per processing stage.

> ⚠️ **Clinical disclaimer** — this is an MVP visual-aid tool. The figures
> are descriptors of the photographs, not a clinical diagnosis and **not** an
> automated follicle density count. A trained trichologist must always review
> the masks and figures before drawing any clinical conclusion.

---

## The API

The CLI above is the trichology pipeline. Next to it ships a FastAPI service
with **three domains**, each returning a PDF report:

| Domain         | Endpoint                       | What it compares                                         |
|----------------|--------------------------------|----------------------------------------------------------|
| Trichology     | `POST /v1/trichology/compare`  | 2 scalp photos (GrabCut + Lab-cluster)                   |
| Dermatology    | `POST /v1/dermatology/compare` | 1–3 lesion photos over time (4 filters)                  |
| **Aesthetics** | `POST /v1/aesthetics/compare`  | Botox before/after — normalised faces + expression lines |

See [Aesthetics — Botox before/after](#aesthetics--botox-beforeafter) for the
full pipeline of the new endpoint.

---

## The report

The pipeline produces **exactly four figures** — one per stage, each a 1×2
side-by-side comparison (A on the left, B on the right, both resized to the
same display height so they are directly comparable, panels glued together
with minimal gap). They live in memory: the CLI bundles them into a PDF and
the API does the same, so nothing is written to disk as PNGs:

| # | Figure | Stage | What you see |
|---|--------|-------|--------------|
| 1 | Original              | Original              | Raw baseline and follow-up photos, untouched. |
| 2 | Sin fondo             | Sin fondo         | GrabCut removes the room/wall, leaving only the head. |
| 3 | B&W alta definición   | B&W alta definición | Grayscale + CLAHE local contrast + unsharp-mask sharpening — individual hair strands pop out. |
| 4 | Cluster Lab k=3       | Cluster Lab k=3      | K-means in Lab space, painted with a 3-colour palette tuned for trichology. Title shows **% of visible scalp** for both photos so the thinning is measured at a glance. |

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
4 · Cluster Lab (k=3) — scalp visible: A 11.9%  ·  B 17.1%
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
Stage 1: original pair             → figura "Original"
    ↓
GrabCut (background mask) + head bbox
    ↓
Crop + resize to same display height (size normalisation)
    ↓
Stage 2: GrabCut background removal (normalised head)  → figura "Sin fondo"
    ↓
Stage 3: B&W + CLAHE + unsharp mask (normalised head)  → figura "B&W alta definición"
    ↓
Stage 4: Lab-cluster (white-filled normalised head)      → figura "Cluster Lab (k=3)"
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
│   ├── report.py           # make_report — bundles the 4 PNGs into a single-page PDF
│   └── face_expression/    # face landmarks, alignment, wrinkle detection
│       ├── face_detection.py    # MediaPipe Face Landmarker + pose + IPD
│       ├── alignment.py         # similarity transform + capture-protocol checks
│       ├── wrinkle_detection.py # per-region expression-line detector
│       ├── overlays.py          # landmark / wrinkle / change-map renderers
│       └── models/
│           └── face_landmarker.task  # vendored MediaPipe model (offline)
└── outputs/                # PDF reports written by the CLI (gitignored)
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
`l2.JPG` the user supplied) and writes a single PDF report to
`./report.pdf`. The pipeline runs entirely in memory — **no PNGs are
written to disk**.

### Bring your own pair

```bash
uv run athenas path/to/baseline.JPG path/to/followup.JPG --report ./report.pdf
```

### CLI flags

| flag | effect |
|------|--------|
| `--report PATH`            | Path of the PDF report to write (default: `report.pdf`). The **only** file produced. |
| `--bg-method [rembg\|grabcut]` | Background-removal backend. `rembg` (default) is a deep-learning portrait matting model that preserves hair edges but downloads ~200 MB on first run; `grabcut` is the OpenCV offline fallback. |
| `--rembg-model NAME`       | rembg model (`birefnet-portrait`, `u2net`, `isnet-general-use`…). Ignored with `--bg-method=grabcut`. |
| `--rembg-alpha-threshold N`| rembg alpha threshold (0–255). Lower → more halo kept. Ignored with `--bg-method=grabcut`. |
| `-v / --verbose`           | Enable DEBUG-level logging. |
| `-h / --help`              | Show help and exit. |

### The PDF report

The four comparison figures are bundled into a **3-page A4 portrait** PDF —
two photo pages (2 figures each) plus a final metrics page. No metrics file
is generated: the scalp-visible percentages live in the stage-4 figure title
and are therefore self-contained on the printed page. Layout:

```
┌──────────────────────────────────┐
│ Athenas — Trichology Report      │
├──────────────────────────────────┤
│ 1 · Original        │ 2 · Sin fondo │
│ 3 · B&W alta def.   │ 4 · Cluster Lab│
├──────────────────────────────────┤
│ Métricas (3ª página)              │
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
from athenas.report import make_report

result = run_pipeline(
    baseline_path="data/baseline.JPG",
    followup_path="data/followup.JPG",
    config=PipelineConfig(),
)
print(result.original)          # matplotlib Figure, in memory
print(result.no_background)
print(result.bw_enhanced)
print(result.cluster)
print(result.scalp_percent_a)   # % de cuero_visible en A

# Bundle the 4 figures into a 3-page PDF
make_report(result, "report.pdf")
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
│   ├── dermatology.py               # POST /v1/dermatology/compare
│   └── aesthetics.py                 # POST /v1/aesthetics/compare
└── services/
        ├── pdf.py                   # trichology + dermatology + aesthetics PDF builders
        ├── trichology/service.py    # async wrap of run_pipeline
        ├── dermatology/
        │   ├── filters.py           # 4 visual filters + parallel_apply_filters
        │   └── service.py           # N-image orchestrator
        └── aesthetics/
            ├── filters.py           # 4 visual filters + change map
            ├── analysis.py          # shared crop, region deltas, face masks
            └── service.py           # before/after orchestrator (align → crop → filters)
```

### Endpoints

| Method | Path                               | Purpose                                                                                      |
|--------|------------------------------------|----------------------------------------------------------------------------------------------|
| `GET`  | `/health`                          | { "status": "ok", "version": "0.2.0" }                                                       |
| `POST` | `/v1/trichology/compare`           | 2 scalp photos → 3-page report (same layout as the CLI)                                      |
| `POST` | `/v1/dermatology/compare`          | 1–3 lesion photos → 5-page report (1 page per filter × 4 + notes), A4 landscape              |
| `POST` | `/v1/aesthetics/compare`           | 2 Botox photos (antes/después) → 3-page report (normalised faces, expression lines, metrics) |
| `GET`  | `/docs`, `/redoc`, `/openapi.json` | Auto-generated by FastAPI                                                                    |

All responses are `application/pdf` — no JSON sidecar in MVP1.

### Guía guiada (UI web)

`static/index.html` es una interfaz mínima —un solo archivo, sin build ni
dependencias— que sirve la propia API en `/`:

```bash
uv run uvicorn app.main:app --reload --port 8000
open http://127.0.0.1:8000/          # la guía
open http://127.0.0.1:8000/docs      # OpenAPI (interfaz de Swagger)
```

Tres pestañas, una por dominio (**Tricología**, **Dermatología**, **Estética**).
Cada una explica qué hace el endpoint, los pasos de uso y los *tips* de captura,
y trae el formulario con los parámetros exactos de la API. Al enviar:

* el PDF se descarga con el nombre que devuelve el endpoint
  (`trichology_<paciente>_<fecha>_<A>_vs_<B>.pdf`) y se puede previsualizar
  en la misma página;
* los errores del servidor se muestran en la página, traducidos al español
  (400 del endpoint, 422 de validación de FastAPI);
* los campos fuera de rango se validan en el navegador antes de enviar.

La URL de la API es editable en la barra superior (se recuerda en
`localStorage`), así que la misma página sirve desde `http://127.0.0.1:8000`
—Docker expone el puerto **9096**— o abriendo el archivo directamente con
`file://`. El archivo también se puede abrir sin servidor: en ese caso apunta
a `http://127.0.0.1:8000` por defecto.

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
runs as a **non-root** user (`uid=1000`), listens on port `8000` inside the
container and is published on the host as **port `9096`**, and
**does not bundle `data/`** — mount your own folder at runtime if you
need sample photos for testing.

> ⚠️ **The image targets `linux/amd64`.** MediaPipe publishes no
> `linux/aarch64` wheel (upstream dropped it in 0.10.20+), so
> `docker-compose.yml` pins `platform: linux/amd64`. On Apple Silicon,
> Docker Desktop emulates it with QEMU — it works, the build and the
> inference are just slower. Building for `linux/arm64` is not possible
> with the current MediaPipe release.

```bash
# Build + run detached
docker compose up --build -d

# Tail logs
docker compose logs -f api

# Healthcheck (host port is 9096, not 8000)
curl -sf http://127.0.0.1:9096/health

# Guided UI (static/index.html served by the app itself)
open http://127.0.0.1:9096/

# Open the auto-generated docs
open http://127.0.0.1:9096/docs

# Tear down
docker compose down
```

Environment overrides via `.env` (optional, not required to boot). Every
variable is interpolated as `${VAR:-default}` in `docker-compose.yml`, so
anything set in `.env` (or in the host env) overrides the default:

```env
ATHENAS_CORS_ALLOW_ORIGINS=["*"]
ATHENAS_MAX_UPLOAD_SIZE_MB=25
ATHENAS_TRICHOLOGY_BG_METHOD=grabcut
ATHENAS_TRICHOLOGY_DISPLAY_HEIGHT_PX=800
ATHENAS_TRICHOLOGY_CLAHE_CLIP_LIMIT=3.0
ATHENAS_DERM_TARGET_HEIGHT_PX=600
ATHENAS_DERM_CLAHE_CLIP_LIMIT=2.0
ATHENAS_DERM_SATURATION_BOOST=1.2
ATHENAS_AESTHETICS_TARGET_HEIGHT_PX=900
ATHENAS_AESTHETICS_FACE_PADDING_FRAC=0.25
ATHENAS_AESTHETICS_WRINKLE_THRESHOLD_PERCENTILE=92.0
ATHENAS_AESTHETICS_WRINKLE_KERNEL_LENGTH=15
ATHENAS_AESTHETICS_WRINKLE_MIN_LINE_LENGTH=12
```

Run the bundled CLI from inside the running container. The `athenas`
console script ships in the venv — `uv` itself is only in the *build*
stage, so it is **not** available at runtime:

```bash
# Copy your samples in, then run the CLI writing into the bound volume
docker compose cp data/baseline.JPG api:/tmp/a.jpg
docker compose cp data/followup.JPG api:/tmp/b.jpg
docker compose exec api athenas /tmp/a.jpg /tmp/b.jpg --report /app/outputs/report.pdf
```

`/app/outputs` is the path bound to the host (`./outputs` → `/app/outputs`);
any other path stays inside the container and is lost when it exits. Note
that `docker compose cp` takes the **service** name (`api`), not the
container name (`athenas-api`).

One-off runs against a folder mounted from the host work the same way:

```bash
docker compose run --rm api athenas /data/sample_a.jpg /data/sample_b.jpg \
  --report /app/outputs/report.pdf
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

# Aesthetics: Botox before/after (the two photos are normalised automatically)
curl -F photo_before=@antes.png -F date_before=2026-01-15 \
     -F photo_after=@despues.png -F date_after=2026-03-02 \
     -F patient_id=P-42 \
  http://127.0.0.1:8000/v1/aesthetics/compare -o aesthetics_P-42.pdf
```

### Configuration

All settings come from environment variables prefixed with `ATHENAS_`
or a local `.env` file:

| Env var                                           | Default        | Effect                                                         |
|---------------------------------------------------|----------------|----------------------------------------------------------------|
| `ATHENAS_CORS_ALLOW_ORIGINS`                      | `["*"]` (open) | MVP1 is open; tighten in MVP2                                  |
| `ATHENAS_MAX_UPLOAD_SIZE_MB`                      | `25`           | per-file cap (advisory; FastAPI doesn't enforce it natively)   |
| `ATHENAS_TRICHOLOGY_BG_METHOD`                    | `"grabcut"`    | Default backend; rembg is per-request opt-in                   |
| `ATHENAS_TRICHOLOGY_DISPLAY_HEIGHT_PX`            | `800`          | Panel height inside the PDF                                    |
| `ATHENAS_TRICHOLOGY_CLAHE_CLIP_LIMIT`             | `3.0`          | B&W filter aggressiveness                                      |
| `ATHENAS_DERM_TARGET_HEIGHT_PX`                   | `600`          | Display height of normalised lesion                            |
| `ATHENAS_DERM_CLAHE_CLIP_LIMIT`                   | `2.0`          | CLAHE / color-contrast pass                                    |
| `ATHENAS_DERM_SATURATION_BOOST`                   | `1.2`          | HSV-S stretch factor                                           |
| `ATHENAS_AESTHETICS_TARGET_HEIGHT_PX`             | `900`          | Display height of the normalised face                          |
| `ATHENAS_AESTHETICS_FACE_PADDING_FRAC`            | `0.25`         | Extra margin around the shared face crop                       |
| `ATHENAS_AESTHETICS_WRINKLE_THRESHOLD_PERCENTILE` | `92.0`         | Percentile of the expression-line detector (higher = stricter) |
| `ATHENAS_AESTHETICS_WRINKLE_KERNEL_LENGTH`        | `15`           | Black-hat kernel length in px                                  |
| `ATHENAS_AESTHETICS_WRINKLE_MIN_LINE_LENGTH`      | `12`           | Minimum length (px) for a detected line segment                |

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

---

## Aesthetics — Botox before/after

`POST /v1/aesthetics/compare` takes two photos of the **same patient** —
one before and one after a Botox session — and returns a 3-page Spanish
PDF. The hard requirement here is **normalisation**: the two photos are
resampled into a single common frame so the only thing that changes
between them is the patient's face, not the camera distance, the zoom or
the head tilt.

### The normalisation pipeline

```text
photo_before ─┐
              ├─► MediaPipe Face Landmarker (478 landmarks + head pose)
photo_after ──┘
                     │
                     ▼
        1 · Similarity alignment  ── cv2.estimateAffinePartial2D over
             (LMEDS, 5 anchor points) — rotation + uniform scale +
             translation only, so the "after" face is warped onto the
             "before" frame: same head size, eyes on the same line.
                     │
                     ▼
        2 · Shared face crop      ── union of both face ovals + padding
             (one rectangle for both images)
                     │
                     ▼
        3 · Common resize         ── both cropped faces scaled to the
             (same target height)   same height (default 900 px)
                     │
                     ▼
        4 · Expression-line filter  ── per-region black-hat (6 orientations)
             + Hessian ridge response, thresholded at an adaptive percentile
                     │
                     ▼
        5 · Change map             ── |before − after| restricted to the
             (face mask only)        face oval, INFERNO colour map
```

Steps 1–3 are what make the comparison trustworthy: after them, **both
faces occupy the same pixels**, so any visible difference is attributable
to the treatment (and the report says so when the capture protocol was
not respected — see *Protocol checks* below).

### The 3 PDF pages

| Page | Content                                                                                           |
|------|---------------------------------------------------------------------------------------------------|
| 1    | Rostros normalizados (antes \| después) + Relieve de líneas de expresión (overlay magenta)        |
| 2    | Blanco y negro alta definición (antes \| después) + Mapa de cambio antes → después                |
| 3    | Métricas por región (densidad de líneas antes/después + Δ %) + chequeos de protocolo + disclaimer |

### The 4 filters

1. **Rostro normalizado** — the aligned, cropped, colour view. This is the reference frame.
2. **Relieve de líneas de expresión** — black-hat morphological filter (thin-line kernel, 6 orientations) combined with a Hessian ridge response; the detected lines are painted **magenta** over the face.
3. **Blanco y negro alta definición** — grayscale + CLAHE + unsharp mask (reuses `athenas.enhance.enhance_bw`).
4. **Mapa de cambio antes → después** — absolute difference of the two normalised faces, restricted to the face oval, rendered with the INFERNO colour map (dark = no change, hot = changed).

### Metrics — density of expression lines per region

The report compares the **line density** (detected-line pixels / region
pixels) before vs after, for seven facial regions:

| Region key             | Label (PDF)             |
|------------------------|-------------------------|
| `frontal`              | Frontal                 |
| `glabellar`            | Glabelar                |
| `periorbital_L` / `_R` | Periorbital izq. / der. |
| `nasolabial_L` / `_R`  | Nasogeniano izq. / der. |
| `perioral`             | Perioral                |

Each row shows the density before, the density after, the absolute delta
and the relative delta (%) — a **negative** delta means fewer expression
lines after the session, which is the expected Botox effect.

### Protocol checks

The report also tells the clinician when the two photos are **not**
comparable, so a bad capture is not silently interpreted as a treatment
result:

| Check         | What it measures                                                          |
|---------------|---------------------------------------------------------------------------|
| `pose_diff`   | Difference in head pose (yaw/pitch/roll) between the two photos           |
| `scale_ratio` | Ratio of the inter-pupillary distance (how different the head sizes were) |
| `lighting`    | Mean and standard deviation of the brightness of both faces               |

Each check reports `info` / `warn` / `fail` and is rendered as `[OK]` or
`[REVISAR]` in the PDF.

### Form fields

| Field               | Type            | Default | Notes                                                                    |
|---------------------|-----------------|---------|--------------------------------------------------------------------------|
| `photo_before`      | file            | —       | **Required.** Face photo before the session.                             |
| `photo_after`       | file            | —       | **Required.** Face photo after the session.                              |
| `date_before`       | string          | —       | Free-form date (`2026-01-15`); also used in the filename.                |
| `date_after`        | string          | —       | Free-form date (`2026-03-02`).                                           |
| `patient_id`        | string          | —       | Shown in the PDF header and in the filename.                             |
| `target_height`     | int (200–2000)  | `900`   | Height in px of the normalised face.                                     |
| `wrinkle_threshold` | float (50–99.5) | `92.0`  | Percentile of the line detector — higher means only the strongest lines. |

The response filename follows
`aesthetics_{patient_id}_{date_before}_{before}_vs_{after}.pdf`.

### Caveats (read before using it clinically)

* The line detector is a **descriptor**, not a clinical measurement. It
  also picks up **eyebrows and hair edges**, which is why the density is
  reported per region and always next to the visual overlay — never alone.
* The two photos must be taken with the **same camera, same distance and
  the same lighting**. The protocol checks flag the obvious violations,
  but they cannot compensate for a bad protocol.
* MediaPipe is used for the landmarks; the model asset
  (`face_landmarker.task`, ~3.8 MB) is **vendored inside the package**
  (`src/athenas/face_expression/models/`) so the API works offline and
  inside the Docker image without any download.
* A photo with no detectable face returns **HTTP 400** with a clear
  Spanish message, not a 500.

### Roadmap (ecosystem)

MVP1 ships three domains under one `/v1` prefix. The layout is designed to
host more diagnosis / advisory endpoints without restructuring:

* `v1/trichology/` — A vs B scalp comparison (shipped)
* `v1/dermatology/` — temporal lesion comparison (shipped)
* `v1/aesthetics/` — Botox before/after face comparison (shipped)
* `v1/trichoscopy/` — close-up follicle imaging (planned)
* `v1/advisory/` — diet / routine recommendations (planned)

Auth, persistent storage and async job queues are deferred to MVP2.

---

## Tests

```bash
uv run pytest -v
```

73 tests:

* 28 library smoke tests cover the small, pure functions across every
  module (`io`, `background`, `enhance`, `segmentation`, `cluster`,
  `comparison`, `report`).
* 5 dermatology filter unit tests lock the filter contracts.
* 6 PDF-builder tests (`tests/test_pdf.py`).
* 13 FastAPI smoke tests cover routing, validation, content types
  (`tests/test_api.py`).
* 21 aesthetics tests (`tests/test_aesthetics.py`) cover the HTTP
  contract, the normalisation maths, the change map, the 3-page PDF
  structure and the real end-to-end pipeline (the last two are skipped
  when the local sample photos are absent).

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
6. **Aesthetics: the line detector is not a measurement** — the
   expression-line density is a *descriptor* of the photograph, not a
   validated clinical metric. It also reacts to eyebrows and hair edges,
   so it must always be read next to the visual overlay, and the two
   photos must come from the same camera, distance and lighting.

---

## License

MIT.# dermato-comparator-timeline
