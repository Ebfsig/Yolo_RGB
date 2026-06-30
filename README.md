[README.md](https://github.com/user-attachments/files/29520408/README.md)
# YOLO Segmentation Dataset Builder for RGB Orthomosaics

Supplementary material — script: `build_yolo_dataset_rgb.py`

This script builds a [YOLO](https://docs.ultralytics.com/datasets/segment/)
instance-segmentation dataset from RGB orthomosaics and their matching
polygon shapefiles. Each annotated polygon is cropped from the orthomosaic,
resized to a fixed tile, and exported together with a YOLO segmentation label
that includes the main polygon and any neighbouring polygon overlapping the
crop window. Tiles are split into `train/` and `val/` subsets and a
`data.yaml` file is written for direct use with YOLO.

---

## Requirements

- Python 3.10 or newer (the code uses the `X | None` type-union syntax).
- Python packages:

```bash
pip install numpy opencv-python rasterio geopandas shapely matplotlib
```

`rasterio`, `geopandas`, and `shapely` rely on GDAL/GEOS; installing them
through `conda` (e.g. `conda install -c conda-forge rasterio geopandas`) is
recommended if the `pip` build fails on your platform.

---

## Input data

For every orthomosaic the script expects a TIF and a shapefile **with the same
base name**, located directly inside `base_path`:

```
base_path/
├── site_01.tif          # RGB orthomosaic (3 bands: R, G, B)
├── site_01.shp          # + .shx, .dbf, .prj, ...
├── site_02.tif
├── site_02.shp
└── ...
```

Requirements for the input:

- **Orthomosaic (TIF):** at least three bands, read as band 1 = Red,
  band 2 = Green, band 3 = Blue. Nodata pixels are expected to be `0`.
- **Shapefile (SHP):** polygon (or multipolygon) geometries. The **class
  label is read from the second column** (column index 1) of the attribute
  table. Rows with empty or `NaN` labels are ignored.
- The shapefile **must have a CRS defined.** If its CRS differs from the
  raster's, it is reprojected automatically. A shapefile without a CRS causes
  that orthomosaic to be skipped with an error.
- Files ending in `Dem.tif`, `DEM.tif`, or `dem.tif` are ignored, so digital
  elevation models stored alongside the orthomosaics are not processed.

The set of classes is discovered automatically by scanning the second column
of every shapefile; class IDs are assigned in alphabetical order.

---

## Configuration

All parameters live in the `Config` dataclass near the top of the script:

| Parameter | Default | Description |
|---|---|---|
| `base_path` | `"K:/Posdoct/Datos/Ortos/"` | Root folder holding the TIF/SHP pairs and where outputs are written. **Change this to your own path.** |
| `buffer_m` | `1` | Buffer added around each polygon before cropping, in **metres**. Requires a metric (projected) CRS. |
| `out_sz` | `1024` | Output tile size in pixels (`out_sz × out_sz`). Should match the `imgsz` used during training. |
| `train_ratio` | `0.8` | Fraction of tiles assigned to `train/`; the remainder goes to `val/`. The split is at the **tile** level. |
| `random_seed` | `42` | Seed governing the train/val shuffle and `random`/`numpy`. Ensures reproducibility. |
| `jpeg_quality` | `85` | JPEG quality (0–100) for the final images. |
| `n_threads` | `cpu_count() // 2` | Number of orthomosaics processed in parallel. Halved by default because each thread loads large rasters and uses substantial memory. |

Two fixed constants control crop filtering:

- `VALID_RATIO_MIN = 0.40` — a crop is discarded if fewer than 40 % of its
  pixels are valid (non-nodata).
- `MAX_DIAG_LOG = 5000` — cap on the number of per-channel diagnostic records
  retained for the contribution plots.

---

## Usage

1. Edit `base_path` (and any other `Config` field) as needed.
2. Place the matching TIF/SHP pairs inside `base_path`.
3. Run the script:

```bash
python build_yolo_dataset_rgb.py
```

Processing runs in parallel across orthomosaics. Progress, skipped-crop
counts, and per-site summaries are printed to the console.

---

## Outputs

All outputs are written under `base_path`:

```
base_path/
├── dataset_yolo_rgb/
│   ├── train/
│   │   ├── images/   # *.jpg tiles
│   │   └── labels/   # *.txt YOLO segmentation labels
│   ├── val/
│   │   ├── images/
│   │   └── labels/
│   ├── data.yaml                  # class names + train/val paths
│   └── channel_contribution.png   # per-channel diagnostic plot
├── temp_rgb/        # temporary PNG tiles (removed on success)
└── ckpt_rgb.json    # checkpoint (removed on success)
```

- **Images:** one `out_sz × out_sz` JPEG tile per annotated polygon. Tiles are
  kept in `[R, G, B]` order in memory and converted to BGR before writing so
  that the files preserve the true scene colours.
- **Labels:** YOLO segmentation format — one line per object,
  `class_id x1 y1 x2 y2 ... xn yn` with normalised (0–1) polygon coordinates.
  Each tile may contain several objects when neighbouring polygons overlap the
  crop window (overlap threshold: 10 % of the neighbour's area).
- **`data.yaml`:** class count, class names, and relative paths to the
  `train`/`val` image folders.
- **`channel_contribution.png`:** mean and per-crop distribution of the
  relative contribution of the R, G, and B channels, as a sanity check on
  band ordering and exposure.

---

## Checkpointing

Completed orthomosaics are recorded in `ckpt_rgb.json`. If the run is
interrupted and restarted, only the pending orthomosaics are reprocessed.

> **Note:** when resuming from a partial checkpoint, tiles from
> already-completed orthomosaics are no longer present in `temp_rgb/`, so only
> the pending sites are processed. To regenerate the **entire** dataset from
> scratch, delete `ckpt_rgb.json` before running.

On a fully successful run, both `temp_rgb/` and `ckpt_rgb.json` are removed
automatically.

---

## Reproducibility notes

- The train/val split is deterministic for a fixed `random_seed`.
- The split is performed at the **tile** level, not the orthomosaic level;
  tiles from the same orthomosaic may therefore appear in both subsets. For a
  spatially independent evaluation, partition by orthomosaic instead.
- Per-thread BLAS/GDAL threading is pinned to a single thread
  (`OMP_NUM_THREADS=1`, etc.) so that parallelism is governed solely by
  `n_threads`.
