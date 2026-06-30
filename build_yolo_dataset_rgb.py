"""
YOLO segmentation dataset builder for RGB orthomosaics.

The three bands of each RGB orthomosaic are read directly as [R, G, B] and
exported as standard 3-channel images compatible with YOLO:

    Channel 0 -> Red   (R, band 1)
    Channel 1 -> Green (G, band 2)
    Channel 2 -> Blue  (B, band 3)

Per-orthomosaic workflow:
    For every polygon in the matching shapefile:
        1. Read the [R, G, B] window covering only the polygon area.
        2. Stack the bands into a uint8 [H, W, 3] tile.
        3. Build the YOLO segmentation label for the main polygon and for
           every neighbouring polygon that overlaps the crop window.
        4. Write one image tile and its label to the dataset.

Tiles are written as PNG in a temporary folder and converted to JPEG when
they are moved into the final train/val split.

Note on colour handling: tiles are kept in [R, G, B] order in memory and are
converted to BGR with cv2.cvtColor before writing, because cv2.imwrite expects
BGR. This guarantees that the stored files preserve the true scene colours.

Line 71: Update this with your file path. Ensure that the name of the orthomosaic and the shapefile match.

Line 73: Change the distance used to split and search for neighbors.

Line 78 Update this with your value
"""

import gc
import glob
import json
import logging
import multiprocessing
import os
import random
import shutil
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field

import cv2
import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import rasterio
from rasterio.transform import rowcol
from rasterio.windows import from_bounds, Window
from shapely.geometry import MultiPolygon, box as shapely_box

os.environ.update({
    "OMP_NUM_THREADS":      "1",
    "OPENBLAS_NUM_THREADS": "1",
    "MKL_NUM_THREADS":      "1",
    "GDAL_NUM_THREADS":     "1",
})

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)
logging.getLogger("rasterio").setLevel(logging.ERROR)
logging.getLogger("fiona").setLevel(logging.ERROR)

# ── Constants ─────────────────────────────────────────────────────────────────
VALID_RATIO_MIN = 0.40   # minimum fraction of non-nodata pixels in a crop
MAX_DIAG_LOG    = 5_000  # cap on diagnostic records kept per orthomosaic


# ── Configuration ─────────────────────────────────────────────────────────────
@dataclass
class Config:
    base_path:    str   = "K:/.../" # change for your path
    buffer_m:     float = 1        # buffer around the polygon (metres)
    out_sz:       int   = 1024     # YOLO output tile size
    train_ratio:  float = 0.8
    random_seed:  int   = 42
    jpeg_quality: int   = 85
    n_threads:    int   = field(
        default_factory=lambda: max(1, multiprocessing.cpu_count() // 2)
    )

    @property
    def temp_dir(self):        return os.path.join(self.base_path, "temp_rgb")
    @property
    def final_dir(self):       return os.path.join(self.base_path, "dataset_yolo_rgb")
    @property
    def checkpoint_file(self): return os.path.join(self.base_path, "ckpt_rgb.json")


# ══════════════════════════════════════════════════════════════════════════════
# Save a single tile together with its YOLO label
# ══════════════════════════════════════════════════════════════════════════════
def save_crop(img_np: np.ndarray,
              annotations: list,      # [(class_id, coords_norm), ...]
              base_path: str,
              file_id: str) -> int:
    """
    Save one image tile and its YOLO segmentation label.

    The input array is ordered [R, G, B]. Because cv2.imwrite interprets the
    array as BGR, it is converted to BGR before writing so that the file on
    disk stores the true scene colours.

    Files are written as PNG in the temporary folder and converted to JPEG
    later when the dataset is finalised.
    """
    # ── Write label ───────────────────────────────────────────────
    label_lines = []
    for cid, coords_norm in annotations:
        line = " ".join(str(round(c, 6)) for c in coords_norm)
        label_lines.append(f"{cid} {line}\n")
    label_text = "".join(label_lines)

    # ── Write image ([R, G, B] -> BGR for cv2.imwrite) ────────────
    img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
    cv2.imwrite(os.path.join(base_path, "images", f"{file_id}.png"), img_bgr)
    with open(os.path.join(base_path, "labels", f"{file_id}.txt"), "w") as f:
        f.write(label_text)

    return 1


# ══════════════════════════════════════════════════════════════════════════════
# Crop a single polygon and build an [R, G, B] image
# ══════════════════════════════════════════════════════════════════════════════
def crop_polygon(src_rgb, poly, buffer_m: float) -> tuple:
    """
    Read only the polygon window from the TIF and return the RGB image.

    Output channels:
        0 -> R  (band 1 of the TIF)
        1 -> G  (band 2 of the TIF)
        2 -> B  (band 3 of the TIF)

    Nodata pixels (fill_value=0) are detected and flagged. If more than
    (1 - VALID_RATIO_MIN) of the crop is nodata, the crop is discarded.

    Returns (crop_uint8 [h, w, 3], win_transform) or (None, None).
    """
    minx, miny, maxx, maxy = poly.bounds
    win = from_bounds(
        minx - buffer_m, miny - buffer_m,
        maxx + buffer_m, maxy + buffer_m,
        src_rgb.transform,
    )

    # Convert to integer indices within the raster bounds
    col_off = max(0, int(win.col_off))
    row_off = max(0, int(win.row_off))
    col_end = min(src_rgb.width,  int(win.col_off + win.width))
    row_end = min(src_rgb.height, int(win.row_off + win.height))

    h, w = row_end - row_off, col_end - col_off
    if h < 2 or w < 2:
        return None, None

    win_clip = Window(col_off, row_off, w, h)
    win_tf   = src_rgb.window_transform(win_clip)

    # ── Read R, G and B (bands 1, 2 and 3) ───────────────────────
    data_rgb = src_rgb.read([1, 2, 3], window=win_clip,
                            boundless=True, fill_value=0).astype(np.float32)
    R = data_rgb[0]   # (h, w)
    G = data_rgb[1]   # (h, w)
    B = data_rgb[2]   # (h, w)

    # Valid-pixel mask: at least one channel > 0
    valid = (R > 0) | (G > 0) | (B > 0)
    if valid.mean() < VALID_RATIO_MIN:
        return None, None

    # ── Stack [R, G, B] as uint8 ──────────────────────────────────
    crop = np.stack([R, G, B], axis=2).clip(0, 255).astype(np.uint8)

    del data_rgb
    return crop, win_tf


def _poly_to_yolo(poly, win_tf, h_real: int, w_real: int) -> list | None:
    """
    Convert a shapely polygon to a flat list of normalised YOLO coordinates.
    """
    coords = []
    for gx, gy in poly.exterior.coords:
        px_col, px_row = rowcol(win_tf, gx, gy)[::-1]
        cx = float(np.clip(px_col / w_real, 0.0, 1.0))
        cy = float(np.clip(px_row / h_real, 0.0, 1.0))
        coords.extend([cx, cy])

    xs = coords[0::2]
    ys = coords[1::2]
    has_interior_x = any(0.0 < x < 1.0 for x in xs)
    has_interior_y = any(0.0 < y < 1.0 for y in ys)
    if not has_interior_x and not has_interior_y:
        return None
    return coords


# ══════════════════════════════════════════════════════════════════════════════
# Full processing of a SINGLE orthomosaic
# ══════════════════════════════════════════════════════════════════════════════
def process_single_set(tif: str, shp_path: str,
                       temp_dir: str, label_map: dict, config: Config):
    """
    Process one orthomosaic (RGB).
    Returns (status, name, stats_counter, diag_list).
    """
    name  = os.path.basename(tif)
    rname = os.path.splitext(name)[0]
    stats = Counter()
    diag  = []

    os.makedirs(os.path.join(temp_dir, "images"), exist_ok=True)
    os.makedirs(os.path.join(temp_dir, "labels"), exist_ok=True)

    try:
        with rasterio.open(tif) as src_rgb:
            H, W = src_rgb.height, src_rgb.width
            logger.info(f"[{name}] {H}x{W} px")

            # ── Crop by shapefile ─────────────────────────────────
            logger.info(f"[{name}] Cropping by shapefile...")
            gdf = gpd.read_file(shp_path)
            gdf = gdf[gdf.geometry.notnull()]

            logger.info(f"[{name}] Raster CRS    : {src_rgb.crs}")
            logger.info(f"[{name}] Shapefile CRS : {gdf.crs}")

            if gdf.crs is None:
                logger.error(f"[{name}] The shapefile has no CRS defined - "
                             f"assign the correct CRS before processing.")
                return "ERROR", name, Counter(), []

            if gdf.crs != src_rgb.crs:
                logger.info(f"[{name}] Reprojecting shapefile -> {src_rgb.crs}")
                gdf = gdf.to_crs(src_rgb.crs)

            # ── Geometry repair ───────────────────────────────────
            def _repair(geom):
                if geom is None or geom.is_empty:
                    return None
                if geom.is_valid:
                    return geom
                fixed = geom.buffer(0)
                if not fixed.is_valid:
                    try:
                        from shapely.validation import make_valid
                        fixed = make_valid(geom)
                    except Exception:
                        return None
                if fixed.is_empty:
                    return None
                if fixed.geom_type == "GeometryCollection":
                    polys = [g for g in fixed.geoms
                             if g.geom_type in ("Polygon", "MultiPolygon")]
                    if not polys:
                        return None
                    fixed = max(polys, key=lambda g: g.area)
                return fixed

            n_before  = len(gdf)
            gdf["geometry"] = gdf["geometry"].apply(_repair)
            gdf = gdf[gdf.geometry.notnull() & ~gdf.geometry.is_empty]
            n_invalid = n_before - len(gdf)
            if n_invalid > 0:
                logger.warning(f"[{name}] {n_invalid} geometries discarded after repair")
            else:
                logger.info(f"[{name}] geometries OK ({n_before} polygons)")

            gdf_idx = gdf.sindex

            n_ok = n_skip_nodata = n_skip_size = n_skip_noannotation = n_neighbors = 0
            for idx, row in gdf.iterrows():
                raw_sp       = row.iloc[1]
                species_name = "" if (raw_sp != raw_sp) else str(raw_sp).strip()
                if not species_name or species_name.lower() == "nan" \
                        or species_name not in label_map:
                    continue
                class_id = label_map[species_name]

                polys = (row.geometry.geoms
                         if isinstance(row.geometry, MultiPolygon)
                         else [row.geometry])

                for pidx, poly in enumerate(polys):
                    crop_u8, win_tf = crop_polygon(
                        src_rgb, poly, config.buffer_m,
                    )
                    if crop_u8 is None:
                        minx, miny, maxx, maxy = poly.bounds
                        win = from_bounds(
                            minx - config.buffer_m, miny - config.buffer_m,
                            maxx + config.buffer_m, maxy + config.buffer_m,
                            src_rgb.transform,
                        )
                        col_off = max(0, int(win.col_off))
                        row_off = max(0, int(win.row_off))
                        col_end = min(src_rgb.width,  int(win.col_off + win.width))
                        row_end = min(src_rgb.height, int(win.row_off + win.height))
                        if (row_end - row_off) < 2 or (col_end - col_off) < 2:
                            n_skip_size += 1
                        else:
                            n_skip_nodata += 1
                        continue

                    h_c, w_c = crop_u8.shape[:2]

                    img_out = cv2.resize(crop_u8,
                                         (config.out_sz, config.out_sz),
                                         interpolation=cv2.INTER_CUBIC)

                    # ── Annotation of the main polygon ────────────────
                    annotations = []
                    main_coords = _poly_to_yolo(poly, win_tf, h_c, w_c)
                    if main_coords:
                        annotations.append((class_id, main_coords))
                    else:
                        sample_pts = list(poly.exterior.coords)[:3]
                        px_vals = []
                        for gx, gy in sample_pts:
                            px_col, px_row = rowcol(win_tf, gx, gy)[::-1]
                            px_vals.append(f"px({px_col:.0f},{px_row:.0f})")
                        logger.warning(
                            f"  [no_coords] idx={idx} crop=({h_c}x{w_c}px) "
                            f"points_in_crop={px_vals} | "
                            f"geo_origin=({list(poly.exterior.coords)[0]})"
                        )

                    # ── Find neighbours that intersect the window ─────
                    minx, miny, maxx, maxy = poly.bounds
                    win_box = (
                        minx - config.buffer_m, miny - config.buffer_m,
                        maxx + config.buffer_m, maxy + config.buffer_m,
                    )
                    win_geom_query = shapely_box(*win_box)
                    candidate_idxs = list(gdf_idx.query(win_geom_query))
                    for nb_iloc in candidate_idxs:
                        nb_row = gdf.iloc[nb_iloc]
                        if nb_row.name == idx:
                            continue
                        raw_nb       = nb_row.iloc[1]
                        nb_species   = "" if (raw_nb != raw_nb) else str(raw_nb).strip()
                        if not nb_species or nb_species.lower() == "nan" \
                                or nb_species not in label_map:
                            continue
                        nb_class_id = label_map[nb_species]

                        nb_geom  = nb_row.geometry
                        nb_polys = (nb_geom.geoms
                                    if isinstance(nb_geom, MultiPolygon)
                                    else [nb_geom])
                        for nb_poly in nb_polys:
                            inter = nb_poly.intersection(win_geom_query)
                            if inter.is_empty or inter.area == 0:
                                continue
                            if (inter.area / nb_poly.area) < 0.10:
                                continue
                            nb_coords = _poly_to_yolo(nb_poly, win_tf, h_c, w_c)
                            if nb_coords:
                                annotations.append((nb_class_id, nb_coords))
                                n_neighbors += 1

                    if not annotations:
                        n_skip_noannotation += 1
                        continue

                    # Per-channel diagnostics
                    if len(diag) < MAX_DIAG_LOG:
                        m = img_out.mean(axis=(0, 1)).astype(float)
                        t = m.sum()
                        diag.append((m / t * 100.0) if t > 0 else np.zeros(3))

                    n = save_crop(
                        img_out, annotations, temp_dir,
                        f"{rname}_{idx}_{pidx}",
                    )
                    stats[species_name] += n
                    n_ok += 1

            n_skip = n_skip_nodata + n_skip_size + n_skip_noannotation
            logger.info(
                f"[{name}] OK={n_ok}  "
                f"skipped={n_skip} "
                f"(nodata={n_skip_nodata}, "
                f"outside_raster={n_skip_size}, "
                f"no_coords={n_skip_noannotation})  "
                f"neighbors={n_neighbors}"
            )
            gc.collect()

        return "OK", name, stats, diag

    except Exception as e:
        logger.error(f"ERROR [{name}]: {e}", exc_info=True)
        return "ERROR", name, Counter(), []


# ══════════════════════════════════════════════════════════════════════════════
# Utilities: classes, checkpoint, final dataset, plots
# ══════════════════════════════════════════════════════════════════════════════
def pre_scan_classes(jobs: list) -> dict:
    logger.info("Scanning classes in shapefiles...")
    classes = set()
    for _, shp in jobs:
        try:
            gdf  = gpd.read_file(shp)
            vals = gdf.iloc[:, 1].dropna().astype(str).str.strip()
            classes.update(v for v in vals if v and v.lower() != "nan")
        except Exception as e:
            logger.warning(f"  {shp}: {e}")
    label_map = {n: i for i, n in enumerate(sorted(classes))}
    logger.info(f"Classes ({len(label_map)}): {list(label_map.keys())}")
    return label_map


def load_checkpoint(path: str) -> set:
    if os.path.exists(path):
        try:
            return set(json.load(open(path)))
        except Exception:
            pass
    return set()


def save_checkpoint(path: str, done: set):
    with open(path, "w") as f:
        json.dump(sorted(done), f, indent=2)


def finalize_dataset(temp_dir: str, final_dir: str, label_map: dict,
                     ratio: float, seed: int, jpeg_quality: int):
    """
    Move images from temp/ to train/val/, converting PNG -> JPEG on the fly.
    """
    for s in ("train", "val"):
        for d in ("images", "labels"):
            os.makedirs(os.path.join(final_dir, s, d), exist_ok=True)

    imgs = glob.glob(os.path.join(temp_dir, "images", "*.png"))
    random.seed(seed)
    random.shuffle(imgs)
    split = int(len(imgs) * ratio)

    moved = skipped = 0
    for i, src_img in enumerate(imgs):
        stem    = os.path.splitext(os.path.basename(src_img))[0]
        src_lbl = os.path.join(temp_dir, "labels", f"{stem}.txt")
        if not os.path.exists(src_lbl):
            logger.warning(f"Missing label: {stem}")
            skipped += 1
            continue

        subset  = "train" if i < split else "val"
        dst_img = os.path.join(final_dir, subset, "images", f"{stem}.jpg")
        img     = cv2.imread(src_img)
        if img is None:
            skipped += 1
            continue

        cv2.imwrite(dst_img, img, [cv2.IMWRITE_JPEG_QUALITY, jpeg_quality])
        shutil.move(src_lbl,
                    os.path.join(final_dir, subset, "labels", f"{stem}.txt"))
        os.remove(src_img)
        moved += 1

    logger.info(f"Dataset: {moved} images moved, {skipped} skipped.")

    with open(os.path.join(final_dir, "data.yaml"), "w") as f:
        f.write(f"path: {os.path.abspath(final_dir)}\n")
        f.write("train: train/images\nval: val/images\n\n")
        f.write(f"nc: {len(label_map)}\n")
        f.write(f"names: {list(label_map.keys())}\n")


def save_plots(all_diag: list, final_dir: str):
    """Diagnostic plots of per-channel contribution."""
    if not all_diag:
        return
    arr    = np.array(all_diag)          # (N, 3)
    labels = ["R (red)", "G (green)", "B (blue)"]
    colors = ["#F44336", "#4CAF50", "#2196F3"]

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    # Bars: mean contribution
    bars = axes[0].bar(labels, arr.mean(axis=0),
                       color=colors, edgecolor="black", linewidth=0.8)
    axes[0].set_title("Mean per-channel contribution\n(across all crops)")
    axes[0].set_ylabel("Relative contribution (%)")
    axes[0].set_ylim(0, arr.mean(axis=0).max() * 1.3)
    for bar, v in zip(bars, arr.mean(axis=0)):
        axes[0].text(bar.get_x() + bar.get_width() / 2, v + 0.3,
                     f"{v:.1f}%", ha="center", fontsize=10, fontweight="bold")
    axes[0].grid(axis="y", linestyle="--", alpha=0.6)
    axes[0].spines[["top", "right"]].set_visible(False)

    # Boxplot: per-crop distribution
    bp = axes[1].boxplot(
        [arr[:, 0], arr[:, 1], arr[:, 2]],
        labels=labels, patch_artist=True,
        medianprops=dict(color="black", linewidth=2),
        flierprops=dict(marker="o", markersize=2, alpha=0.3),
    )
    for patch, c in zip(bp["boxes"], colors):
        patch.set_facecolor(c)
        patch.set_alpha(0.7)
    axes[1].set_title(f"Per-crop distribution  (n={len(all_diag):,})")
    axes[1].set_ylabel("Relative contribution (%)")
    axes[1].grid(axis="y", linestyle="--", alpha=0.6)
    axes[1].spines[["top", "right"]].set_visible(False)

    plt.tight_layout()
    path = os.path.join(final_dir, "channel_contribution.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Plot saved: {path}")


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    config = Config()
    random.seed(config.random_seed)
    np.random.seed(config.random_seed)

    logger.info("=" * 60)
    logger.info("YOLO dataset builder - [R, G, B] orthomosaics")
    logger.info(f"Buffer: {config.buffer_m} m  |  "
                f"Threads: {config.n_threads}  |  "
                f"Seed: {config.random_seed}")
    logger.info("=" * 60)

    # ── Discover jobs (TIF + SHP only, no DEM) ────────────────────
    ortho_tifs = sorted(
        f for f in glob.glob(os.path.join(config.base_path, "*.tif"))
        if not f.endswith("Dem.tif")
        and not f.endswith("DEM.tif")
        and not f.endswith("dem.tif")
    )
    logger.info(f"TIFs found in {config.base_path}: {len(ortho_tifs)}")

    jobs = []
    for tif in ortho_tifs:
        base   = os.path.splitext(tif)[0]
        name   = os.path.basename(tif)
        shp_path = base + ".shp"

        if os.path.exists(shp_path):
            jobs.append((tif, shp_path))
            logger.info(f"  OK {name}  ->  SHP: {os.path.basename(shp_path)}")
        else:
            logger.warning(f"  -- {name}  ->  missing: SHP ({os.path.basename(shp_path)})")

    if not jobs:
        logger.error("No valid jobs found (TIF + SHP).")
        raise SystemExit(1)
    logger.info(f"Valid jobs: {len(jobs)}")

    # ── Classes ───────────────────────────────────────────────────
    label_map = pre_scan_classes(jobs)
    if not label_map:
        logger.error("No valid classes found in the shapefiles.")
        raise SystemExit(1)

    # ── Checkpointing ─────────────────────────────────────────────
    done_set = load_checkpoint(config.checkpoint_file)
    pending  = [(t, s) for t, s in jobs
                if os.path.basename(t) not in done_set]
    logger.info(f"Pending: {len(pending)} / {len(jobs)}  "
                f"(already completed: {len(done_set)})")

    if done_set and pending:
        logger.warning(
            "NOTE: a partial checkpoint exists. Images from the already "
            "completed TIFs are NOT in temp/ - only the pending ones will be "
            "processed. To regenerate the FULL dataset, delete the file: "
            f"{config.checkpoint_file}"
        )

    # ── Parallel processing ────────────────────────────────────────
    global_stats = Counter()
    all_diag     = []

    with ThreadPoolExecutor(max_workers=config.n_threads) as ex:
        futures = {
            ex.submit(
                process_single_set,
                tif, shp, config.temp_dir, label_map, config,
            ): tif
            for tif, shp in pending
        }
        for fut in as_completed(futures):
            status, name, lstats, ldiag = fut.result()
            if status == "OK":
                global_stats.update(lstats)
                rem = MAX_DIAG_LOG - len(all_diag)
                if rem > 0:
                    all_diag.extend(ldiag[:rem])
                done_set.add(name)
                save_checkpoint(config.checkpoint_file, done_set)
                logger.info(f"  OK {name} - {sum(lstats.values())} images")
            else:
                logger.warning(f"  -- {name} - will retry on next run.")

    # ── Finalise dataset ──────────────────────────────────────────
    temp_imgs = os.path.join(config.temp_dir, "images")
    n_temp    = len(glob.glob(os.path.join(temp_imgs, "*.png"))) \
                if os.path.exists(temp_imgs) else 0

    if n_temp == 0:
        logger.error("No images were generated. Check the errors above.")
        raise SystemExit(1)

    logger.info(f"\nFinalising dataset ({n_temp} images in temp)...")
    finalize_dataset(config.temp_dir, config.final_dir, label_map,
                     config.train_ratio, config.random_seed, config.jpeg_quality)
    save_plots(all_diag, config.final_dir)
    shutil.rmtree(config.temp_dir, ignore_errors=True)

    if os.path.exists(config.checkpoint_file):
        os.remove(config.checkpoint_file)

    logger.info(f"\nDataset ready: {config.final_dir}")
    logger.info(f"   Total images: {sum(global_stats.values()):,}")
    for sp, cnt in sorted(global_stats.items()):
        logger.info(f"   {sp}: {cnt}")
