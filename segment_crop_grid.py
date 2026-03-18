#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import logging
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Sequence

import nrrd
import diplib as dip
import numpy as np

from mitoUnpackRecon import load_tiff_recon, parse_metadata, reshape_image, save_metadata


# --------------------------------------------------------------------------------------
# Logging
# --------------------------------------------------------------------------------------

def setup_logging(log_dir: Path, base_name: str = "extract_pipeline") -> None:
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / f"{base_name}.log"

    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[logging.FileHandler(log_path, mode="a"), logging.StreamHandler(sys.stdout)],
    )
    logging.info("=" * 70)
    logging.info("New logging session")
    logging.info("Log file: %s", log_path)
    logging.info("=" * 70)


# --------------------------------------------------------------------------------------
# Dataclasses
# --------------------------------------------------------------------------------------

@dataclass(frozen=True)
class SortSpec:
    order: tuple[str, str, str] = ("y", "x", "z")
    ascending: tuple[bool, bool, bool] = (False, True, True)

    @staticmethod
    def from_string(spec: str) -> "SortSpec":
        """
        Parse e.g.
            z+,y-,x+
            y-,x+,z+
        """
        parts = [p.strip().lower() for p in spec.split(",")]
        if len(parts) != 3:
            raise ValueError("Sort spec must have exactly 3 comma-separated axis items, e.g. 'z+,y-,x+'")

        order: list[str] = []
        ascending: list[bool] = []

        for p in parts:
            if len(p) != 2 or p[0] not in {"x", "y", "z"} or p[1] not in {"+", "-"}:
                raise ValueError(f"Invalid sort spec token: {p!r}")
            order.append(p[0])
            ascending.append(p[1] == "+")

        if set(order) != {"x", "y", "z"}:
            raise ValueError("Sort spec must contain x, y, z exactly once each")

        return SortSpec(tuple(order), tuple(ascending))


@dataclass(frozen=True)
class SizeHintsMM:
    minimum: tuple[float, float, float] | None = None  # (z, y, x)
    maximum: tuple[float, float, float] | None = None  # (z, y, x)


@dataclass
class ExtractConfig:
    threshold: float = 200.0
    margin_mm: float = 5.0
    bins: tuple[int, int, int] = (1, 1, 1)  # (x, y, z)
    sort_spec: SortSpec = field(default_factory=SortSpec)
    min_size_mm: tuple[float, float, float] | None = None  # (z, y, x)
    max_size_mm: tuple[float, float, float] | None = None  # (z, y, x)
    fallback_top_n_by_z: int | None = None
    min_component_voxels: int = 100000
    opening_diameter_mm: float = 3.0


@dataclass
class LabelInfo:
    label_id: int
    lower_xyz: tuple[int, int, int]
    upper_xyz: tuple[int, int, int]
    extent_xyz: tuple[int, int, int]
    mid_xyz: tuple[float, float, float]

    @property
    def z_extent(self) -> int:
        return self.extent_xyz[2]


# --------------------------------------------------------------------------------------
# Parsing helpers
# --------------------------------------------------------------------------------------

def expand_input_patterns(patterns: Sequence[str]) -> list[Path]:
    files: list[Path] = []
    seen: set[Path] = set()

    for pattern in patterns:
        p = Path(pattern)
        if any(ch in pattern for ch in ["*", "?", "["]):
            matches = sorted(Path().glob(pattern))
        elif p.is_dir():
            matches = sorted(p.glob("*.nrrd")) + sorted(p.glob("*.tif")) + sorted(p.glob("*.tiff"))
        else:
            matches = [p]

        for m in matches:
            m = m.resolve()
            if m.suffix.lower() in {".nrrd", ".tif", ".tiff"} and m.exists() and m not in seen:
                files.append(m)
                seen.add(m)

    return files


def parse_triplet_mm(text: str | None) -> tuple[float, float, float] | None:
    if text is None:
        return None
    parts = [x.strip() for x in text.split(",")]
    if len(parts) != 3:
        raise ValueError("Expected three comma-separated values in order z,y,x")
    z, y, x = map(float, parts)
    return (z, y, x)


def parse_bins(text: str) -> tuple[int, int, int]:
    parts = [x.strip() for x in text.split(",")]
    if len(parts) != 3:
        raise ValueError("Expected bins as three comma-separated integers in order x,y,z")
    x, y, z = map(int, parts)
    return (x, y, z)


def parse_int_list(text: str) -> list[int]:
    return [int(x.strip()) for x in text.split(",") if x.strip()]


def read_counts_file(path: Path) -> list[int]:
    """
    Supports either:
      4
      6
      5

    or CSV rows whose first non-empty field is the count.
    """
    counts: list[int] = []
    with path.open("r", newline="") as f:
        sample = f.read(2048)
        f.seek(0)
        try:
            dialect = csv.Sniffer().sniff(sample)
        except csv.Error:
            dialect = csv.excel

        reader = csv.reader(f, dialect=dialect)
        for row in reader:
            row = [c.strip() for c in row if c.strip()]
            if not row:
                continue
            counts.append(int(row[0]))
    return counts


def read_labels_file(path: Path) -> list[list[str]]:
    """
    Each line corresponds to one input file.
    Labels on a line may be comma- or semicolon-separated.

    Example:
      Xc-02,Cx_03,ab_04,dd-05
      foo,bar,baz
    """
    out: list[list[str]] = []
    with path.open("r", newline="") as f:
        for line in f:
            s = line.strip()
            if not s:
                out.append([])
                continue
            if ";" in s:
                parts = [x.strip() for x in s.split(";")]
            else:
                parts = [x.strip() for x in s.split(",")]
            out.append([p for p in parts if p])
    return out


def sanitize_filename(name: str) -> str:
    forbidden = '<>:"/\\|?*'
    return "".join("_" if c in forbidden else c for c in name).strip()


def normalize_per_file_counts(
    n_files: int,
    count_scalar: int | None,
    count_list: list[int] | None,
    count_file: Path | None,
) -> list[int | None]:
    provided = sum(x is not None for x in [count_scalar, count_list, count_file])
    if provided > 1:
        raise ValueError("Provide only one of --n, --n-list, or --n-file")

    if count_scalar is not None:
        return [count_scalar] * n_files

    if count_list is not None:
        if len(count_list) == 1:
            return count_list * n_files
        if len(count_list) != n_files:
            raise ValueError(f"--n-list has length {len(count_list)} but there are {n_files} input files")
        return count_list

    if count_file is not None:
        counts = read_counts_file(count_file)
        if len(counts) == 1:
            return counts * n_files
        if len(counts) != n_files:
            raise ValueError(f"{count_file} contains {len(counts)} counts but there are {n_files} input files")
        return counts

    return [None] * n_files


def normalize_per_file_labels(
    n_files: int,
    labels_file: Path | None,
    labels_inline: list[str] | None,
) -> list[list[str] | None]:
    """
    Two modes:
    1. --labels-file : each line = labels for one file
    2. --labels      : repeated argument, one per file, e.g.
         --labels Xc-02,Cx_03,ab_04,dd-05 --labels foo,bar,baz
       If only one --labels is given and multiple files are processed, it is used for the first
       file only unless user intentionally duplicated it.
    """
    provided = sum(x is not None for x in [labels_file, labels_inline])
    if provided > 1:
        raise ValueError("Provide only one of --labels-file or --labels")

    if labels_file is not None:
        rows = read_labels_file(labels_file)
        if len(rows) == 1 and n_files > 1:
            return rows * n_files
        if len(rows) != n_files:
            raise ValueError(f"{labels_file} contains {len(rows)} label rows but there are {n_files} input files")
        return rows

    if labels_inline is not None:
        parsed_rows = []
        for item in labels_inline:
            row = [x.strip() for x in item.split(",") if x.strip()]
            parsed_rows.append(row)

        if len(parsed_rows) == 1 and n_files == 1:
            return parsed_rows

        if len(parsed_rows) == 1 and n_files > 1:
            # Conservative behavior: same label set for all files.
            return parsed_rows * n_files

        if len(parsed_rows) != n_files:
            raise ValueError(f"--labels provided {len(parsed_rows)} times but there are {n_files} input files")
        return parsed_rows

    return [None] * n_files


# --------------------------------------------------------------------------------------
# Sorting / binning
# --------------------------------------------------------------------------------------

def _bin_axis(vals: np.ndarray, n_bins: int, offset_steps: int = 25) -> np.ndarray:
    vals = np.asarray(vals, dtype=float)
    n = vals.size
    if n == 0 or n_bins <= 1:
        return np.zeros(n, dtype=int)

    if n > 4:
        vmin, vmax = np.percentile(vals, [5, 95])
    else:
        vmin, vmax = vals.min(), vals.max()

    pitch = (vmax - vmin) / max(n_bins - 1, 1)
    if pitch <= 0:
        pitch = np.std(vals) or 1.0

    off0 = vals.min()
    offsets = off0 + np.linspace(0, pitch, offset_steps, endpoint=False)

    best_off = off0
    best_score = np.inf

    for off in offsets:
        k = np.rint((vals - off) / pitch)
        k = np.clip(k, 0, n_bins - 1)
        centers = off + k * pitch
        score = np.mean(np.abs(vals - centers))
        if score < best_score:
            best_score = score
            best_off = off

    k = np.rint((vals - best_off) / pitch)
    return np.clip(k, 0, n_bins - 1).astype(int)


def sort_label_infos(label_infos: list[LabelInfo], bins: tuple[int, int, int], sort_spec: SortSpec) -> list[LabelInfo]:
    if not label_infos:
        return []

    xs = np.array([li.mid_xyz[0] for li in label_infos], dtype=float)
    ys = np.array([li.mid_xyz[1] for li in label_infos], dtype=float)
    zs = np.array([li.mid_xyz[2] for li in label_infos], dtype=float)

    nx, ny, nz = bins
    xg = _bin_axis(xs, nx)
    yg = _bin_axis(ys, ny)
    zg = _bin_axis(zs, nz)

    axis_to_binned = {"x": xg, "y": yg, "z": zg}
    axis_to_raw = {"x": xs, "y": ys, "z": zs}
    axis_to_bins = {"x": nx, "y": ny, "z": nz}

    keys = []
    for axis, asc in zip(sort_spec.order, sort_spec.ascending):
        key = axis_to_binned[axis] if axis_to_bins[axis] > 1 else axis_to_raw[axis]
        key = key if asc else -key
        keys.append(key)

    order = np.lexsort(tuple(keys[::-1]))
    return [label_infos[i] for i in order]


# --------------------------------------------------------------------------------------
# Core extractor
# --------------------------------------------------------------------------------------

class NRRDExtractor:
    def __init__(self, config: ExtractConfig):
        self.config = config

    def _load_input(self, input_path: Path, output_dir: Path) -> tuple[np.ndarray, dict]:
        suffix = input_path.suffix.lower()
        if suffix == ".nrrd":
            return nrrd.read(input_path.as_posix())

        if suffix not in {".tif", ".tiff"}:
            raise ValueError(f"Unsupported input format: {input_path}")

        imgs, mitometa, no_pages, flat_dims = load_tiff_recon(input_path.as_posix())
        voxel_size = parse_metadata(mitometa)
        if voxel_size is None:
            raise ValueError(f"{input_path} does not contain usable scan metadata")

        output_dir.mkdir(parents=True, exist_ok=True)
        save_metadata(mitometa, (output_dir / f"{input_path.stem}_ScanSettings.xml").as_posix())

        header = {
            "type": "uint16",
            "encoding": "raw",
            "spacings": voxel_size,
            "units": ["mm", "mm", "mm"],
        }
        return reshape_image(imgs, no_pages, flat_dims), header

    def segment_objects(self, img_dip: dip.Image, px_spacing_xyz: tuple[float, float, float]) -> dip.Image:
        logging.info("Segmentation threshold: %.3f", self.config.threshold)

        mask = dip.FixedThreshold(img_dip, self.config.threshold)

        # Opening diameter defined in mm, converted conservatively using smallest spacing.
        se_diameter_px = max(1.0, self.config.opening_diameter_mm / min(px_spacing_xyz))
        mask = dip.Opening(mask, dip.SE(se_diameter_px))

        labelled = dip.Label(mask, minSize=self.config.min_component_voxels)
        logging.info("Connected components computed")
        return labelled

    def _mm_to_px_zyx(
        self,
        mm_triplet_zyx: tuple[float, float, float] | None,
        spacing_xyz: tuple[float, float, float],
    ) -> tuple[int, int, int] | None:
        if mm_triplet_zyx is None:
            return None

        sz, sy, sx = mm_triplet_zyx
        dx, dy, dz = spacing_xyz
        # return in z,y,x order to match mm input convention
        return (
            int(np.floor(sz / dz)),
            int(np.floor(sy / dy)),
            int(np.floor(sx / dx)),
        )

    def _margin_px_xyz(self, margin_mm: float, spacing_xyz: tuple[float, float, float]) -> tuple[int, int, int]:
        dx, dy, dz = spacing_xyz
        return (
            int(np.floor(margin_mm / dx)),
            int(np.floor(margin_mm / dy)),
            int(np.floor(margin_mm / dz)),
        )

    def collect_label_infos(
        self,
        img_dip: dip.Image,
        labelled_img: dip.Image,
        spacing_xyz: tuple[float, float, float],
    ) -> list[LabelInfo]:
        
        bboxes = dip.MeasurementTool.Measure(labelled_img, labelled_img, ["Minimum", "Maximum"])
        num_labels = bboxes.NumberOfObjects()
        mins = bboxes["Minimum"]
        maxs = bboxes["Maximum"]

        margin_px_xyz = self._margin_px_xyz(self.config.margin_mm, spacing_xyz)
        min_px_zyx = self._mm_to_px_zyx(self.config.min_size_mm, spacing_xyz)
        max_px_zyx = self._mm_to_px_zyx(self.config.max_size_mm, spacing_xyz)

        size_xyz = tuple(int(img_dip.Size(d)) for d in range(3))
        label_infos: list[LabelInfo] = []

        for k in range(1, num_labels + 1):
            # DIPlib returns coordinates in x,y,z order.
            min_xyz = tuple(int(mins[k][d]) for d in range(3))
            max_xyz = tuple(int(maxs[k][d]) for d in range(3))
            extent_xyz = tuple(max_xyz[d] - min_xyz[d] for d in range(3))

            # Apply size filtering if requested.
            # User-facing size hints are given in z,y,x.
            extent_zyx = (extent_xyz[2], extent_xyz[1], extent_xyz[0])

            if min_px_zyx is not None and any(extent_zyx[d] < min_px_zyx[d] for d in range(3)):
                logging.info("Skipping label %d due to undersize: extent_zyx=%s, min_zyx=%s", k, extent_zyx, min_px_zyx)
                continue

            if max_px_zyx is not None and any(extent_zyx[d] > max_px_zyx[d] for d in range(3)):
                logging.info("Skipping label %d due to oversize: extent_zyx=%s, max_zyx=%s", k, extent_zyx, max_px_zyx)
                continue

            lower_xyz = tuple(max(0, min_xyz[d] - margin_px_xyz[d]) for d in range(3))
            upper_xyz = tuple(min(size_xyz[d] - 1, max_xyz[d] + margin_px_xyz[d] + 1) for d in range(3))
            mid_xyz = tuple((min_xyz[d] + max_xyz[d]) / 2.0 for d in range(3))

            label_infos.append(
                LabelInfo(
                    label_id=k,
                    lower_xyz=lower_xyz,
                    upper_xyz=upper_xyz,
                    extent_xyz=extent_xyz,
                    mid_xyz=mid_xyz,
                )
            )

        return label_infos

    def select_label_infos(self, label_infos: list[LabelInfo]) -> list[LabelInfo]:
        """
        Behavior:
        - If min/max size hints were provided, keep all size-valid components.
        - If neither min nor max size hints are provided, require N and keep top N by Z extent.
        """
        if self.config.min_size_mm is None and self.config.max_size_mm is None:
            n = self.config.fallback_top_n_by_z
            if n is None:
                raise ValueError("No size hints were supplied, so N must be supplied to select top objects by Z extent")
            selected = sorted(label_infos, key=lambda li: li.z_extent, reverse=True)[:n]
            logging.info("Selected top %d objects by Z extent", len(selected))
            return selected

        logging.info("Selected %d objects after size filtering", len(label_infos))
        return label_infos

    def save_label_order_csv(self, out_dir: Path, ordered_infos: list[LabelInfo]) -> None:
        csv_path = out_dir / "label_order.csv"
        with csv_path.open("w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["label_id", "x_start", "y_start", "z_start", "extent_x", "extent_y", "extent_z"])
            for li in ordered_infos:
                x0, y0, z0 = li.lower_xyz
                ex, ey, ez = li.extent_xyz
                w.writerow([li.label_id, x0, y0, z0, ex, ey, ez])

    def build_output_name(self, i: int, li: LabelInfo, labels_for_file: list[str] | None) -> str:
        if labels_for_file is not None and i - 1 < len(labels_for_file):
            return sanitize_filename(labels_for_file[i - 1])

        x0, y0, z0 = li.lower_xyz
        ex, ey, ez = li.extent_xyz
        return f"object_{i:02d}_X{x0}_Y{y0}_Z{z0}_extX{ex}_extY{ey}_extZ{ez}"

    def extract_and_save_components(
        self,
        labelled_img: dip.Image,
        img_dip: dip.Image,
        header: dict,
        output_dir: Path,
        labels_for_file: list[str] | None,
        spacing_xyz: tuple[float, float, float],
    ) -> None:
        
        output_dir.mkdir(parents=True, exist_ok=True)

        label_infos = self.collect_label_infos(img_dip, labelled_img, spacing_xyz)
        label_infos = self.select_label_infos(label_infos)
        ordered_infos = sort_label_infos(label_infos, self.config.bins, self.config.sort_spec)

        if not ordered_infos:
            logging.warning("No valid labels found after filtering/selection")
            return

        self.save_label_order_csv(output_dir, ordered_infos)

        for i, li in enumerate(ordered_infos, start=1):
            x0, y0, z0 = li.lower_xyz
            x1, y1, z1 = li.upper_xyz
            k = li.label_id

            slices = [slice(x0, x1), slice(y0, y1), slice(z0, z1)]
            cropped = img_dip.At(slices)

            # Optional masking logic could go here if you want per-label isolation.
            # At the moment this preserves the original intensity crop.

            out_name = self.build_output_name(i, li, labels_for_file)
            out_path = output_dir / f"{out_name}.nrrd"

            cropped_header = header.copy()
            cropped_header["sizes"] = np.array(cropped.Sizes())
            cropped_header["encoding"] = "raw"

            logging.info("Saving %s", out_path)
            nrrd.write(out_path.as_posix(), np.asarray(cropped), header=cropped_header, compression_level=0)

    def process_file(self, input_path: Path, output_dir: Path, labels_for_file: list[str] | None) -> None:
        setup_logging(output_dir, base_name=input_path.stem)

        logging.info("Loading %s", input_path)
        data, header = self._load_input(input_path, output_dir)
        img_dip = dip.Image(data)

        spacings = header.get("spacings")
        if spacings is None or len(spacings) != 3:
            raise ValueError(f"{input_path} does not contain a valid 'spacings' entry")

        # NRRD spacings correspond to array axes z,y,x or physical axes depending on producer.
        # Your original code assumes it is compatible with the loaded array and DIPlib image.
        # Keep the same practical convention here: convert to x,y,z for coordinate work.
        dz, dy, dx = [float(v) for v in spacings]
        spacing_xyz = (dx, dy, dz)

        labelled_img = self.segment_objects(img_dip, spacing_xyz)
        self.extract_and_save_components(
            labelled_img=labelled_img,
            img_dip=img_dip,
            header=header,
            output_dir=output_dir,
            labels_for_file=labels_for_file,
            spacing_xyz=spacing_xyz,
        )


# --------------------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------------------

def build_arg_parser() -> argparse.ArgumentParser:
    epilog = """
Examples

1) Extract 4 largest objects by Z extent from all matching files:
   python extract_nrrd.py "data/*.nrrd" --n 4

2) Use size filtering instead of top-N fallback:
   python extract_nrrd.py sample.nrrd --min-size-mm 1900,40,40 --max-size-mm 2600,80,80

3) Different N per file:
   python extract_nrrd.py a.nrrd b.nrrd c.nrrd --n-list 4,6,5

4) N values from file (one count per line):
   python extract_nrrd.py "data/*.nrrd" --n-file counts.txt

5) Explicit labels for one file:
   python extract_nrrd.py sample.nrrd --n 4 --labels "Xc-02,Cx_03,ab_04,dd-05"

6) Label rows from file, one line per input file:
   python extract_nrrd.py a.nrrd b.nrrd --labels-file labels.txt

7) Custom output root:
   python extract_nrrd.py "data/*.nrrd" --n 4 --output-root extracted/

Notes

- Size hints are given in z,y,x order in millimetres.
- Bins are given in x,y,z order.
- Sort spec uses axis+direction tokens, e.g. z+,y-,x+.
- If neither --min-size-mm nor --max-size-mm is supplied, the pipeline selects top N by Z extent.
"""
    p = argparse.ArgumentParser(
        description="3D segmentation and object extraction pipeline for NRRD or reconstructed TIFF input",
        epilog=epilog,
        formatter_class=argparse.RawTextHelpFormatter,
    )

    p.add_argument(
        "inputs",
        nargs="+",
        help="Input files, directories, or glob patterns, e.g. sample.nrrd sample.tif data/ 'data/*.nrrd'",
    )

    p.add_argument("--threshold", type=float, default=200.0, help="Fixed threshold for segmentation")
    p.add_argument("--margin-mm", type=float, default=5.0, help="Cropping margin in mm around each object")
    p.add_argument("--opening-diameter-mm", type=float, default=3.0, help="Morphological opening diameter in mm")
    p.add_argument("--min-component-voxels", type=int, default=100000, help="Minimum voxel count for connected components")
    p.add_argument("--bins", type=parse_bins, default=(1, 1, 1), help="Number of bins in x,y,z order, e.g. 1,1,1")
    p.add_argument("--sort", type=str, default="y-,x+,z+", help="Sort order, e.g. z+,y-,x+ or y-,x+,z+")

    p.add_argument("--min-size-mm", type=str, default=None, help="Minimum size in z,y,x mm, e.g. 1900,40,40")
    p.add_argument("--max-size-mm", type=str, default=None, help="Maximum size in z,y,x mm, e.g. 2600,80,80")

    p.add_argument("--n", type=int, default=None, help="Number of objects to extract for all files")
    p.add_argument("--n-list", type=str, default=None, help="Comma-separated object counts per file, e.g. 4,6,5")
    p.add_argument("--n-file", type=Path, default=None, help="Text/CSV file with one object count per line")

    p.add_argument(
        "--labels",
        action="append",
        default=None,
        help=(
            "Comma-separated labels for one file. Repeat once per file if needed. "
            "Example: --labels 'Xc-02,Cx_03,ab_04,dd-05'"
        ),
    )
    p.add_argument("--labels-file", type=Path, default=None, help="Text file with one label row per input file")

    p.add_argument(
        "--output-root",
        type=Path,
        default=None,
        help=(
            "Optional root output directory. If omitted, each file is written to "
            "<input_parent>/<input_stem>/"
        ),
    )

    return p


def resolve_output_dir(input_file: Path, output_root: Path | None) -> Path:
    if output_root is None:
        return input_file.parent / input_file.stem
    return output_root / input_file.stem


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()

    input_files = expand_input_patterns(args.inputs)
    if not input_files:
        raise SystemExit("No input NRRD or TIFF files found")

    min_size_mm = parse_triplet_mm(args.min_size_mm)
    max_size_mm = parse_triplet_mm(args.max_size_mm)

    counts = normalize_per_file_counts(
        n_files=len(input_files),
        count_scalar=args.n,
        count_list=parse_int_list(args.n_list) if args.n_list else None,
        count_file=args.n_file,
    )

    labels_per_file = normalize_per_file_labels(
        n_files=len(input_files),
        labels_file=args.labels_file,
        labels_inline=args.labels,
    )

    sort_spec = SortSpec.from_string(args.sort)

    for input_file, n_for_file, labels_for_file in zip(input_files, counts, labels_per_file):
        cfg = ExtractConfig(
            threshold=args.threshold,
            margin_mm=args.margin_mm,
            bins=args.bins,
            sort_spec=sort_spec,
            min_size_mm=min_size_mm,
            max_size_mm=max_size_mm,
            fallback_top_n_by_z=n_for_file,
            min_component_voxels=args.min_component_voxels,
            opening_diameter_mm=args.opening_diameter_mm,
        )

        out_dir = resolve_output_dir(input_file, args.output_root)
        extractor = NRRDExtractor(cfg)

        logging.info("Processing %s", input_file)
        logging.info("Output dir: %s", out_dir)
        logging.info("Labels: %s", labels_for_file if labels_for_file is not None else "<auto>")
        logging.info("N fallback: %s", n_for_file)

        extractor.process_file(input_file, out_dir, labels_for_file)


if __name__ == "__main__":
    main()
