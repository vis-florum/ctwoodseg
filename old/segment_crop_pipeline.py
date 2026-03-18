from __future__ import annotations

import argparse
import csv
import logging
import os
import sys
import re

from sklearn.cluster import DBSCAN
from datetime import datetime
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Sequence

import nrrd
import diplib as dip
import numpy as np

from multiprocessing import Pool, cpu_count

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



def segment_objects(img_dip, px_scale, t=200):
    logging.info(f"Fixed threshold: {t:.2f}")

    mask = dip.FixedThreshold(img_dip, t)
    # se_diameter = 1.5 / px_scale
    se_diameter = 3 / px_scale

    # Remove small artifacts
    mask = dip.Opening(mask, dip.SE(se_diameter))

    # Label connected components
    labelled_img = dip.Label(mask, minSize=100000)
    
    return labelled_img


def save_label_nrrd(labelled_img, header, out_path):
    """Saves the label image as a NRRD file."""
    label_array = np.asarray(labelled_img, dtype=np.int8)
    label_header = header.copy()
    label_header['type'] = 'int8'
    label_header['enconding'] = 'raw'

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    nrrd.write(out_path, label_array, header=label_header, compression_level=0)
    

# def cluster_axis(coords_singleaxis, tolerance):
#     """Clusters 1D coordinates using DBSCAN, returns group labels for each input."""
#     coords_singleaxis = np.array(coords_singleaxis).reshape(-1, 1)
#     # DBSCAN eps = tolerance, min_samples=1 means even isolated points get a label
#     labels = DBSCAN(eps=tolerance, min_samples=1, metric='euclidean').fit_predict(coords_singleaxis)
#     return labels


# def cluster_and_sort_labels(label_positions, valid_labels, postol_px):
#     """
#     Groups objects using 1D DBSCAN in Z and Y, sorts by Z group (asc), Y group (desc), X (asc).
#     cc_px: [X, Y, Z] center-to-center pixel estimates.
#     """
#     # Gather mid-point coordinates
#     mid_zs = np.array([label_positions[k]["mid_z"] for k in valid_labels])
#     mid_ys = np.array([label_positions[k]["mid_y"] for k in valid_labels])
#     mid_xs = np.array([label_positions[k]["mid_x"] for k in valid_labels])

#     # Compute grouping tolerance (half of estimated spacing)
#     z_tolerance_px = postol_px[2]
#     y_tolerance_px = postol_px[1]

#     # Cluster using DBSCAN
#     z_groups = cluster_axis(mid_zs, z_tolerance_px)
#     y_groups = cluster_axis(mid_ys, y_tolerance_px)

#     # Map group -> first mid_z
#     group_to_first_mid_z = {}
#     for group in set(z_groups):
#         idxs = np.where(z_groups == group)[0]
#         # Use the first sample's mid_z
#         first_idx = idxs[0]
#         group_to_first_mid_z[group] = mid_zs[first_idx]

#     # Sort group labels by ascending mid_z of first sample
#     sorted_groups = sorted(group_to_first_mid_z, key=lambda g: group_to_first_mid_z[g])

#     # Build group info: (z_group, y)
#     group_info = {k: (z_groups[idx], label_positions[k]["mid_y"]) for idx, k in enumerate(valid_labels)}

#     # Output: for each group in sorted order, output its members sorted by Y descending
#     sorted_labels = []
#     for z_group in sorted_groups:
#         labels_in_z = [k for idx, k in enumerate(valid_labels) if z_groups[idx] == z_group]
#         labels_in_z_sorted = sorted(labels_in_z, key=lambda k: group_info[k][1], reverse=True)
#         sorted_labels.extend(labels_in_z_sorted)
    
#     return sorted_labels

# import numpy as np




def _bin_axis(vals, n_bins, offset_steps=25):
    """
    Map 1D coords to integer bins [0..n_bins-1] on an evenly spaced grid.
    Works with holes. Deterministic. O(n + offset_steps*n).
    """
    vals = np.asarray(vals, float)
    n = vals.size
    if n_bins <= 1 or n == 0:
        return np.zeros(n, dtype=int), 1.0, (vals.min() if n else 0.0)

    # Robust pitch from quantiles to avoid stretching by holes
    vmin, vmax = np.percentile(vals, [5, 95]) if n > 4 else (vals.min(), vals.max())
    # avoid zero division for single-point or near-constant cases
    p0 = (vmax - vmin) / max(n_bins - 1, 1)
    if p0 <= 0:
        p0 = np.std(vals) or 1.0

    # Search offset within one pitch to align centers to data
    # Centers: off + p0 * k, k=0..n_bins-1
    off0 = np.min(vals)
    offsets = off0 + np.linspace(0, p0, offset_steps, endpoint=False)

    best_off, best_score = None, np.inf
    for off in offsets:
        # project to nearest bin index
        k = np.rint((vals - off) / p0)
        # clip into valid bin range
        k = np.clip(k, 0, n_bins - 1)
        centers = off + k * p0
        # L1 is robust; use mean abs residual
        score = np.mean(np.abs(vals - centers))
        if score < best_score:
            best_score, best_off = score, off

    # Final assignment with best offset
    k = np.rint((vals - best_off) / p0)
    k = np.clip(k, 0, n_bins - 1).astype(int)
    return k, p0, best_off

def create_sort_spec(primary_axis, secondary_axis, tertiary_axis,
                    primary_asc=True, secondary_asc=False, tertiary_asc=True):
    """
    Create a sort specification for object labeling.
    
    Args:
        primary_axis, secondary_axis, tertiary_axis: 'x', 'y', or 'z'
        XXX_asc: True for ascending, False for descending
        
    Returns:
        dict: Sort specification with axes and directions
        
    Examples:
        Front-to-back, bottom-to-top, left-to-right:
            create_sort_spec('z', 'y', 'x', True, True, True)
        Bottom-to-top (rows), left-to-right (columns), front-to-back (layers):
            create_sort_spec('y', 'x', 'z', False, True, True)
    """
    return {
        "order": [primary_axis, secondary_axis, tertiary_axis],
        "ascending": [primary_asc, secondary_asc, tertiary_asc]
    }


def cluster_and_sort_labels_known_bins(label_positions, valid_labels,
                                       bins=(1, 1, 1), sort_spec=None):
    """
    Cluster and sort labels according to bins and sort specification.
    
    Args:
        label_positions: dict of label -> {"mid_x", "mid_y", "mid_z", ...}
        valid_labels: list of label indices to sort
        bins: (nx, ny, nz) number of bins per axis
        sort_spec: dict with "order" (list of 'x','y','z') and "ascending" (list of bool)
                   If None, defaults to Z asc, Y desc, X asc
    
    If a dimension has 1 bin, fall back to raw coordinate for that dimension.
    Handles holes naturally.
    """
    if sort_spec is None:
        sort_spec = create_sort_spec('z', 'y', 'x', 
                                     primary_asc=True, secondary_asc=False, tertiary_asc=True)
    
    xs = np.array([label_positions[k]["mid_x"] for k in valid_labels], float)
    ys = np.array([label_positions[k]["mid_y"] for k in valid_labels], float)
    zs = np.array([label_positions[k]["mid_z"] for k in valid_labels], float)

    nx, ny, nz = bins
    xg, _, _ = _bin_axis(xs, nx)
    yg, _, _ = _bin_axis(ys, ny)
    zg, _, _ = _bin_axis(zs, nz)

    # Map axis letter to binned and raw coords
    axis_to_binned = {'x': xg, 'y': yg, 'z': zg}
    axis_to_raw = {'x': xs, 'y': ys, 'z': zs}
    axis_to_nbins = {'x': nx, 'y': ny, 'z': nz}

    # Build sort keys in reverse order (lexsort priority is right-to-left)
    sort_keys = []
    for axis, is_asc in zip(sort_spec["order"], sort_spec["ascending"]):
        nbins = axis_to_nbins[axis]
        # Use binned if multiple bins, else raw
        key = axis_to_binned[axis] if nbins > 1 else axis_to_raw[axis]
        # Negate for descending
        key = key if is_asc else -key
        sort_keys.append(key)
    
    # lexsort: last tuple element has highest priority
    sort_keys.reverse()
    order = np.lexsort(tuple(sort_keys))
    sorted_labels = [valid_labels[i] for i in order]

    return sorted_labels


def get_valid_labels(img_dip,labelled_img,margin_px,size_hint_px_inf,size_hint_px_sup):
    # Fetch Bounding boxes for all objects at once.
    max_size = img_dip.Sizes()
    bboxes = dip.MeasurementTool.Measure(labelled_img, labelled_img, ["Minimum", "Maximum"])
    num_labels = bboxes.NumberOfObjects()
    mins = bboxes["Minimum"]
    maxs = bboxes["Maximum"]

    # loop through labels once and compute bounding box stats
    valid_labels = []
    label_positions = {}
    
    for k in range(1, num_labels + 1):
        lower = []
        upper = []
        extent = []

        for d in range(3):  # 0 = X, 1 = Y, 2 = Z
            min_d = int(mins[k][d])
            max_d = int(maxs[k][d])
            ext_d = max_d - min_d

            extent.append(ext_d)

            lo = max(0, min_d - margin_px)
            hi = min(max_size[d] - 1, max_d + margin_px + 1)

            lower.append(lo)
            upper.append(hi)

        # Skip if size does not meet requirements
        if any(extent[d] < size_hint_px_inf[d] for d in range(3)):
            logging.info(f"Skipping label {k} due to undersize: {extent[::-1]}  (X, Y, Z), infemum is: {size_hint_px_inf[::-1]}")
            print(f"Skipping label {k} due to undersize: {extent[::-1]}  (X, Y, Z), infemum is: {size_hint_px_inf[::-1]}")
            continue
        if any(extent[d] > size_hint_px_sup[d] for d in range(3)):
            logging.info(f"Skipping label {k} due to oversize: {extent[::-1]} (X, Y, Z), supremum is: {size_hint_px_sup[::-1]}")
            print(f"Skipping label {k} due to oversize: {extent[::-1]} (X, Y, Z), supremum is: {size_hint_px_sup[::-1]}")
            continue
        
        valid_labels.append(k)
        label_positions[k] = {
            "extent": extent,
            "lower": lower,
            "upper": upper,
            "mid_x": (mins[k][2] + maxs[k][2]) / 2,
            "mid_y": (mins[k][1] + maxs[k][1]) / 2,
            "mid_z": (mins[k][0] + maxs[k][0]) / 2 
        }

    return valid_labels, label_positions


def naming_strategy_parsed(i, k, label_positions, labels_in_order=None):
    """
    Default naming strategy: uses parsed labels or falls back to numeric.
    
    Args:
        i: 1-indexed object counter
        k: label index from segmentation
        label_positions: dict with label info
        labels_in_order: list of label names (may be None)
        
    Returns:
        str: base filename (without .nrrd extension)
    """
    lower = label_positions[k]["lower"]
    upper = label_positions[k]["upper"]
    extent = label_positions[k]["extent"]
    z0, y0, x0 = lower
    extent_x, extent_y, extent_z = extent[::-1]  # [X, Y, Z]
    
    if labels_in_order is not None and i - 1 < len(labels_in_order):
        return labels_in_order[i - 1]
    else:
        return f"object_{i:02d}_X{x0}_Y{y0}_Z{z0}_extX{extent_x}_extY{extent_y}_extZ{extent_z}"


def naming_strategy_sequential(i, k, label_positions, **kwargs):
    """
    Simple sequential naming: object_01, object_02, etc.
    
    Args:
        i: 1-indexed object counter
        k: label index from segmentation
        label_positions: dict with label info
        **kwargs: ignored
        
    Returns:
        str: base filename (without .nrrd extension)
    """
    return f"object_{i:02d}"


def naming_strategy_with_coordinates(i, k, label_positions, **kwargs):
    """
    Naming with coordinates: object_01_X100_Y200_Z300.
    
    Args:
        i: 1-indexed object counter
        k: label index from segmentation
        label_positions: dict with label info
        **kwargs: ignored
        
    Returns:
        str: base filename (without .nrrd extension)
    """
    lower = label_positions[k]["lower"]
    z0, y0, x0 = lower
    return f"object_{i:02d}_X{x0}_Y{y0}_Z{z0}"


def extract_and_save_components(
        labelled_img, img_dip, output_path, header, bins=(1,1,1), 
        labels_in_order=None, margin_mm=5, postol_mm=None, 
        size_hint_mm_inf=None, size_hint_mm_sup=None,
        sort_spec=None, naming_func=None):
    
    os.makedirs(output_path, exist_ok=True)

    dx, dy, dz = header["spacings"]
    margin_px = int(margin_mm // dx)

    if size_hint_mm_inf is None:
        size_hint_px_inf = [0, 0, 0]
    else:
        size_hint_px_inf = [int(size_hint_mm_inf[d] // dx) for d in range(3)]
    if size_hint_mm_sup is None:
        size_hint_px_sup = [img_dip.Size(d) for d in range(3)]
    else:
        size_hint_px_sup = [int(size_hint_mm_sup[d] // dx) for d in range(3)]
    

    valid_labels, label_positions = get_valid_labels(img_dip,labelled_img,margin_px,size_hint_px_inf,size_hint_px_sup)


    if naming_func is None:
        naming_func = naming_strategy_parsed
    
    if len(valid_labels) > 0:
        # Binning / Clustering of the objects
        logging.info(f"Clustering {len(valid_labels)} objects...")
        # postol_px = [int(postol_mm[d] // dx) for d in range(3)]
        # final_label_order = cluster_and_sort_labels(label_positions, valid_labels, postol_px)
        final_label_order = cluster_and_sort_labels_known_bins(label_positions, valid_labels, bins, sort_spec=sort_spec)
        
        # Save label order as CSV
        label_order_path = os.path.join(output_path, "label_order.csv")
        with open(label_order_path, "w") as f:
            f.write("Label, X_start, Y_start, Z_start\n")
            for k in final_label_order:
                lower = label_positions[k]["lower"]
                upper = label_positions[k]["upper"]
                extent = label_positions[k]["extent"]
                f.write(f"{k}, {lower[2]}, {lower[1]}, {lower[0]}\n")


        # Crop and save the objects in right order with right names
        for i, k in enumerate(final_label_order, start=1): # k is label index
            lower = label_positions[k]["lower"]
            upper = label_positions[k]["upper"]
            extent = label_positions[k]["extent"]

            # Original: lower = [x, y, z]
            slices = [
                slice(lower[0], upper[0]),  # Z
                slice(lower[1], upper[1]),  # Y
                slice(lower[2], upper[2])   # X
            ]
            # cropped = img_dip[slices]
            cropped = img_dip.At(slices)
            # TODO: also crop the label image, then dilate all labels that are not the current label by 1 pixel and then set those other labels to zero
            label_cropped = labelled_img.At(slices)
            other_labels_mask = (label_cropped != 0) & (label_cropped != k)
            other_labels_mask = dip.Dilation(other_labels_mask, dip.SE(1))
             # set to 0 where other labels are present
            # cropped[other_labels_mask] = 0
            
            # label_cropped = dip.IfThenElse(other_labels_mask, 0, label_cropped)
            # current_label_mask = (label_cropped == k)
            # cropped = dip.IfThenElse(current_label_mask, cropped, 0)

            # Generate filename using naming strategy
            base_name = naming_func(i, k, label_positions, labels_in_order=labels_in_order)
            out_path = os.path.join(output_path, base_name + ".nrrd")

            cropped_header = header.copy()
            cropped_header['sizes'] = np.array(cropped.Sizes())
            cropped_header['enconding'] = 'raw'

            logging.info(f"Saving {out_path}...")
            nrrd.write(out_path, np.asarray(cropped), header=cropped_header, compression_level=0)
            logging.info(f"Saved: {out_path}")
    else:
        logging.warning((f"No valid labels found!"))


###############################################################################################

    
def process_file_single(filename, labels_in_order, bins, t,
                margin_mm, postol_mm, size_hint_mm_inf,
                sort_spec=None, naming_func=None):
    """
    Process a single NRRD file: segment, cluster, and extract objects.
    
    Args:
        filename: Path to NRRD file
        labels_in_order: List of label names or None
        bins: (nx, ny, nz) tuple for binning
        t: Threshold value for segmentation
        margin_mm: Margin around objects in mm
        postol_mm: Position tolerance in mm (deprecated)
        size_hint_mm_inf: Minimum size hint [Z, Y, X] in mm
        sort_spec: Sort specification from create_sort_spec(), or None for default
        naming_func: Naming function or None for default
    """
    output_path = filename.parent / filename.stem
    label_file = filename.parent / (filename.stem + "_labels.nrrd")
    
    setup_logging(output_path, base_name=filename.stem)
    
    logging.info(f"Loading {filename}...")
    data, header = nrrd.read(filename.as_posix())
    img_dip = dip.Image(data)
    dx, dy, dz = header["spacings"]

    labelled_img = segment_objects(img_dip, dx, t)

    # save_label_nrrd(labelled_img, header, os.path.join(output_path, str(label_file))
    
    extract_and_save_components(labelled_img, img_dip, output_path, header, bins=bins,
                                labels_in_order=labels_in_order, margin_mm=margin_mm, postol_mm=postol_mm, 
                                size_hint_mm_inf=size_hint_mm_inf, size_hint_mm_sup=None,
                                sort_spec=sort_spec, naming_func=naming_func)
    

def parse_nrrd_objects(fn_str:str):
    # Pattern: One or more groups like A_01-03
    pattern = r'([A-Z])_(\d{1,3})-(\d{1,3})'
    matches = re.findall(pattern,fn_str)
    labels_in_order = []
    
    for letter, start, end in matches:
        s, e = int(start), int(end)
        for i in range(s, e + 1):
            labels_in_order.append(f"{letter}{i:03d}")

    return labels_in_order


def process_file_multi(filenames, bins, t, margin_mm,
                         postol_mm, size_hint_mm_inf,
                         nr_threads=4, anonymous=False,
                         sort_spec=None, naming_func=None):
    """
    Process multiple NRRD files in sequence (or parallel if uncommented).
    
    Args:
        filenames: List of Path objects
        bins, t, margin_mm, postol_mm, size_hint_mm_inf: See process_file_single()
        nr_threads: Number of threads for parallel processing
        anonymous: If True, use numeric naming instead of parsed labels
        sort_spec: Sorting specification
        naming_func: Naming function
    """
    arglist = []
    
    for fn in filenames:
        labels_in_order = parse_nrrd_objects(fn.name)
        if anonymous:
            for id in range(len(labels_in_order)):
                labels_in_order[id] = f"{id:03d}"
        
        args = (fn, labels_in_order, bins, t, margin_mm,
                postol_mm, size_hint_mm_inf, sort_spec, naming_func)
        arglist.append(args)
        process_file_single(fn, labels_in_order, bins, t, margin_mm,
                    postol_mm, size_hint_mm_inf, 
                    sort_spec=sort_spec, naming_func=naming_func)
    
    # with Pool(processes=nr_threads) as pool:
    #     pool.starmap(process_file_single, arglist) # unpack positional args


def main():
    margin_mm = 5  # mm  

    nrrd_base = Path("/media/Store-HDD/johannes-data/CT-Data/Aalborg-reclaimed")
    # filenames = sorted(list(nrrd_base.glob('*.nrrd')))
    
    filenames = [nrrd_base / "20260309.134740.RB_1.nrrd"]
    nr_threads = 4
    size_hint_mm_inf = [1900,40,40] # Z, Y, X
    postol_mm = [100,15,15]
    bins = (1,1,1) # Z, Y, X
    
    
    anonymous=False
    t = 200

    # Configure sorting and naming strategies
    # Example 1: Default sorting (Z asc, Y desc, X asc) with parsed labels
    sort_spec = create_sort_spec('y', 'x', 'z', 
                                 primary_asc=False, secondary_asc=True, tertiary_asc=True)
    naming_func = naming_strategy_parsed
    
    # Example 2: Left-to-right, bottom-to-top, front-to-back with sequential names
    # sort_spec = create_sort_spec('x', 'y', 'z', 
    #                              primary_asc=True, secondary_asc=False, tertiary_asc=True)
    # naming_func = naming_strategy_sequential
    
    # Example 3: Custom coordinate naming
    # naming_func = naming_strategy_with_coordinates

    process_file_multi(filenames, bins, t, margin_mm,
                         postol_mm, size_hint_mm_inf,
                         nr_threads, anonymous=anonymous,
                         sort_spec=sort_spec, naming_func=naming_func)

if __name__ == "__main__":
    main()

# TODO:
# Chekck order of G 4-9!!
# double check all orders...