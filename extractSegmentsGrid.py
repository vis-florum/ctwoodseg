import nrrd
import diplib as dip
import numpy as np
from sklearn.cluster import DBSCAN
import os
import re
from pathlib import Path
import logging
from datetime import datetime
import re
from collections import defaultdict
import matplotlib.pyplot as plt
from multiprocessing import Pool, cpu_count


def setup_logging(log_dir, base_name="extract_pipeline"):
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    # timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    # log_path = os.path.join(log_dir, f"{base_name}_{timestamp}.log")
    log_path = os.path.join(log_dir, f"{base_name}.log")

    # Remove all existing handlers
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)

    logging.basicConfig(
        level=logging.DEBUG,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[logging.FileHandler(log_path, mode="a")]
    )

    # Add new session banner to log
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    logging.info("\n\n" + "=" * 60)
    logging.info(f" New logging session started at {timestamp} ")
    logging.info(f"Logging started. Output will be written to {log_path}")
    logging.info("=" * 60 + "\n")
    

def load_nrrd_as_dip(filename):
    data, header = nrrd.read(filename)
    img_dip = dip.Image(data)
    return img_dip, header


def findInterMode(img_dip, rho_min, rho_max):
    '''Find the minimum between two modes of the image histogram'''
    logging.debug("Finding inter modes...")

    # Sample in the middle of the image
    img_dip_slice = img_dip[img_dip.Size(0)//2,:,:].Squeeze()
    mask = dip.RangeThreshold(img_dip_slice, lowerBound=rho_min, upperBound=rho_max)
    
    if not np.any(mask):  # Avoid errors if no pixels in range
        logging.warning("No valid pixels found in range.")
        return None

    # rhos = img_dip[mask]
    rhos = img_dip_slice.At(mask)
    nbins = rho_max - rho_min + 1
    freq, bin_edges = np.histogram(rhos, bins=nbins)
    freq = dip.Gauss(freq, 3)

    histmin = np.argmin(freq)
    t = bin_edges[histmin]

    return t


def segment_objects(img_dip, px_scale, rho_min=100, rho_max=500):
    # t = findInterMode(img_dip, rho_min, rho_max)
    t = 200
    logging.info(f"Intermode threshold: {t:.2f}")

    mask = dip.FixedThreshold(img_dip, t)
    # se_diameter = 1.5 / px_scale
    se_diameter = 3 / px_scale

    # Morphological closing to reconnect fragments
    # mask = dip.Closing(mask, dip.SE(se_diameter))  # use spherical SE

    # Remove small artifacts
    mask = dip.Opening(mask, dip.SE(se_diameter))
    
    # For screws and metal artefacts:
    # mask = dip.FillHoles(mask)
    # mask = dip.Erosion(mask)
    # mask = dip.Erosion(mask)
    # mask = dip.Erosion(mask)
    # mask = dip.Dilation(mask)
    # mask = dip.Dilation(mask)
    # mask = dip.Dilation(mask)
    
    # quickly save one slice of the mask as 8 bit binary nrrd
    # idx_z = mask.Size(0)//2
    # idx_z = mask.Size(0) - 150
    # nrrd.write("debug_mask_slice.nrrd", np.asarray(mask[idx_z-50:idx_z+50,:,:].Squeeze(), dtype=np.uint8)*255, header={'type': 'uint8', 'encoding': 'raw'})
    

    # Label connected components
    label_img = dip.Label(mask, minSize=100000)
    #nrrd.write("debug_labels_slice.nrrd", np.asarray(label_img[idx_z-50:idx_z+50,:,:].Squeeze(), dtype=np.uint8), header={'type': 'uint8', 'encoding': 'raw'})
    
    return label_img


def save_label_nrrd(label_img, header, out_path):
    """Saves the label image as a NRRD file."""
    label_array = np.asarray(label_img, dtype=np.int8)
    label_header = header.copy()
    label_header['type'] = 'int8'

    # Ensure output directory exists
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    nrrd.write(out_path, label_array, header=label_header, compression_level=1)
    


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

def cluster_and_sort_labels_known_bins(label_positions, valid_labels,
                                       bins=(1, 1, 1)):
    """
    bins=(nx, ny, nz). Sort by: Z bin asc, then Y bin desc, then X asc.
    If a dimension has 1 bin, fall back to raw coordinate for that dimension as tiebreak.
    Handles holes naturally.
    """
    xs = np.array([label_positions[k]["mid_x"] for k in valid_labels], float)
    ys = np.array([label_positions[k]["mid_y"] for k in valid_labels], float)
    zs = np.array([label_positions[k]["mid_z"] for k in valid_labels], float)

    nx, ny, nz = bins
    xg, _, _ = _bin_axis(xs, nx)
    yg, _, _ = _bin_axis(ys, ny)
    zg, _, _ = _bin_axis(zs, nz)

    # Primary sort keys: Z bin asc, then Y bin desc, then X asc (raw)
    # If nz==1 use raw Z asc to avoid randomization by tiny drift; similarly for ny
    zkey = zg if nz > 1 else zs
    ykey = -yg if ny > 1 else -ys   # descending
    xkey = xs if nx > 1 else xs

    order = np.lexsort((xkey, ykey, zkey))  # last key has highest priority
    sorted_labels = [valid_labels[i] for i in order]

    return sorted_labels




def extract_and_save_components(label_img, img_dip, output_dir, header, bins=(1,1,1), labels_in_order=None, margin_mm=5, postol_mm=None, size_hint_mm_inf=None, size_hint_mm_sup=None):
    os.makedirs(output_dir, exist_ok=True)

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
    
    
    # Fetch Bounding boxes for all objects at once.
    max_size = img_dip.Sizes()
    bboxes = dip.MeasurementTool.Measure(label_img, label_img, ["Minimum", "Maximum"])
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

    if len(valid_labels) > 0:
        # Binning / Clustering of the objects
        logging.info(f"Clustering {len(valid_labels)} objects...")
        postol_px = [int(postol_mm[d] // dx) for d in range(3)]
        final_label_order = cluster_and_sort_labels_known_bins(label_positions, valid_labels, bins)
        
        # Save label order as CSV
        label_order_path = os.path.join(output_dir, "label_order.csv")
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
            label_cropped = label_img.At(slices)
            other_labels_mask = (label_cropped != 0) & (label_cropped != k)
            other_labels_mask = dip.Dilation(other_labels_mask, dip.SE(1))
             # set to 0 where other labels are present
            cropped[other_labels_mask] = 0
            
            # label_cropped = dip.IfThenElse(other_labels_mask, 0, label_cropped)
            # current_label_mask = (label_cropped == k)
            # cropped = dip.IfThenElse(current_label_mask, cropped, 0)

            # Save using NRRD
            z0, y0, x0 = lower
            extent_x, extent_y, extent_z = extent[::-1]  # [X, Y, Z]
            if labels_in_order is None:
                out_path = os.path.join(
                    output_dir,
                    f"object_{i:02d}_X{x0}_Y{y0}_Z{z0}.nrrd"
                )
            elif i-1 < len(labels_in_order):
                out_path = os.path.join(
                    output_dir,
                    labels_in_order[i-1] + ".nrrd"
                )
            else:
                out_path = os.path.join(
                    output_dir,
                    f"unknown-{i-len(labels_in_order):02d}_X{x0}_Y{y0}_Z{z0}_extX{extent_x}_extY{extent_y}_extZ{extent_z}.nrrd"
                )

            cropped_header = header.copy()
            cropped_header['sizes'] = np.array(cropped.Sizes())

            logging.info(f"Saving {out_path}...")
            nrrd.write(out_path, np.asarray(cropped), header=cropped_header, compression_level=1)
            logging.info(f"Saved: {out_path}")
    else:
        logging.warning((f"No valid labels found!"))


###############################################################################################

def process_RAW(nrrd_base,filename,labels_in_order,margin_mm,bins,postol_mm,size_hint_mm_inf,size_hint_mm_sup):   
    scan_name = filename.stem.split(".")[0]
    nrrd_file = filename.as_posix()
    output_dir = (Path(nrrd_base) / scan_name).as_posix()
    
    setup_logging(output_dir,base_name=scan_name)
    
    logging.info(f"Loading {nrrd_file}...")
    img_dip, header = load_nrrd_as_dip(nrrd_file)
    dx, dy, dz = header["spacings"]
    label_img = segment_objects(img_dip, dx)
    save_label_nrrd(label_img, header, os.path.join(output_dir, f"{scan_name}_labels.nrrd"))
    extract_and_save_components(label_img, img_dip, output_dir, header, bins=bins,
                                labels_in_order=labels_in_order, margin_mm=margin_mm, postol_mm=postol_mm, 
                                size_hint_mm_inf=size_hint_mm_inf, size_hint_mm_sup=size_hint_mm_sup)
    

def parse_nrrd_objects(fn):
    # Pattern: One or more groups like A_01-03
    pattern = r'([A-Z])_(\d{1,3})-(\d{1,3})'
    matches = re.findall(pattern, fn.name)
    labels_in_order = []
    
    for letter, start, end in matches:
        s, e = int(start), int(end)
        for i in range(s, e + 1):
            labels_in_order.append(f"{letter}{i:03d}")

    return labels_in_order


def process_RAW_parallel(nrrd_base, nr_threads=8, filenames=None, anonymous=False):
    # Input
    margin_mm = 20  # mm  
    size_hint_mm_inf = [700,35,35] # Z, Y, X
    size_hint_mm_sup = None
    postol_mm = [100,50,30]
    bins = (1,1,1) # Z, Y, X
    
    arglist = []
    
    for fn in filenames:
        labels_in_order = parse_nrrd_objects(fn)
        if anonymous:
            for id in range(len(labels_in_order)):
                labels_in_order[id] = f"{id:03d}"
        
        args = (nrrd_base,fn,labels_in_order,
                margin_mm,bins,postol_mm,
                size_hint_mm_inf,size_hint_mm_sup)
        arglist.append(args)
    
    with Pool(processes=nr_threads) as pool:
        pool.starmap(process_RAW, arglist) # automatically unpacks each tuple in arglist into positional arguments



def main():
    nr_threads = 4
    nrrd_base = "./G"
    # filenames = sorted(list(Path(nrrd_base).glob('*.nrrd')))
    filenames = [Path(nrrd_base) / "G_04-06.nrrd"]
    anonymous=False
    process_RAW_parallel(nrrd_base,nr_threads,filenames,anonymous=anonymous)

if __name__ == "__main__":
    main()
