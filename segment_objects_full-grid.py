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

    # Label connected components
    label_img = dip.Label(mask, minSize=100000)
    return label_img


def save_label_nrrd(label_img, header, out_path):
    """Saves the label image as a NRRD file."""
    label_array = np.asarray(label_img, dtype=np.int8)
    label_header = header.copy()
    label_header['type'] = 'int8'

    # Ensure output directory exists
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    nrrd.write(out_path, label_array, header=label_header, compression_level=1)
    

def cluster_axis(coords, tolerance):
    """Clusters 1D coordinates using DBSCAN, returns group labels for each input."""
    coords = np.array(coords).reshape(-1, 1)
    # DBSCAN eps = tolerance, min_samples=1 means even isolated points get a label
    labels = DBSCAN(eps=tolerance, min_samples=1, metric='euclidean').fit_predict(coords)
    return labels


def cluster_and_sort_labels(label_positions, valid_labels, postol_px):
    """
    Groups objects using 1D DBSCAN in Z and Y, sorts by Z group (asc), Y group (desc), X (asc).
    cc_px: [X, Y, Z] center-to-center pixel estimates.
    """
    # Gather mid-point coordinates
    mid_zs = np.array([label_positions[k]["mid_z"] for k in valid_labels])
    mid_ys = np.array([label_positions[k]["mid_y"] for k in valid_labels])
    # mid_xs = np.array([label_positions[k]["mid_x"] for k in valid_labels])

    # Compute grouping tolerance (half of estimated spacing)
    z_tolerance_px = postol_px[2]
    # y_tolerance_px = postol_px[1]

    # Cluster using DBSCAN
    z_groups = cluster_axis(mid_zs, z_tolerance_px)
    # y_groups = cluster_axis(mid_ys, y_tolerance_px)

    # Map group -> first mid_z
    group_to_first_mid_z = {}
    for group in set(z_groups):
        idxs = np.where(z_groups == group)[0]
        # Use the first sample's mid_z
        first_idx = idxs[0]
        group_to_first_mid_z[group] = mid_zs[first_idx]

    # Sort group labels by ascending mid_z of first sample
    sorted_groups = sorted(group_to_first_mid_z, key=lambda g: group_to_first_mid_z[g])

    # Build group info: (z_group, y)
    group_info = {k: (z_groups[idx], label_positions[k]["mid_y"]) for idx, k in enumerate(valid_labels)}

    # Output: for each group in sorted order, output its members sorted by Y descending
    sorted_labels = []
    for z_group in sorted_groups:
        labels_in_z = [k for idx, k in enumerate(valid_labels) if z_groups[idx] == z_group]
        labels_in_z_sorted = sorted(labels_in_z, key=lambda k: group_info[k][1], reverse=True)
        sorted_labels.extend(labels_in_z_sorted)
    
    return sorted_labels



def extract_and_save_components(label_img, img_dip, output_dir, header, labels_pic=None, margin_mm=5, postol_mm=None, size_hint_mm_inf=None, size_hint_mm_sup=None):
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
    
    
    # Fet Bounding boxes for all objects at once.
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
        # size_hint_dip_inf = size_hint_px_inf[::-1]
        # size_hint_dip_sup = size_hint_px_sup[::-1]
        if any(extent[d] < size_hint_px_inf[d] for d in range(3)):
            logging.info(f"Skipping label {k} due to undersize: {extent[::-1]}  (X, Y, Z), infemum is: {size_hint_px_inf[::-1]}")
            print(f"Skipping label {k} due to undersize: {extent[::-1]}  (X, Y, Z), infemum is: {size_hint_px_inf[::-1]}")
            continue
        # if extent[1] > size_hint_dip[1] * 4 or extent[2] > size_hint_dip[2] * 4:
        #     logging.info(f"Skipping label {k} due to size: {extent[::-1]} (X, Y, Z)")
        #     continue
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
        final_label_order = cluster_and_sort_labels(label_positions, valid_labels, postol_px)
        
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

            # Save using NRRD
            z0, y0, x0 = lower
            extent_x, extent_y, extent_z = extent[::-1]  # [X, Y, Z]
            if labels_pic is None:
                out_path = os.path.join(
                    output_dir,
                    f"object_{i:02d}_X{x0}_Y{y0}_Z{z0}.nrrd"
                )
            elif i-1 < len(labels_pic):
                out_path = os.path.join(
                    output_dir,
                    labels_pic[i-1] + ".nrrd"
                )
            else:
                out_path = os.path.join(
                    output_dir,
                    f"unknown-{i-len(labels_pic):02d}_X{x0}_Y{y0}_Z{z0}_extX{extent_x}_extY{extent_y}_extZ{extent_z}.nrrd"
                )

            cropped_header = header.copy()
            cropped_header['sizes'] = np.array(cropped.Sizes())
            # cropped_header["encoding"] = "raw"
            #
            logging.info(f"Saving {out_path}...")
            nrrd.write(out_path, np.asarray(cropped), header=cropped_header, compression_level=1)
            logging.info(f"Saved: {out_path}")
    else:
        logging.warning((f"No valid labels found!"))


###############################################################################################

def process_general():
    # Input
    margin_mm = 10  # mm
    filenames = []
    image_orders = []
    size_hint_mm_inf = [1500,90,90] # Z, Y, X
    size_hint_mm_sup = None
    postol_mm = [100,100,100]
    
    nrrd_base = ""
    
    # size_hint_mm_inf = None
    # size_hint_mm_sup = None
    # postol_mm = [100,100,15]
    
    # nrrd_base = "./Jan-NDT-CT/Spruce"
    # filename = "Jan_Msc_Spruce.nrrd"
    # prefix = "S"
    
    # nrrd_base = "./Jan-NDT-CT/Oak"
    # filename = "Jan_Msc_Oak.nrrd"
    # prefix = "O"
    
    # nrrd_base = "./Jan-NDT-CT/Beech"
    # filename = "Jan_Msc_Beech.nrrd"
    # prefix = "B"

    # # Naming convention: S1-{size_nom}, S2-{size_nom}, ...
    # image_order = []
    # for size_nom in [50, 40, 30, 20]:
    #     names = [f"{prefix}{i:01d}-{str(size_nom)}" for i in range(1, 9)]
    #     image_order.extend(names)
    # image_order.append("water")
    
    
    # # Blackoak
    # nrrd_base = "./Jan-NDT-CT/Blackoak"
    # filename = "Jan_Msc_Blackoak.nrrd"
    
    # image_order = ["BO1-50", "BO1-40", "BO1-30", "BO1-20", "water"]
    
    
    # # EXTRA
    # nrrd_base = "./Jan-NDT-CT/Extra"
    # filename = "Jan_Msc_Extra.nrrd"
    
    # image_order = ["SE1-50", "SE2-50", "SE1-40", "SE2-40", "OE1-40", "SE1-30", "SE2-30", "OE1-30", "OE1-50", "OE2-50"]
    # names = [f"BE{i:01d}-20" for i in range(1, 21)]
    # image_order.extend(names)
    # image_order.extend(["water"])
    
    
    # # LINN
    # nrrd_base = "./Linn_Reclaimed-Timber"
    # filename = "Linn-Exjobb_3.nrrd"
    
    # image_order = ["inre", "undre", "stora", "waer"]
    
        
    # # SCHNEPPS
    # nrrd_base = "./Schnepps_Knots/pieces"
    # filename = "Doka_Julian_MSc-pieces.nrrd"
    
    # image_order = []
    # names = [f"A{i:01d}" for i in range(1, 7)]
    # image_order.extend(names)
    # names = [f"B{i:01d}" for i in range(1, 5)]
    # image_order.extend(names)
    # names = [f"C{i:01d}" for i in range(1, 6)]
    # image_order.extend(names)
    # names = ["D2","D1","water"]
    # image_order.extend(names)
    
    
    # nrrd_base = "./Schnepps_Knots/B"
    # filename = "Doka_Julian_MSc-B.nrrd"
    
    # image_order = []
    # names = [f"B{i:01d}" for i in range(1, 6)]
    
    
    
    # filenames.append(filename)
    # image_orders.append(image_order)
    
    for i in range(len(filenames)):
        filename = filenames[i]
        image_order = image_orders[i]

        scan_name = os.path.split(filename)[-1].split(".")[0]
        nrrd_file = os.path.join(nrrd_base, filename)
        output_dir = nrrd_base
        labels_pic = image_order
        setup_logging(nrrd_base,base_name=scan_name)


        # Pipeline
        logging.info(f"Loading {nrrd_file}...")
        img_dip, header = load_nrrd_as_dip(nrrd_file)
        dx, dy, dz = header["spacings"]
        label_img = segment_objects(img_dip, dx)
        save_label_nrrd(label_img, header, os.path.join(output_dir, f"{scan_name}_labels.nrrd"))
        extract_and_save_components(label_img, img_dip, output_dir, header, labels_pic=labels_pic, margin_mm=margin_mm, postol_mm=postol_mm, size_hint_mm_inf=size_hint_mm_inf, size_hint_mm_sup=size_hint_mm_sup)
    
    
def process_RAW(nrrd_base,filename,image_order,margin_mm,postol_mm,size_hint_mm_inf,size_hint_mm_sup):
# def process_RAW(args):  # argument unpacking for parallelisation
#     nrrd_base, filename, image_order, margin_mm, postol_mm, size_hint_mm_inf, size_hint_mm_sup = args
    
    scan_name = filename.stem.split(".")[0]
    batch_name = scan_name[0]
    nrrd_file = filename.as_posix()
    output_dir = (Path(nrrd_base) / scan_name).as_posix()
    
    labels_pic = [f"{batch_name}{num}" for num in image_order]
    setup_logging(output_dir,base_name=scan_name)
    
    logging.info(f"Loading {nrrd_file}...")
    img_dip, header = load_nrrd_as_dip(nrrd_file)
    dx, dy, dz = header["spacings"]
    label_img = segment_objects(img_dip, dx)
    save_label_nrrd(label_img, header, os.path.join(output_dir, f"{scan_name}_labels.nrrd"))
    extract_and_save_components(label_img, img_dip, output_dir, header, 
                                labels_pic=labels_pic, margin_mm=margin_mm, postol_mm=postol_mm, 
                                size_hint_mm_inf=size_hint_mm_inf, size_hint_mm_sup=size_hint_mm_sup)
    


def process_RAW_parallel(nrrd_base, nr_threads=8, filenames=None):
    # Input
    margin_mm = 5  # mm  
    size_hint_mm_inf = [700,20,20] # Z, Y, X
    size_hint_mm_sup = None
    postol_mm = [100,30,30]
    
    # Extract prefix letter from base path (e.g., "./K" → "K")
    prefix_letter = Path(nrrd_base).name.strip("./")

    arglist = []
    
    for fn in filenames:
        # Extract the base name (e.g., J_01-02.nrrd → 01-02)
        pattern = rf'{prefix_letter}_(\d{{1,3}})-(\d{{1,3}})\.nrrd'
        match = re.search(pattern, fn.name)
        
        image_order = []
        if match:
            from_to = [int(match.group(1)), int(match.group(2))]
            image_ids = range(from_to[0],from_to[-1]+1)
            image_order = [f"{n:03d}" for n in image_ids]   # 3 leading 0s
        
        args = (nrrd_base,fn,image_order,
                margin_mm,postol_mm,
                size_hint_mm_inf,size_hint_mm_sup)
        arglist.append(args)
    
    with Pool(processes=nr_threads) as pool:
        # pool.map(process_RAW, arglist)
        pool.starmap(process_RAW, arglist) # automatically unpacks each tuple in arglist into positional arguments



def main():
    nr_threads = 6
    nrrd_base = "./N"
    filenames = sorted(list(Path(nrrd_base).glob('*.nrrd')))
    # filenames = [Path(nrrd_base) / "K_19-20.nrrd"]
    process_RAW_parallel(nrrd_base,nr_threads,filenames)

if __name__ == "__main__":
    main()
    