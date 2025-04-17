import nrrd
import diplib as dip
import numpy as np
import os
import logging
from datetime import datetime
import re
from collections import defaultdict
import matplotlib.pyplot as plt

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
    t = findInterMode(img_dip, rho_min, rho_max)
    logging.info(f"Intermode threshold: {t:.2f}")

    mask = dip.FixedThreshold(img_dip, t)
    se_diameter = 1.5 / px_scale

    # Morphological closing to reconnect fragments
    # mask = dip.Closing(mask, dip.SE(se_diameter))  # use spherical SE

    # Remove small artifacts
    mask = dip.Opening(mask, dip.SE(se_diameter))

    # Label connected components
    label_img = dip.Label(mask, minSize=100000)
    return label_img



def extract_and_save_components(label_img, original_img, output_dir, header, labels_pic=None, margin_mm=5, size_hint_mm=[1,1,1]):
    os.makedirs(output_dir, exist_ok=True)

    dx, dy, dz = header["spacings"]
    margin_px = int(margin_mm // dx)
    size_hint_px = [int(size_hint_mm[d] // dx) for d in range(3)]
    max_size = original_img.Sizes()

    # Fet Bounding boxes for all objects at once:
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
        size_hint_dip = size_hint_px[::-1]
        if any(extent[d] < size_hint_dip[d] for d in range(3)):
            logging.info(f"Skipping label {k} due to size: {extent[::-1]}  (X, Y, Z)")
            continue
        if extent[1] > size_hint_dip[1] * 4 or extent[2] > size_hint_dip[2] * 4:
            logging.info(f"Skipping label {k} due to size: {extent[::-1]} (X, Y, Z)")
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

    # Step 1: bin labels into rows by X
    row_tolerance_mm = size_hint_mm[0]
    row_tolerance_px = int(row_tolerance_mm // dx)
    rows_dict = defaultdict(list)

    # Grouping by row bin (rounded X / tolerance)
    x_baseline = min(label_positions[k]["mid_x"] for k in valid_labels)

    for k in valid_labels:
        x = label_positions[k]["mid_x"]
        row_index = round((x - x_baseline) / row_tolerance_px)
        rows_dict[row_index].append(k)


    # Step 2: sort rows by X (bottom to top), and each row by Y (right to left)
    sorted_row_indices = sorted(rows_dict.keys())  # bottom to top

    final_label_order = []
    for row_idx in sorted_row_indices:
        row = rows_dict[row_idx]
        # Sort this row right to left (OLOFs Photo)
        row_sorted = sorted(row, key=lambda k: label_positions[k]["mid_y"], reverse=True)
        final_label_order.extend(row_sorted)

    # ix = 0  # true running index of label

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
        # cropped = original_img[slices]
        cropped = original_img.At(slices)

        # Save using NRRD
        if labels_pic is None:
            z0, y0, x0 = lower
            out_path = os.path.join(output_dir, f"object_{i:02d}_X{x0}_Y{y0}_Z{z0}.nrrd")
        else:
            out_path = os.path.join(output_dir, labels_pic[i-1] + ".nrrd")
        
        cropped_header = header.copy()
        cropped_header['sizes'] = np.array(cropped.Sizes())
        # cropped_header["encoding"] = "raw"
        #
        logging.info(f"Saving {out_path}...")
        nrrd.write(out_path, np.asarray(cropped), header=cropped_header, compression_level=2)
        logging.info(f"Saved: {out_path}")    


def main():
    # Input
    margin_mm = 10  # mm
    filenames = []
    image_orders = []

    # nrrd_base = "/media/johhub/Speicher1/CT-Data/Madeira"
    # size_hint_mm = [100,100,900]
    # filename = "20211013.163715.Madeira_pole3.nrrd"
    # image_order = ["Pole_3"]
    # filenames.append(filename)
    # image_orders.append(image_order)
    
    
    # nrrd_base = "Linn"

    # size_hint_mm = [25,43,900]
    # filename = "Linn-exjobb_1.nrrd"
    # image_order = ["1_Foder_H", "6_Taklist" ,"2_Ribb"]
    # filenames.append(filename)
    # image_orders.append(image_order)

    # size_hint_mm = [22,86,550]
    # filename = "Linn-Exjobb_2.nrrd"
    # image_order = ["5_Fasad_Planka", "3_Foder_V" ,"1_Trappracke"]
    # filenames.append(filename)
    # image_orders.append(image_order)



    ###
    # nrrd_base="RAW/A"
    # size_hint_mm=[45,100,2000] # mm
    
    # filename = "20250304.121842.A47_A46.nrrd"
    # image_order = [int(num) for num in re.findall(r'\d+', filename.split(".")[2])]

    # all nrrd files in RAW/A
    # nrrds = os.listdir(nrrd_base)
    # nrrds = [f for f in nrrds if f.endswith('.nrrd')]
    # for nrrd_file in nrrds:
    #     image_order = [int(num) for num in re.findall(r'\d+', nrrd_file.split(".")[2])]
    #     filenames.append(nrrd_file)
    #     image_orders.append(image_order)


    ###
    # nrrd_base="RAW/B"
    # size_hint_mm=[40,40,400] # mm
    
    # filename = "20250225.153927.B1-10.nrrd"
    # image_order = [1,6,8,5,2,3,4,7,9,10]

    # filename = "20250226.084612.B11-20.nrrd"
    # image_order = range(11,21)
    # filenames.append(filename)
    # image_orders.append(image_order)
    
    # filename = "20250226.174754.B21-30.nrrd"
    # image_order = [22,23,24,25,21,26,27,28,29,30]
    # filenames.append(filename)
    # image_orders.append(image_order)
    
    # filename = "20250227.081318.B31-40.nrrd"
    # image_order = range(31,41)
    # filenames.append(filename)
    # image_orders.append(image_order)
    
    # filename = "20250227.083550.B41-50.nrrd"
    # image_order = [41,43,44,45,42,46,47,48,49,50]
    # filenames.append(filename)
    # image_orders.append(image_order)
    
    # filename = "20250227.090004.B51-60.nrrd"
    # image_order = range(51,61)
    # filenames.append(filename)
    # image_orders.append(image_order)
    
    # filename = "20250227.095042.B61-70.nrrd"
    # image_order = [62,63,64,68,61,65,66,67,69,70]
    # filenames.append(filename)
    # image_orders.append(image_order)
    
    # filename = "20250227.101115.B71-80.nrrd"
    # image_order = [72,73,76,78,71,74,75,77,79,80]
    # filenames.append(filename)
    # image_orders.append(image_order)
    
    # filename = "20250227.102943.B81-90.nrrd"
    # image_order = range(81,91)
    # filenames.append(filename)
    # image_orders.append(image_order)
    
    # filename = "20250227.104940.B91-100.nrrd"
    # image_order = range(91,101)
    # filenames.append(filename)
    # image_orders.append(image_order)
    
    # filename = "20250227.111209.B101-110.nrrd"
    # image_order = range(101,111)
    # filenames.append(filename)
    # image_orders.append(image_order)
    
    # filename = "20250227.114125.B111-112.nrrd"
    # image_order = [111,112]
    # filenames.append(filename)
    # image_orders.append(image_order)
    

    ###
    # nrrd_base="RAW/C"
    # size_hint_mm=[40,60,400]   # mm
    
    
    # filename = "20250225.090929.C1-4.nrrd"
    # image_order = [1,2,3,4]
    # filenames.append(filename)
    # image_orders.append(image_order)
    
    # filename = "20250225.095401.C5-9.nrrd"
    # image_order = [5,6,7,8,9]
    # filenames.append(filename)
    # image_orders.append(image_order)
    
    # filename = "20250225.101451.C10-14.nrrd"
    # image_order = [10,11,12,13,14]
    # filenames.append(filename)
    # image_orders.append(image_order)
    
    # filename = "20250225.103850.C15-19.nrrd"
    # image_order = [16,17,18,19,15]
    # filenames.append(filename)
    # image_orders.append(image_order)
    
    # filename = "20250225.105828.C20-24.nrrd"
    # image_order = [20,21,22,23,24]
    # filenames.append(filename)
    # image_orders.append(image_order)
    
    # filename = "20250225.115558.C25-29.nrrd"
    # image_order = [26,27,28,29] + [25]  # from bottom left to top right
    
    # filename = "20250225.121610.C30-34.nrrd"
    # image_order = [30,32,33,34,31]
    # filenames.append(filename)
    # image_orders.append(image_order)
    
    # filename = "20250225.123946.C35-39.nrrd"
    # image_order = [36,37,38,39,35]
    # filenames.append(filename)
    # image_orders.append(image_order)
    
    # filename = "20250225.130313.C40-44.nrrd"
    # image_order = [40,41,42,43,44]
    # filenames.append(filename)
    # image_orders.append(image_order)
    
    # filename = "20250225.143734.C45-49.nrrd"
    # image_order = [49,48,45,46,47]
    # filenames.append(filename)
    # image_orders.append(image_order)
    
    # filename = "20250225.134431.C50-54.nrrd"
    # image_order = [50,52,51,53,54]
    # filenames.append(filename)
    # image_orders.append(image_order)
    
    # filename = "20250225.145334.C55-56.nrrd"
    # image_order = [55,56]
    # filenames.append(filename)
    # image_orders.append(image_order)


    ###
    # nrrd_base="RAW/D"
    # size_hint_mm=[40,110,700] # mm

    # filename = "20250227.134135.D_01-17.nrrd"
    # image_order = [10,13,1,3,17]
    
    # filename = "20250227.145538.D_02-20.nrrd"
    # image_order = [9,2,16,14,20]
    # filenames.append(filename)
    # image_orders.append(image_order)

    # filename = "20250227.160626.D_04-19.nrrd"
    # image_order = [4,19,8,7,11]
    # filenames.append(filename)
    # image_orders.append(image_order)

    # filename = "20250227.171729.D_05-18.nrrd"
    # image_order = [6,5,15,18,12]
    # filenames.append(filename)
    # image_orders.append(image_order)

    # filename = "20250228.100635.D_21-31.nrrd"
    # image_order = [26,31,21,22,24]
    # filenames.append(filename)
    # image_orders.append(image_order)

    # filename = "20250228.093413.D_23-35.nrrd"
    # image_order = [35,27,25,23,32]
    # filenames.append(filename)
    # image_orders.append(image_order)

    # filename = "20250228.081953.D_28-34.nrrd"
    # image_order = [33,34,29,30,28]
    # filenames.append(filename)
    # image_orders.append(image_order)

    for i in range(len(filenames)):
        filename = filenames[i]
        image_order = image_orders[i]

        scan_name = filename.split(".")[2]
        batch_name = scan_name[0]
        nrrd_file = os.path.join(nrrd_base, filename)
        output_dir = os.path.join(nrrd_base, scan_name)
        labels_pic = [f"{batch_name}{num}" for num in image_order]
        setup_logging(nrrd_base,base_name=scan_name)

        # Pipeline
        logging.info(f"Loading {nrrd_file}...")
        img_dip, header = load_nrrd_as_dip(nrrd_file)
        dx, dy, dz = header["spacings"]
        label_img = segment_objects(img_dip, dx)
        extract_and_save_components(label_img, img_dip, output_dir, header, labels_pic=labels_pic, margin_mm=margin_mm, size_hint_mm=size_hint_mm)
    


if __name__ == "__main__":
    main()