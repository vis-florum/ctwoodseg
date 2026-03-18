#!/usr/bin/python3.10

# Created by: Johannes Huber
# Date: 2022

# Python Dependencies:
# - numpy
# - tifffile
# - nrrd
# - xmltodict

# Description:
# This script is used to unpack a reconstructed image from a TIFF file and save it as a NRRD file.
# The metadata is saved as a separate XML file, with the same name as the NRRD file.

# Usage: 
# python3 mitoUnpackRecon.py /path/to/image.tiff
# python3 mitoUnpackRecon.py /path/to/image.tiff --compress
# or for many images:
# python3 mitoUnpackRecon.py /path/to/image1.tiff /path/to/image2.tiff /path/to/image3.tiff --compress

import argparse
import os
import nrrd
import tifffile
import xmltodict
import numpy as np

def load_tiff_recon(filename):
    """
    Load TIFF image and extract image data and metadata.
    """
    with tifffile.TiffFile(filename) as fid:
        imgs = fid.series[0].asarray()
        # trajec = fid.series[1].asarray()  # float64!
        mitometa = fid.pages[-1].asarray()[0]
        no_pages = len(fid.pages)
        flat_dims = fid.pages[0].tags["PageNumber"].value

    return imgs, mitometa, no_pages, flat_dims

def parse_metadata(mitometa):
    """
    Parse metadata to extract voxel size.
    """
    if len(mitometa) > 12:
        is_sino = False
        mitometa[mitometa == 32] = 95   # replace space with underscore
        mitometa = np.array([i for i in mitometa if i not in [91, 93]])   # remove brackets from XML element names
        metadict = xmltodict.parse(mitometa)
        if list(metadict.keys())[0] == "logInfo":
            volumedict = metadict["logInfo"]["dimVox"]
            voxel_size = [float(volumedict[key]) for key in ["dimVox0", "dimVox1", "dimVox2"]]
        elif list(metadict.keys())[0] == "CT":
            volumedict = metadict["CT"]["Volume"]
            voxel_size = [float(volumedict[key]) for key in ["Voxel_Dim_X", "Voxel_Dim_Y", "Voxel_Dim_Z"]]
    else:
        print("No scanning data found, it could be a sinogram!")
        voxel_size = None

    return voxel_size

def reshape_image(imgs, no_pages, flat_dims):
    """
    Reshape image data to I, J, K dimensions.
    """
    if no_pages > 2:
        K, I, J = imgs.shape
    else:
        J = imgs.shape[1]
        I, K = flat_dims
        imgs = imgs.reshape(K, I, J)

    imgs = np.transpose(imgs, (1, 2, 0))
    return imgs

def save_metadata(metadata, filepath):
    """
    Save metadata to a file.
    """
    with open(filepath, "w") as f:
        for i in metadata:
            f.write(chr(i))

def save_recon(imgs, voxel_size, filepath, compress=False):
    """
    Save image data as NRRD file.
    """
    comp_level = 0
    encoding = "raw"
    if compress:
        comp_level = 6
        encoding = "gzip"

    nrrd.write(
        filepath,
        imgs,
        detached_header=False,
        header={
            "type": "uint16",
            "encoding": encoding,
            "spacings": voxel_size,
            "units": ["mm", "mm", "mm"],
        },
        compression_level=comp_level,
    )


def main(image_filepath,compress=False):
    """
    Main function to process the image file.
    """
    if not os.path.isfile(image_filepath):
        raise Exception("File not found!")

    # rootpath, filename = os.path.split(image_filepath)
    fullname = os.path.splitext(image_filepath)[0]

    imgs, mitometa, no_pages, flat_dims = load_tiff_recon(image_filepath)
    voxel_size = parse_metadata(mitometa)
    imgs = reshape_image(imgs, no_pages, flat_dims)

    save_metadata(mitometa, fullname + "_ScanSettings.xml")
    save_recon(imgs, voxel_size, fullname + ".nrrd",compress=compress)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process a TIFF image file.")
    # parser.add_argument("image_file", type=str, help="Path to the image file.")
    # allow multiple files
    parser.add_argument("image_file", type=str, nargs="+", help="Path to the image file.")
    parser.add_argument("--compress", action="store_true", help="Enable gzip compression for the output file.")

    args = parser.parse_args()
    
    # run the main function for each image file
    for image_file in args.image_file:
        main(image_file,compress=args.compress)
