#!/bin/bash

# Usage: ./collect_hdf.sh /full/or/relative/path/to/out

# Exit on unset vars or command failure
set -eu

# Check if argument was given
if [ $# -ne 1 ]; then
    echo "Usage: $0 /path/to/out_directory"
    exit 1
fi

OUT_DIR="$1"
HDF_DIR="$OUT_DIR/HDF"

# Create HDF directory if it doesn't exist
mkdir -p "$HDF_DIR"

# Loop through all D*/ subdirectories in OUT_DIR
for dir in "$OUT_DIR"/B*/; do
    # Skip if nothing matches
    [ -d "$dir" ] || continue

    dirname=$(basename "$dir")
    hdf_file="$dir/${dirname}.h5"

    if [ -f "$hdf_file" ]; then
        echo "Copying $hdf_file to $HDF_DIR/"
        cp "$hdf_file" "$HDF_DIR/"
    else
        echo "No file $hdf_file found."
    fi
done
