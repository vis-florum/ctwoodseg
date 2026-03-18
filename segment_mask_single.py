from ctwoodgrad import segment_wood_slicewise, threshold_slicewise_MT, findInterMode
import argparse
import pathlib
import nrrd
import matplotlib.pyplot as plt
import time

import numpy as np
import diplib as dip


def expand_inputs(inputs: list[str]) -> list[pathlib.Path]:
    paths = []
    for s in inputs:
        p = pathlib.Path(s)
        if any(ch in s for ch in "*?["):
            paths.extend(p.parent.glob(p.name))
        else:
            paths.append(p)
    return sorted(set(paths))


def process_file(path: pathlib.Path, mask: bool = True, outdir: pathlib.Path | None = None) -> pathlib.Path:
    img, header = nrrd.read(str(path))
    
    outdir = outdir if outdir is not None else path.parent

    mask, _ = segment_wood_slicewise(img)   # segment biggest object and fill cavities
    
    if mask:
        outpath = outdir / f"{path.stem}-mask.nrrd"
        
        mask_uint8 = mask.astype('uint8')
        header = header.copy()
        header["type"] = "uint8"
        header["encoding"] = "gzip"
        nrrd.write(str(outpath), mask_uint8, header, compression_level=3)  # is fastest among the first 3!
    else:
        outpath = outdir / "masked" / f"{path.stem}.nrrd"
        outpath.parent.mkdir(parents=True, exist_ok=True)
        
        img[~mask] = 0  # mask object
        
        header = header.copy()
        header["encoding"] = "raw"
        nrrd.write(str(outpath), img, header, compression_level=0)

    return outpath


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Apply slice-wise wood segmentation to NRRD files."
    )
    parser.add_argument(
        "files",
        nargs="+",
        help="One or more input files or glob patterns, e.g. ./test/*.nrrd",
    )
    parser.add_argument(
        "-o",
        "--outdir",
        type=pathlib.Path,
        help="Output directory. Defaults to the input file directory.",
    )

    args = parser.parse_args()

    files = expand_inputs(args.files)
    if not files:
        parser.error("no input files found")

    if args.outdir is not None:
        args.outdir.mkdir(parents=True, exist_ok=True)

    for path in files:
        process_file(path, args.outdir)


if __name__ == "__main__":
    main()