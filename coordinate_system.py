import os
import re
from pathlib import Path
import nrrd
import diplib as dip
import numpy as np

nrrd_base = Path("RAW/Batch1/C")
filename = nrrd_base/"C40-44/C40.nrrd"

data, header = nrrd.read(filename.as_posix())
img_dip = dip.Image(data)

t = 200
# B = data > t
B_solid = dip.FixedThreshold(img_dip, t)
L_object = dip.Label(B_solid, mode="largest")    # solid label is 1
B_object = L_object > 0
dip.ListObjectLabels(L_object)
assert len(dip.ListObjectLabels(L_object)) == 1


# L_air = dip.Label(~B_object)
L_internalobjects = dip.Label(~B_object, boundaryCondition=["remove","remove","remove"])   # Remove border-touching objects

# Map the labels to 1 + label_nr (shift all by 1)
labels_internalobjects = dip.ListObjectLabels(L_internalobjects)    # BG 0 i not included
lut = np.zeros(len(labels_internalobjects) + 1, dtype=np.uint32)
lut[1:] = np.array(labels_internalobjects) + 1
L_internalobjects = dip.LookupTable(lut).Apply(L_internalobjects)
assert len(dip.ListObjectLabels(L_internalobjects)) == len(lut) - 1
assert np.min(dip.ListObjectLabels(L_internalobjects)) == 2 

L_object_and_internals = L_object | L_internalobjects  # solid is label 1, pores start at label 2

# Visual check
# outimg = dip.Image.Squeeze(L_object_and_internals[L_object_and_internals.Size(0)//2,:,:])
# outimg = outimg >= 2    # Should be only internal holes etc, not touching image border
# outimg.Show()


# L_air_ID_surrounding = dip.ListObjectLabels(L_air, region="edges")  # only those regions touching the image border
# L_air_ID_all = dip.ListObjectLabels(L_air)
# L_air_ID_pores = dip.ListObjectLabels(L_internalobjects)


# Fill pores using Graph Representation of the regions
# If edge weights are large, then the relative areag connecting the regions is small
G_solid_and_internals = dip.RegionAdjacencyGraph(L_object_and_internals, mode="touching")    # region with ID 0 (the background) is not included in the graph
MSF_solid_and_internals = G_solid_and_internals.MinimumSpanningForest([1])  # Only the regions touching label 1

# MSF_solid_and_internals.RemoveLargestEdges(5)

L_closed = dip.Relabel(L_object_and_internals,MSF_solid_and_internals)
dip.ListObjectLabels(L_closed)

bboxes = dip.MeasurementTool.Measure(L_closed, L_closed, ["Minimum", "Maximum"])
mins = bboxes["Minimum"]
maxs = bboxes["Maximum"]
z_mids = [int(mins[k][0] + maxs[k][0]) // 2 for k in mins.keys()]
 

# Visual check
outimg = dip.Image.Squeeze(L_closed[L_closed.Size(0)//2,:,:])
outimg = outimg == 1    # Should be only internal holes etc, not touching image border
outimg.Show()

k = 0
outimg = dip.Image.Squeeze(L_closed[z_mids[k],:,:])
outimg = dip.Image.Squeeze(img_dip[z_mids[k],:,:])
outimg.Show()


B_filled = dip.FillHoles(B_object)
B_closed = L_closed > 0
# check if identical
T = B_filled == B_closed
dip.Minimum(T)[0]
dip.Maximum(T)

LT = dip.Label(~T)
bboxes = dip.MeasurementTool.Measure(LT, LT, ["Minimum", "Maximum"])
mins = bboxes["Minimum"]
maxs = bboxes["Maximum"]
z_mids = [int(mins[k][0] + maxs[k][0]) // 2 for k in mins.keys()]
 
k = 1
outimg = dip.Image.Squeeze(LT[z_mids[k],:,:])
outimg.Show()
outimg = dip.Image.Squeeze(B_filled[z_mids[k],:,:])
outimg.Show()
outimg = dip.Image.Squeeze(B_closed[z_mids[k],:,:])
outimg.Show()
outimg = dip.Image.Squeeze(img_dip[z_mids[k],:,:])
outimg.Show()


header['type'] = 'uint32'
header['encoding'] = 'raw'
out_array = np.asarray(L_closed, dtype=np.uint32)
out_path = filename.stem + "_L.nrrd"
nrrd.write(out_path, out_array, header=header, compression_level=0)

