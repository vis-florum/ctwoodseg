import os
import re
from pathlib import Path
import nrrd
import diplib as dip
import numpy as np

nrrd_base = Path("data/C")
filename = nrrd_base/"C40-44/C40.nrrd"

data, header = nrrd.read(filename.as_posix())
img_dip = dip.Image(data)

t = 200
# B = data > t
B_solid = dip.FixedThreshold(img_dip, t)
L_solid = dip.Label(B_solid, mode="largest")    # solid label is 1
B_solid = L_solid > 0


# L_air = dip.Label(~B_solid)
L_pores = dip.Label(~B_solid, boundaryCondition=["remove","remove","remove"])   # Remove border-touching objects

# Map the labels to 1 + label_nr
l_pores = dip.ListObjectLabels(L_pores)
lut = np.zeros(len(l_pores)+1, dtype=np.uint32)
lut[1:] = np.array(l_pores) + 1
L_pores = dip.LookupTable(lut).Apply(L_pores)


L_solid_and_pores = L_solid | L_pores  # solid is label 1, pores start at label 2
dip.ListObjectLabels(L_solid_and_pores)

# L_air_ID_surrounding = dip.ListObjectLabels(L_air, region="edges")  # only those regions touching the image border
# L_air_ID_all = dip.ListObjectLabels(L_air)
# L_air_ID_pores = dip.ListObjectLabels(L_pores)


# fill pores
# By graphs???
G_solid_and_pores = dip.RegionAdjacencyGraph(L_solid_and_pores)
MSF_solid_and_pores = dip.MinimumSpanningForest(G_solid_and_pores,[1])

MSF_solid_and_pores.RemoveLargestEdges(MSF_solid_and_pores.NumberOfEdges())

L_solid_and_pores_combi = dip.Relabel(L_solid_and_pores,MSF_solid_and_pores)

dip.Maximum(L_solid_and_pores_combi)[0]
dip.Minimum(L_solid_and_pores_combi)[0]

# By morphology
B_solid_and_pores = L_solid_and_pores > 0


header['type'] = 'uint32'
header['encoding'] = 'raw'
out_array = np.asarray(L_solid_and_pores_combi, dtype=np.uint32)
out_path = filename.stem + "_L.nrrd"
nrrd.write(out_path, out_array, header=header, compression_level=0)




header['type'] = 'int8'
header['encoding'] = 'raw'
out_array = np.asarray(B_solid, dtype=np.int8)
out_path = filename.stem + "_B.nrrd"
nrrd.write(out_path, out_array, header=header, compression_level=0)

D = dip.EuclideanDistanceTransform(B_solid_and_pores)


header['type'] = 'float32'
header['encoding'] = 'raw'
out_array = np.asarray(D, dtype=np.float32)
out_path = filename.stem + "_D.nrrd"
nrrd.write(out_path, out_array, header=header, compression_level=0)
