
# def showslice(im,s=1):
#     if im.Dimensionality() == 3:
#         im[s,:,:].Squeeze().Show(colormap="inferno")
#     elif im.Dimensionality() == 2:
#         im.Show(colormap="inferno")
    
# filename = "/media/Store-HDD/johannes-data/CT-Data/Aalborg-reclaimed/20260309.134740.RB_1/RB-2026-18-CT.nrrd"
# img, header = nrrd.read(filename)

# imgd = dip.Image(img)
# M0 = imgd > 100

# s1 = 10031
# img_s1 = imgd[s1,:,:].Squeeze()
# t = findInterMode(img_s1,100,350)
# showslice(img_s1>t)
# showslice(img_s1>100)
# showslice(img_s1)
# img_s11 = dip.Copy(img_s1)
# img_s11[img_s11>900] = 900
# showslice(img_s11)

# m1 = dip.HysteresisThreshold(img_s1,lowThreshold=100,highThreshold=200)
# showslice(m1)

# m = (900>img_s1) & m1
# h1 = dip.Histogram(img_s1, mask=m)
# m = (900>img_s1) & (img_s1>100)
# h2 = dip.Histogram(img_s1, mask=m)

# c = h1.BinCenters()
# f = [h1[b] for b in range(0,len(c))]
# plt.plot(c,f, label="hysteres")
# c = h2.BinCenters()
# f = [h2[b] for b in range(0,len(c))]
# plt.plot(c,f, label=">100")
# plt.legend()
# plt.show()
# plt.close()

# diff = m1 & ~(img_s1>100)
# diff.Show() 

# (img_s1>100).Show()
# m1.Show()

# s2 = 700
# img_s2 = imgd[s2,:,:].Squeeze()
# showslice(img_s2)

# meth = "background"
# m1,t1 = dip.Threshold(img_s1,method=meth)
# m2,t2 = dip.Threshold(img_s2,method=meth)
# t1
# t2
# showslice(m1)
# showslice(m2)

# m1 = dip.HysteresisThreshold(img_s1,lowThreshold=120,highThreshold=200)
# m2 = dip.HysteresisThreshold(img_s2,lowThreshold=172,highThreshold=200)
# showslice(img_s1)
# showslice(img_s2)
# findInterMode(img_s1,100,350)
# findInterMode(img_s2,100,350)
# findInterMode(img_s2,100,500)
# showslice(m1)
# showslice(m2)

# t = []
# dz = img.shape[2]
# for s in np.arange(img.shape[2]-dz,img.shape[2]):
#     t.append(findInterMode(img[:,:,s],100,350))

# flat = np.mean(img[:,:,-dz::],axis=1)
# plt.plot(t,color="red")
# plt.imshow(flat)
# plt.show()
# plt.close()


################## Hists
def get_hist(img_np, lower=100, upper=1500):
    mask = (img_np >= lower) & (img_np <= upper)
    if not np.any(mask):  # Avoid errors if no pixels in range
        return None

    rhos = img_np[mask]
    nbins = upper - lower + 1
    freq, bin_edges = np.histogram(rhos, bins=nbins)
    freq = dip.Gauss(freq, 3)
    
    return freq, bin_edges

def extract_hist_slicewise(img_np, lower=100, upper=1500):
    sz = img_np.shape[2]
    bin_edges = np.arange(lower, upper + 1,dtype=np.int16)
    histograms = np.empty((sz, len(bin_edges)))
    for k in range(sz):
        freq, _ = get_hist(img_np[:,:,k], lower, upper)
        histograms[k,:] = freq
    return histograms

# hists = extract_hist_slicewise(img)
    
# surfaces plot of hists
outpath = outdir / f"{path.stem}_hist.pdf"

fig = plt.figure(figsize=(10, 6))
ax = fig.add_subplot(111, projection='3d')
X, Y = np.meshgrid(np.arange(hists.shape[1]), np.arange(hists.shape[0]))
ax.plot_surface(X, Y, hists, cmap='viridis')
ax.set_xlabel('Intensity')
ax.set_ylabel('Slice index')
ax.set_zlabel('Frequency')
plt.tight_layout()
plt.savefig(outpath, bbox_inches='tight', dpi=300)
plt.close()

# Save hists as 2D nrrd:
permute hists to get the last axis to 1 and first axis to 3
hist_3D = hists.T[:,None,:]

outpath = outdir / f"{path.stem}_hist.nrrd"
header = header.copy()
header["sizes"] = hist_3D.shape
header["spacings"] = [.1,.1,.3]
header["type"] = "float"
header["encoding"] = "raw"
nrrd.write(str(outpath), hist_3D, header, compression_level=0)