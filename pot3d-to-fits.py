from fits2hdf.idi import IdiHdulist
from fits2hdf.io.fitsio import *
from fits2hdf.io.hdfio import restricted_hdf_keywords
import os.path

import argparse
import h5py
from scipy.interpolate import RectSphereBivariateSpline
from astropy.io import fits


def read_pot3d_hdf(infile, mode='r+', verbosity=0):
    is_file = False
    if isinstance(infile, h5py.Group):
        h = infile
    else:
        h = h5py.File(infile, mode=mode)
        is_file = True

    # Read the order of HDUs from file
    hdu_order = {}
    ipos = 0
    for gname in h.keys():
        pos = ipos
        ipos += 1
        hdu_order[pos] = gname

    data_out = None
    for pos, gname in hdu_order.items():
        group = h[gname]

        # Form header dict from
        h_vals = {}
        for key, values in group.attrs.items():
            if key not in restricted_hdf_keywords:
                h_vals[key] = values

        try:
            h_comment = group["COMMENT"]
        except Exception:
            h_comment = None
        try:
            h_history = group["HISTORY"]
        except Exception:
            h_history = None

        if gname != "Data":
            continue
        
        print("Reading %s" % gname)

        colats_in = np.array([h["dim2"][i] for i in range(h["dim2"].shape[0])])
        lons_in = np.array([h["dim3"][i] for i in range(h["dim3"].shape[0])])

        print("shape", group.shape)
        g = np.transpose(group, axes=[1,0,2])
        g = g[((colats_in >= 0) & (colats_in <= np.pi)),:,:]
        g = g[:,(lons_in >= 0) & (lons_in < 2 * np.pi),:]
        print("shape final", g.shape)
        interp = RectSphereBivariateSpline(colats_in[(colats_in >= 0) & (colats_in <= np.pi)],
                                           lons_in[(lons_in >= 0) & (lons_in < 2 * np.pi)],
                                           g[:,:,-1], s=0, pole_continuity=True)
        colats_out = np.deg2rad(np.linspace(0.5,179.5,180))
        lons_out = np.deg2rad(np.linspace(0.5,359.5,360))
        colats, lons = np.meshgrid(colats_out, lons_out)
        data_out = interp.ev(colats, lons).T
        polarity = np.sign(data_out)

    if is_file:
        h.close()

    return data_out, polarity


def convert_one_hdf_to_fits(args=None):
    """ Convert a HDF5 (in HDFITS format) to a FITS file

    An input and output directory must be specified, and all files with a matching
    extension will be converted. Command line options set the run-time settings.
    """

    # Parse options and arguments
    parser = argparse.ArgumentParser(description='Convert HDF5 produced by Pot3D to FITS.')
    parser.add_argument('file_in', help='input file (hdf5)')
    parser.add_argument('file_out', help='output file (fits)')
    args = parser.parse_args()

    file_in  = args.file_in
    file_out = args.file_out

    print("Reading %s" % file_in)
    br, polarity = read_pot3d_hdf(file_in)
    print("Saving %s" % file_out)
    hdu = fits.PrimaryHDU(br)
    hdu.writeto(file_out, overwrite=True)

if __name__ == '__main__':
    convert_one_hdf_to_fits()
