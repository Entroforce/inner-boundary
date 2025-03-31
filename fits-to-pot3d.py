from fits2hdf.idi import IdiHdulist
from fits2hdf.io.fitsio import *
from fits2hdf.io.hdfio import *
from scipy.interpolate import *

import argparse
from astropy.io import fits


def convert_one_fits_to_hdf(args=None):
    parser = argparse.ArgumentParser(description='Convert GONG FITS files to HDF5 for POT3D')
    parser.add_argument('file_in', help='input fits')
    parser.add_argument('file_out', help='output h5')
    parser.add_argument('-f', '--flip', action='store_true')
    parser.add_argument('-s', '--sine-lat', action='store_true')
    parser.add_argument('--los', action='store_true')
    parser.add_argument('--abs', action='store_true')

    args = parser.parse_args()

    print("Reading  %s" % args.file_in)
    idi_hdu = read_fits(args.file_in)
    print("FLIP: ", args.flip)
    print("LOS: ", args.los)
    print("SINE-LAT: ", args.sine_lat)
    print("ABS: ", args.abs)
    
    print("Creating %s" % args.file_out)
    outfile = args.file_out
    with h5py.File(outfile, mode='w') as h:
        idi_hdu.hdf = h

        for gkey, gdata in idi_hdu.items():
            print("Creating %s" % gkey)

            if isinstance(idi_hdu[gkey], IdiImageHdu):
                # assuming single item
                print("Shape is ", idi_hdu[gkey].data.shape) # must be (180, 360)
                if args.sine_lat:
                    colats_in = np.arccos(np.linspace(179/180.,-179/180,idi_hdu[gkey].data.shape[0]))
                else:
                    colats_in = np.deg2rad(np.linspace(0.5,179.5,idi_hdu[gkey].data.shape[0]))
                lons_in = np.deg2rad(np.linspace(0.5,359.5,idi_hdu[gkey].data.shape[1]))
                data = idi_hdu[gkey].data
                if args.los:
                    data = (data.T / np.sin(colats_in)).T
                if args.abs:
                    data = np.abs(data)
                interp = RectSphereBivariateSpline(colats_in, lons_in, data, s=0, pole_continuity=True)

                colats_out = np.deg2rad(np.linspace(0.,180.,181))
                lons_out = np.deg2rad(np.linspace(0.0,360.,361))
                colats, lons = np.meshgrid(colats_out, lons_out)
                data_out = interp.ev(colats, lons)
                dim1 = bs.create_dataset(h, "dim1", colats_out)
                dim2 = bs.create_dataset(h, "dim2", lons_out)
                # actually the input colatitudes should decrease, but RectSphereBivariateSpline
                # does not like decrease, so we flip the direction here
                if args.flip:
                    dset = bs.create_dataset(h, "Data", np.flip(data_out, 1))
                else:
                    dset = bs.create_dataset(h, "Data", data_out)
                dset.attrs["DIMENSION_LABELS"] = np.string_(["dim1", "dim2"])
                dim1.attrs["CLASS"] = "DIMENSION_SCALE"
                dim1.attrs["NAME"] = "dim1"
                dim2.attrs["CLASS"] = "DIMENSION_SCALE"
                dim2.attrs["NAME"] = "dim2"
                dset.attrs["DIMENSION_LIST"] = np.array([(dim1.ref), (dim2.ref)], dtype=h5py.ref_dtype)
                dt = np.dtype([('dataset', h5py.ref_dtype), ('dimension', 'i4')])
                dim1.attrs["REFERENCE_LIST"] = np.array([(dset.ref, 0)], dtype=dt)
                dim2.attrs["REFERENCE_LIST"] = np.array([(dset.ref, 1)], dtype=dt)

            elif isinstance(idi_hdu[gkey], IdiPrimaryHdu):
                pass

            #write_headers(h, idi_hdu[gkey])
            if six.PY2:
                unicode_dt = h5py.special_dtype(vlen=unicode)
            else:
                unicode_dt = h5py.special_dtype(vlen=str)

if __name__ == '__main__':
    convert_one_fits_to_hdf()
