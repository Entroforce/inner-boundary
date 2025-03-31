import numpy as np
import os
import sunpy.coordinates
import astropy.time

from astropy.io import fits
from netCDF4 import Dataset
from math import cos, pi

from scipy import interpolate

import argparse


parser = argparse.ArgumentParser()


parser.add_argument('fits_b', type=str,
                    help='fits file with radial B on 21.5Rsun (nonpolar)')
parser.add_argument('fits_bp', type=str,
                    help='fits file with polar radial B')
parser.add_argument('fits_v', type=str,
                    help='fits file with radial V on 21.5Rsun')
parser.add_argument('outfile', type=str,
                    help='resulting bnd.nc file')

parser.add_argument('--date', type=str,
                    help='date (YYYY-MM-DD)',
                    default='')

parser.add_argument('--gong', type=str,
                    help='GONG fits file (optional)',
                    action='store',
                    default='')

parser.add_argument('--stop', action='store_true')

parser.add_argument('--cr', action='store_true')

args = parser.parse_args()

bnd_file = args.outfile

### READ WSA

fits_b = fits.open(args.fits_b)
fits_bp = fits.open(args.fits_bp)
fits_v = fits.open(args.fits_v)

def fix_nans(array):
  x = np.arange(0, array.shape[1])
  y = np.arange(0, array.shape[0])

  array = np.ma.masked_invalid(array)
  xx, yy = np.meshgrid(x, y)
  x1 = xx[~array.mask]
  y1 = yy[~array.mask]
  newarr = array[~array.mask]

  return interpolate.griddata((x1, y1), newarr.ravel(),
                          (xx, yy),
                             method='cubic')

b_wsa = np.array(fits_b[0].data)
bp_wsa = np.array(fits_bp[0].data)
v_wsa = np.array(fits_v[0].data)

v_wsa = fix_nans(v_wsa)
b_wsa = np.flip(b_wsa, axis=[0])


# values from ENLIL setting (found in bnd.nc)
vfast = 750.
dfast = 400.
xalpha = 0.05
dscale = 2.075 # (np+na+ne)/np
bfactor = 4.5e-3 / np.average(b_wsa)

tfast = 1.5e6 # K

### VELOCITY
v_wsa *=  1000
v_bnd = v_wsa

### DENSITY

## density transformed

d_bnd = np.ndarray(v_bnd.shape, np.float64)
b1_bnd = np.ndarray(v_bnd.shape, np.float64)
b3_bnd = np.ndarray(v_bnd.shape, np.float64)
bp_bnd = np.ndarray(v_bnd.shape, np.float64)
t_bnd = np.ndarray(v_bnd.shape, np.float64)

pressure_const = 2.459e10 * (4.5e-3 ** 2) + 1.293e-4 * dfast * tfast
nv_const = dfast * vfast

print(2.459e10 * (4.5e-3 ** 2), 1.293e-4 * 400 * 1.5e6)

for x in range(d_bnd.shape[0]):
  colatitude = (x + 0.5) * pi / d_bnd.shape[0]
  for y in range(d_bnd.shape[1]):
    d = dfast * vfast / (v_bnd[x, y] / 1000)
    d_bnd[x, y] = d * 1.67262171e-27 * 1e6 / dscale
    b1_bnd[x, y] = 4.5e-7 # Tesla
    bp_bnd[x, y] = (100.0 if bp_wsa[x, y] > 0 else -100.0)
    b3_bnd[x, y] = -b1_bnd[x, y] * 2 * pi * np.sin(colatitude) * 695700000 * 21.5 /(27.2753 * 86400) / v_bnd[x, y]
    t_bnd[x, y] = tfast * dfast / d

### SAVE NETCDF
res = Dataset(args.outfile, "w", format='NETCDF3_CLASSIC')

n2 = res.createDimension('n2', v_bnd.shape[0])
n3 = res.createDimension('n3', v_bnd.shape[1])
nblk = res.createDimension('nblk', 1)
ntime = res.createDimension('ntime', 1)

V1 = res.createVariable('V1', 'f8', ('ntime', 'nblk', 'n3', 'n2'))
D = res.createVariable('D', 'f8', ('ntime', 'nblk', 'n3', 'n2'))
B1 = res.createVariable('B1', 'f8', ('ntime', 'nblk', 'n3', 'n2'))
T = res.createVariable('T', 'f8', ('ntime', 'nblk', 'n3', 'n2'))
B3 = res.createVariable('B3', 'f8', ('ntime', 'nblk', 'n3', 'n2'))
BP = res.createVariable('BP', 'f8', ('ntime', 'nblk', 'n3', 'n2'))

def transform(a):
  res = np.transpose(a)
  return np.flip(res, axis=[1])

V1[0, 0, :, :] = transform(v_bnd)
D[0, 0, :, :] = transform(d_bnd)
B1[0, 0, :, :] = transform(b1_bnd)
T[0, 0, :, :] = transform(t_bnd)
B3[0, 0, :, :] = transform(b3_bnd)
BP[0, 0, :, :] = transform(bp_bnd)

res.close()
