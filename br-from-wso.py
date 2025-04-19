import os.path
import numpy as np
import argparse
import scipy
import math
from astropy.io import fits

# http://wso.stanford.edu/words/pfss.pdf

# CR2066 LOS
# http://wso.stanford.edu/Harmonic.los/CR2066

NLON = 360
NLAT = 180

g = [[0.0], #[-2.202],
     [-24.744, -5.609],
     [5.718,   4.185,  11.916],
     [-24.968,  -1.495,  -3.365,  -5.612],
     [-1.484,  -2.129,  -4.002,   2.239,  -2.207],
     [-12.301,  -1.331,   2.000,   1.109,   2.921,  10.630],
     [1.875,   0.411,   2.608,  -0.442,   1.149,  -4.454,  -2.087 ],
     [ -5.185,  -3.051,  -1.868,  -0.423,  -0.890,  -2.466,   1.577,  -4.214 ],
     [ 0.564,   0.383,  -0.324,   0.987,  -0.811,   2.489,   2.217,   1.378,   5.576 ]
     ]

h = [[0.000],
     [0.000, 10.445],
     [0.000,  -5.814,  -2.652],
     [0.000, 2.401, 1.485, -7.022],
     [0.000, 1.117, 1.205, 2.624, 8.765],
     [0.000, 0.979, 0.923, 3.115, -3.172,  -2.612],
     [0.000,   0.387,   0.210,  -2.657,  -1.985,  -0.728,  -4.084],
     [0.000,   0.995,   0.745,  -0.509,   2.154,   0.272,   3.285,   4.906 ],
     [0.000,   0.471,   0.480,   3.138,   1.022,  -0.512,  -0.237,  -3.418,   0.102 ]
     ]

def main(args=None):
    parser = argparse.ArgumentParser(description='Generate fits file from WSO harmonic coefficients at photosphere')
    parser.add_argument('nmax', help='Nmax')
    parser.add_argument('file_out', help='output file (fits)')
    args = parser.parse_args()

    nmax = int(args.nmax)
    file_out = args.file_out

    br = np.zeros(shape=(NLAT, NLON))
    for i in range(NLAT):
        colat = np.pi * (i + 0.5) /  NLAT
        P = scipy.special.lpmn(nmax, nmax, np.cos(colat))[0]
        for j in range(NLON):
            lon = 2 * np.pi * (j + 0.5) / NLON

            br[i][j] = 0
            for l in range(nmax + 1):
                for m in range(l + 1):
                    # pfss.pdf, eq. (24)
                    norm = ((-1) ** m) * np.sqrt(math.factorial(l - m) / math.factorial(l + m))
                    if m > 0:
                        norm *= 2
                    # pfss.pdf, eq. (9), r = R0 = 1
                    br[i][j] += (l + 1 + l * ((1.0 / 2.5) ** (2 * l + 1))) *\
                    P[m][l] * norm *\
                    (g[l][m] * np.cos(m * lon) +\
                     h[l][m] * np.sin(m * lon)) * 1e-2
                    
    print("Saving %s" % file_out)
    hdu = fits.PrimaryHDU(br)
    hdu.writeto(file_out, overwrite=True)

if __name__ == '__main__':
    main()
