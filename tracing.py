import numpy as np
import astropy.coordinates as astrocoords
import astropy.units as u
import astropy.constants as const
import pfsspy
from data_types import Grid, OutputLike
import itertools
from matplotlib import pyplot as plt
from astropy.io import fits
import argparse

import h5py
from scipy.interpolate import *
from streamtracer import StreamTracer
import sunpy
import time

NLAT = 90
NLON = 180

def save_HDU(arr, name):
    hdu = fits.PrimaryHDU(arr)
    hdu.writeto(name, overwrite=True)

def remap_to_uniform_lat(map2d):
    map2d = np.copy(map2d)
    N = map2d.shape[1]
    pts_old = 2 * np.arccos(1 - 2.0 * np.array([i / float(N - 1) for i in range(N)])) / np.pi - 1
    pts_new = np.linspace(-1, 1, N)
    for i in range(map2d.shape[0]):
        vals = map2d[i,:]
        map2d[i,:] = np.interp(pts_new, pts_old, vals)
    return map2d

NLAT_3DMAP = 180
NLON_3DMAP = 360

def read_pot3d_3dmap(bp_file, bt_file, br_file, r_ss, model= "pfss"):
    # output grid
    NR = 149
    min_r = 1

    if model == "pfcs":
        NR = 300
        min_r = 2.5

    output_map = np.zeros((NLON_3DMAP + 1, NLAT_3DMAP + 1, NR + 1, 3), np.float64)
    grid_pot3d = Grid(NLAT_3DMAP, NLON_3DMAP, NR, r_ss, min_r)

    coords_out = np.empty(((NLON_3DMAP + 1) * (NLAT_3DMAP + 1) * (NR + 1), 3))
    r_grid     = np.exp(grid_pot3d.rg)
    #r_grid     = grid_pot3d.rg
    lon_grid   = grid_pot3d.pg
    colat_grid = np.arccos(grid_pot3d.sg)
    
    print("Building output grid...")
    start = time.time()
    sin_lng = np.sin(lon_grid)
    cos_lng = np.cos(lon_grid)
    sin_colat = np.sin(colat_grid)
    cos_colat = np.cos(colat_grid)
    u = 0
    for coord in itertools.product(range(len(lon_grid)), range(len(colat_grid)), r_grid):
        (lng, colat, r) = coord
        coords_out[u][0] = r * sin_colat[colat] * cos_lng[lng]
        coords_out[u][1] = r * sin_colat[colat] * sin_lng[lng]
        coords_out[u][2] = r * cos_colat[colat]
        u += 1
    end = time.time()
    print("Time:", end - start)

    index = 0
    interps = []
    for f in [bp_file, bt_file, br_file]:
        h = h5py.File(f, mode='r+')
        
        # radial
        dim1 = np.array([h["dim1"][i] for i in range(h["dim1"].shape[0])])
        # colatitude
        dim2 = np.array([h["dim2"][i] for i in range(h["dim2"].shape[0])])
        # longitude
        dim3 = np.array([h["dim3"][i] for i in range(h["dim3"].shape[0])])

        values_3d = h["Data"][:,:,:]

        print("Data shape", values_3d.shape)

        # input grid
        print("Building grid...")
        coords = np.empty((dim1.size * dim2.size * dim3.size, 3))
        start = time.time()

        sin_lng = np.sin(dim3)
        cos_lng = np.cos(dim3)
        sin_colat = np.sin(dim2)
        cos_colat = np.cos(dim2)
        u = 0
        for inds in itertools.product(range(len(dim3)), range(len(dim2)), dim1):
            (lng, colat, r) = inds
            coords[u][0] = r * sin_colat[colat] * cos_lng[lng]
            coords[u][1] = r * sin_colat[colat] * sin_lng[lng]
            coords[u][2] = r * cos_colat[colat]
            u += 1
        end = time.time()
        print("Time:", end - start)

        values = values_3d.flatten()
        print("Building interpolator...")
        interp = NearestNDInterpolator(coords, values)
        print("Interpolating...")
        start = time.time()
        values_out = interp(coords_out).reshape((NLON_3DMAP + 1, NLAT_3DMAP + 1, NR + 1))
        end = time.time()
        print("Time:", end - start)
        
        output_map[:,:,:,index] = values_out
        # force longitude loop (would not be required if interpolation were stable)
        output_map[-1,:,:,index] = output_map[0,:,:,index]

        index += 1
        interps.append(interp)

    print("Returning 3D map")
    return output_map, grid_pot3d, interps

def vector_grid(pot3d_output, grid):
    from streamtracer import VectorGrid

    # The indexing order on the last index is (phi, s, r)
    vectors = pot3d_output.bg.copy()
    print("vectors shape", vectors.shape)

    # Correct s direction for coordinate system distortion
    sqrtsg = grid._sqrtsg_correction
    # phi correction
    with np.errstate(divide='ignore', invalid='ignore'):
        vectors[..., 0] /= sqrtsg
    
    # At the poles B_phi is now infinite. Since changes to B_phi at the
    # poles have little effect on the field line, set to zero to allow for
    # easy handline in the streamline integrator
    vectors[:, 0, :, 0] = 0
    vectors[:, -1, :, 0] = 0

    # s correction
    vectors[..., 1] *= -sqrtsg

    grid_spacing = grid._grid_spacing
    
    # Cyclic only in the phi direction
    # (theta direction becomes singular at the poles so it is not cyclic)
    cyclic = [True, False, False]
    origin_coord = [0, -1, np.log(grid.min_r)]
    #origin_coord = [0, -1, grid.min_r]

    vector_grid = VectorGrid(vectors, cyclic=cyclic,
                                origin_coord=origin_coord, grid_coords=[grid.pg, grid.sg, grid.rg])
    return vector_grid


def sph2cart(r, theta, phi):
    """
    Convert spherical coordinates to cartesian coordinates.
    """
    x = r * np.sin(theta) * np.cos(phi)
    y = r * np.sin(theta) * np.sin(phi)
    z = r * np.cos(theta)
    return x, y, z


def strum2cart(rho, s, phi):
    """
    Convert strumfric coordinates to cartesian coordinates.
    """
    r = np.exp(rho)
    #r = rho
    theta = np.arccos(s)
    return sph2cart(r, theta, phi)

def strum2sph(phi, s, rho):
    """
    Convert strumfric coordinates to cartesian coordinates.
    """
    r = np.exp(rho)
    theta = np.arccos(s)
    return [r, theta, phi]

def trace(seeds, output, step_size=1, max_steps=-1):
    if max_steps == -1:
        max_steps = int(4 * output.grid.nr / step_size)

    # Get a grid
    v_grid = vector_grid(output, output.grid)

    # Do the tracing
    #
    # Normalise step size to the radial cell size, so step size is a
    # fraction of the radial cell size.
    tracer = StreamTracer(max_steps, step_size)
    tracer.ds = step_size * output.grid._grid_spacing[2]

    tracer.trace(seeds, v_grid)
    xs = tracer.xs
    # Filter out of bounds points out
    rho_ss = np.log(output.grid.rss)
    #rho_ss = output.grid.rss

    min_r = output.grid.min_r
    
    xs = [x[(x[:, 2] <= rho_ss) & (x[:, 2] >= np.log(min_r)) &
            (np.abs(x[:, 1]) < 1), :] for x in xs]

    xs = [np.stack(strum2cart(x[:, 2], x[:, 1], x[:, 0]), axis=-1) for x in xs]
    flines = [pfsspy.fieldline.FieldLine(x[:, 0], x[:, 1], x[:, 2], output) for x in xs]
    return pfsspy.fieldline.FieldLines(flines)

coeffs_swpc = {
    "v0": 285,
    "v1": 910,
    "alpha": 0.22222,
    "beta": 1.,
    "w": 2.,
    "gamma": 0.8,
    "delta": 2.,
    "psi": 3.
}

coeffs_simplified = {
    "v0": 254,
    "v1": 1090,
    "alpha": 0.22222,
    "beta": 1.,
    "w": 2.,
    "gamma": 0.8,
    "delta": 1,
    "psi": 1
}

def wsa(fp, d, coeff):
    a1 = (coeff["v1"] - coeff["v0"]) / ((1. + fp) ** coeff["alpha"])
    a2 = (coeff["beta"] - coeff["gamma"] * np.exp(-((d / coeff["w"])  ** coeff["delta"])))
    a3 = a2 ** coeff["psi"]
    return coeff["v0"] + a1 * a3

def distance_to_coronal_hole_boundary(topologies, field_lines_fp, latitude, longitude):
    # initialize the distance to coronal hole vector.
    d = np.zeros(len(field_lines_fp))
    # longitude and latitude uniform grids in radians.
    # location of closed magnetic field lines (footprint).
    latitude = latitude[np.where(topologies == 0)[1]]
    longitude = longitude[np.where(topologies == 0)[0]]

    for ii in range(len(d)):
        try:
            phi2 = field_lines_fp[ii].solar_footpoint.lon.to(u.rad).value
            theta2 = field_lines_fp[ii].solar_footpoint.lat.to(u.rad).value
            d_full_sun = np.arccos(np.cos(latitude) * np.cos(theta2) * (np.sin(phi2) * np.sin(longitude) +\
                                                                        np.cos(phi2) * np.cos(longitude)) +\
                                   np.sin(latitude) * np.sin(theta2))
            d[ii] = np.min(d_full_sun)
        except:
            d[ii] = 0
    return d

def draw_flines(flines, filename):
    print("Building plot...")
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    print("Total field lines:", len(flines))
    for ind in range(0, len(flines), 100):
        field_line = flines[ind]
        color = {0: 'black', -1: 'tab:blue', 1: 'tab:red'}.get(field_line.polarity)
        coords = field_line.coords
        coords.representation_type = 'cartesian'
        ax.plot(coords.x / const.R_sun,
            coords.y / const.R_sun,
            coords.z / const.R_sun,
            color=color, linewidth=1)
    plt.savefig(filename)

def trace_map_pot3d(input_prefix, lon, colat, model="pfss", draw=False):
    r_ss = 2.5
    min_r = 1
    if model == "pfcs":
        r_ss = 21.5
        min_r = 2.5
        
    # colat = np.linspace(np.pi - (np.pi / 2 / NLAT), 0 + (np.pi / 2 / NLAT), NLAT, endpoint=True)
    # lon   = np.linspace(0 + (np.pi / NLON), 2 * np.pi - (np.pi / NLON), NLON, endpoint=True) 
    
    pot3d_map, grid_pot3d, interp = read_pot3d_3dmap(input_prefix + '_bp_' + model + '.h5',
                                                     input_prefix + '_bt_' + model + '.h5',
                                                     input_prefix + '_br_' + model + '.h5', r_ss, model)
        
    output = OutputLike(pot3d_map, grid_pot3d)
    output.interp = interp
         
    save_HDU(remap_to_uniform_lat(output.bg[:,:,0,0].reshape([NLON_3DMAP + 1, NLAT_3DMAP + 1])).T, input_prefix + '_bp_inner_' + model + '.fits')
    save_HDU(remap_to_uniform_lat(output.bg[:,:,-1,0].reshape([NLON_3DMAP + 1, NLAT_3DMAP + 1])).T, input_prefix + '_bp_outer_' + model + '.fits')
    save_HDU(remap_to_uniform_lat(output.bg[:,:,0,1].reshape([NLON_3DMAP + 1, NLAT_3DMAP + 1])).T, input_prefix + '_bt_inner_' + model + '.fits')
    save_HDU(remap_to_uniform_lat(output.bg[:,:,-1,1].reshape([NLON_3DMAP + 1, NLAT_3DMAP + 1])).T, input_prefix + '_bt_outer_' + model + '.fits') 
    save_HDU(remap_to_uniform_lat(output.bg[:,:,0,2].reshape([NLON_3DMAP + 1, NLAT_3DMAP + 1])).T, input_prefix + '_br_inner_' + model + '.fits')
    save_HDU(remap_to_uniform_lat(output.bg[:,:,-1,2].reshape([NLON_3DMAP + 1, NLAT_3DMAP + 1])).T, input_prefix + '_br_outer_' + model + '.fits')
         
    print("Tracing down from {}Rs...".format(r_ss))
    r     = [r_ss]
    seeds = []
    if model == "pfcs":
        seeds = np.array(list(itertools.product(lon, np.cos(colat), np.log(r))))
    else:
        seeds = np.stack([lon, np.cos(colat), np.resize(np.log(r), [len(lon),])], axis=1)
    flines_down = trace(seeds, output, max_steps=5000, step_size=0.2)
    print("Finished tracing")

    if (model == "pfss"):
        # photosphere intersection coords for further polarity check
        coords_inter = []
        for fline in flines_down:
            coords_inter.append([fline.solar_footpoint.lon.value, fline.solar_footpoint.lat.value + 90])

        coords_inter = np.floor(np.array(coords_inter) / [360, 180] * np.array([NLON, NLAT])).astype(np.int32)
        coords_inter[:, 0] = coords_inter[:, 0] % NLON
        coords_inter[:, 1] = np.minimum(coords_inter[:, 1], NLAT - 1)

        # get expansion factors
        #ef = flines_down.expansion_factors

        polarities_pfss = []
        ef = []
        fplat = []
        fplon = []
        for i in range(len(seeds)):
            s = seeds[i]
            sph = strum2sph(s[0], s[1], s[2])
            headpoint = sph2cart(sph[0], sph[1], sph[2])
            footpoint = sph2cart(1.0, (90 - flines_down[i].solar_footpoint.lat.value) / 180 * np.pi,
                                 flines_down[i].solar_footpoint.lon.value / 180 * np.pi)
            b_head = np.array([ii(headpoint) for ii in output.interp[0:3]])
            if b_head[2] > 0:
                polarities_pfss.append(1)
            else:
                polarities_pfss.append(-1)
            b_foot = np.array([ii(footpoint) for ii in output.interp[0:3]])
            ef1 = np.linalg.norm(b_foot) / np.linalg.norm(b_head) * 1.0 / (2.5 * 2.5)
            #ef1 = output.interp[3](footpoint) / output.interp[3](headpoint) * 1.0 / (2.5 * 2.5)
            ef.append(ef1)
            fplon.append(flines_down[i].solar_footpoint.lon.value)
            fplat.append(flines_down[i].solar_footpoint.lat.value)
            

        ef = np.array(ef)
        fp = ef.reshape([NLON, NLAT]).T
        save_HDU(fp, input_prefix + '_ef.fits')

        fplon = np.array(fplon)
        fplat = np.array(fplat)
        fplon = fplon.reshape([NLON, NLAT]).T
        fplat = fplat.reshape([NLON, NLAT]).T
        save_HDU(fplon, input_prefix + '_fplon.fits')
        save_HDU(fplat, input_prefix + '_fplat.fits')
        
        polarities_pfss = np.array(polarities_pfss)
        polarities_pfss = polarities_pfss.reshape([NLON, NLAT]).T
        save_HDU(polarities_pfss, input_prefix + '_br_outer_pfcs_polarity.fits')
        
        # trace up with common grid
        print("Tracing up from 1Rs...")
        r     = [1.0]
        colat_simple = np.linspace(np.pi - (np.pi / 2 / NLAT), 0 + (np.pi / 2 / NLAT), NLAT, endpoint=True)
        lon_simple   = np.linspace(0 + (np.pi / NLON), 2 * np.pi - (np.pi / NLON), NLON, endpoint=True) 
        seeds = np.array(list(itertools.product(lon_simple, np.cos(colat_simple), np.log(r))))

        flines = trace(seeds, output, max_steps=5000, step_size=0.2)
        print("Finished tracing")

        # get polarities
        polarities = flines.polarities.reshape([NLON, NLAT])

        # backward tracing check on polarities
        cnt_fake = 0
        for pt in coords_inter:
            if polarities[pt[0], pt[1]] == 0:
                cnt_fake += 1
                polarities[pt[0], pt[1]] = 2 # new value for different color
        print("Filtered fake closed points", cnt_fake)
    
        save_HDU(polarities.T, input_prefix + '_topology.fits')
        
        # calculate d on target coordinates
        d = distance_to_coronal_hole_boundary(polarities, flines_down, np.pi / 2 - colat_simple, lon_simple).reshape([NLON, NLAT]) #np.pi / 2 - 
        d = d.T * 180 / np.pi
        save_HDU(d, input_prefix + '_d.fits')

        if draw:
            draw_flines(flines, 'field_lines_up.png')

        return fp, d
    
    elif model == "pfcs":
        # get points on 2.5Rs
        coords_inter = []
        for fline in flines_down:
            coords_inter.append([fline.solar_footpoint.lon.value, fline.solar_footpoint.lat.value + 90])

        coords_inter = np.array(coords_inter)
        return coords_inter[:, 0], coords_inter[:, 1]

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Tracing & WSA')
    parser.add_argument('input_prefix', help='')
    parser.add_argument('--slow-tracer', action='store_true')
    args = parser.parse_args()

    colat = np.linspace(np.pi - (np.pi / 2 / NLAT), 0 + (np.pi / 2 / NLAT), NLAT, endpoint=True)
    lon   = np.linspace(0 + (np.pi / NLON), 2 * np.pi - (np.pi / NLON), NLON, endpoint=True)

    remap_lon, remap_colat = trace_map_pot3d(args.input_prefix, lon, colat, "pfcs", False)
    
    fp, d = trace_map_pot3d(args.input_prefix, remap_lon / 180 * np.pi, (180 - remap_colat) / 180 * np.pi, "pfss", True)

    print("Running WSA...")
    remap_lon = None
    remap_colat = None
    colat = None
    lon = None
    v_wsa = wsa(fp, d, coeffs_swpc)
    save_HDU(v_wsa, args.input_prefix + '_v_wsa.fits')

