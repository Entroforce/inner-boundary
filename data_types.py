import functools

import numpy as np
import astropy.units as u
import pfsspy

def sph2cart(r, theta, phi):
    """
    Convert spherical coordinates to cartesian coordinates.
    """
    x = r * np.sin(theta) * np.cos(phi)
    y = r * np.sin(theta) * np.sin(phi)
    z = r * np.cos(theta)
    return x, y, z


class OutputLike:
    def __init__(self, bg, grid, lon0=180):
        self.bg = bg
        self.grid = grid
        self._lon0 = lon0 * u.deg
        self.coordinate_frame = "heliographic_carrington"
        self._modbg = np.linalg.norm(self.bg, axis=-1)
        
    def _integrate_one_way(self, dt, start_point):
        x = start_point

        r = []
        theta = []
        phi = []

        reached = False
        d = self._bTrace(x)

        niter = 5000

        if np.abs(x[0] - self.grid.min_r) < 1e-8:
            if d[0] * dt < 0:
                #print("  we are down, wont move")
                return np.array(sph2cart(r, theta, phi))
            
            for i in range(niter):
                if x[0] > self.grid.rss:
                    #print("  went above in", i, "steps")
                    reached = True
                    break

                if x[0] < self.grid.min_r - 1e-8:
                    reached = True
                    break

                k1 = self._bTrace(x)
                k2 = self._bTrace(x + dt * 0.5 * k1)
                k3 = self._bTrace(x + dt * 0.5 * k2)
                k4 = self._bTrace(x + dt * k3)
                
                x += dt * (k1 + 2 * k2 + 2 * k3 + k4) / 6
                r.append(x[0])
                theta.append(x[1])
                phi.append(x[2])

        if np.abs(x[0] - self.grid.rss) < 1e-8:
            if d[0] * dt > 0:
                #print("  we are up, wont move")
                return np.array(sph2cart(r, theta, phi))
            
            xprev = x.copy()
            for i in range(niter):
                #print('x', x, 'xprev', xprev, 'dt', dt)
                if x[0] < self.grid.min_r:
                    dt = dt * 0.5
                    x = xprev.copy()
                elif x[0] < self.grid.min_r + 1e-3:
                    #print("  went below in", i, "steps")
                    r.append(x[0])
                    theta.append(x[1])
                    phi.append(x[2])
                    reached = True
                    break
                else:
                    r.append(x[0])
                    theta.append(x[1])
                    phi.append(x[2])
                    xprev = x.copy()
                    k1 = self._bTrace(x)
                    k2 = self._bTrace(x + dt * 0.5 * k1)
                    k3 = self._bTrace(x + dt * 0.5 * k2)
                    k4 = self._bTrace(x + dt * k3)
                    x += dt * (k1 + 2 * k2 + 2 * k3 + k4) / 6

        if not reached:
            print("  boundary not reached")

        r = np.array(r)
        theta = np.array(theta)
        phi = np.array(phi)
        
        return np.array(sph2cart(r, theta, phi))
    
    def _bTrace(self, coord):
        r, theta, phi = coord
        x, y, z = sph2cart(r, theta, phi)

        # this goes as phi, theta, r
        b1 = np.array([ii(x, y, z) for ii in self.interp[0:3]])
        b1 /= np.linalg.norm(b1)
        # this goes as dr/ds, dtheta/ds, dphi/ds
        return np.array([b1[2], b1[1] / r, b1[0] / r / np.sin(theta)])
    
class Grid:
    """
    Grid on which the pfsspy solution is calculated.
     
     Notes
     -----
     The PFSS solution is calculated on a "strumfric" grid defined by
     
     - :math:`\rho = \log (r)`
     - :math:`s = \cos (\theta )`
     - :math:`\phi`
     
     where :math:`r, \theta, \phi` are spherical cooridnates that have ranges
     
     - :math:`min_r < r < rss`
     - :math:`0 < \theta < \pi`
     - :math:`0 < \phi < 2\pi`
     """
    def __init__(self, ns, nphi, nr, rss, min_r = 1):
        self.ns = ns
        self.nphi = nphi
        self.nr = nr
        self.min_r = min_r
        self.rss = rss
          
    @property
    def ds(self):
        """
          Cell size in cos(theta).
          """
        return 2.0 / self.ns
     
    @property
    def dr(self):
        """
          Cell size in log(r).
          """
        #return (self.rss - self.min_r) / self.nr
        return (np.log(self.rss) - np.log(self.min_r)) / self.nr
    
    @property
    def dp(self):
        """
          Cell size in phi.
          """
        return 2 * np.pi / self.nphi
    
    @property
    def rc(self):
        """
          Location of the centre of cells in log(r).
          """
        return np.linspace(np.log(self.min_r) + 0.5 * self.dr, np.log(self.rss) - 0.5 * self.dr, self.nr)
        #return np.linspace((self.min_r) + 0.5 * self.dr, (self.rss) - 0.5 * self.dr, self.nr)
     
    @property
    def sc(self):
        """
        Location of the centre of cells in cos(theta).
          """
        return np.linspace(-1 + 0.5 * self.ds, 1 - 0.5 * self.ds, self.ns)
     
    @property
    def pc(self):
        """
          Location of the centre of cells in phi.
          """
        return np.linspace(0.5 * self.dp, 2 * np.pi - 0.5 * self.dp, self.nphi)
     
    @property
    def rg(self):
        """
          Location of the edges of grid cells in log(r).
          """
        return np.linspace(np.log(self.min_r), np.log(self.rss), self.nr + 1)
     #return np.linspace((self.min_r), (self.rss), self.nr + 1)
     
    @property
    def sg(self):
        """
          Location of the edges of grid cells in cos(theta).
          """
        return np.linspace(-1, 1, self.ns + 1)
     
    @property
    def pg(self):
        """
          Location of the edges of grid cells in phi.
        """
        return np.linspace(0, 2 * np.pi, self.nphi + 1)
     
    @property
    def _grid_spacing(self):
        """
          Return grid spacing as a 3-len list.
          """
        return [self.dp, self.ds, self.dr]
     
    @property
    @functools.lru_cache()
    def _sqrtsg_correction(self):
        """
          The sqrt(1 - sg**2) correction needed to trace natively. Computed here
          once and cached for performance.
          """
        # Correct s direction for coordinate system distortion
        _, sg, _ = np.meshgrid(self.pg, self.sg, self.rg,
                               indexing='ij')
        return np.sqrt(1 - sg**2)
