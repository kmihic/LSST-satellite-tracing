import numpy as np 
from ephem import Observer, EarthSatellite, readtle, Sun
from astropy import constants as const
from astropy import units as u
from scipy import sparse
import matplotlib.pyplot as plt
import warnings
import healpy as hp



def get_pos_Sun(time_mjd, o_name='LSST'):

  sun = Sun() 
  observer = get_observer(o_name)
  observer.date = get_time_in_pyephem_format(time_mjd) # convert to pyephem datetime
  sun.compute(observer)
  return sun.az, sun.alt, sun.ra, sun.dec  
     
  
def plot_intersect(ax, ay, bx, by, LLx, LLy, URx, URy, title, radii=0, offset=0, clr='b',linewidth = 2):

  plt.rcParams.update({"text.usetex" : True})
  x=np.array([LLx, LLx, URx, URx, LLx])
  y=np.array([LLy, URy, URy, LLy, LLy])
  plt.plot(x, y, linestyle = '-', linewidth = 1, c = 'r')
  if radii > 0:
    theta = np.linspace(0, 2*np.pi, 100)
    x = radii*np.cos(theta) + offset
    y = radii*np.sin(theta) + offset
    plt.plot(x,y,linewidth = 1, c = 'r')
  for ii in range(0,np.size(ax)):
    plt.plot([ax[ii], bx[ii]],[ay[ii], by[ii]], linestyle = '-', linewidth = linewidth, c = clr, marker='.')
  plt.title(title,fontsize=12)  
  plt.axis('auto')
  plt.gca().set_aspect('equal', adjustable='box')
  
def repelem(Q, n_rep_r, n_rep_c): 
  
  # get arrays
  data = Q.data
  col = Q.indices
  row = np.repeat(np.arange(0,Q.get_shape()[0]),np.diff(Q.indptr))
  #expand in col 
  data = np.repeat(data, n_rep_c)
  row = np.repeat(row, n_rep_c)
  col=np.concatenate(np.arange(0,n_rep_c) + col[:,None]*n_rep_c)   
  #expand in row
  data = np.repeat(data, n_rep_r)
  col = np.repeat(col, n_rep_r)
  row = np.concatenate(np.arange(0,n_rep_r) + row[:,None]*n_rep_r)
  
  M = Q.get_shape()[0]
  N = Q.get_shape()[1]
  Q = sparse.csr_matrix((data,(row,col)),(M*n_rep_r,N*n_rep_c))
  return Q
  
def repelem_coo(Q, n_rep): 
  
  #expand in col 
  data = np.repeat(Q.data, n_rep)
  row = np.repeat(Q.row, n_rep)
  col=np.concatenate(np.arange(0,n_rep) + Q.col[:,None]*n_rep) 
  
  #expand in row
  data = np.repeat(data, n_rep)
  col = np.repeat(col, n_rep)
  row = np.concatenate(np.arange(0,n_rep) + row[:,None]*n_rep)
  
  Q = sparse.coo_matrix((data,(row,col)))
  
  return Q
  
  
def satellite_mean_motion(altitude, mu=const.GM_earth, r_earth=const.R_earth):
    '''
    Compute mean motion of satellite at altitude in Earth's gravitational field.

    See https://en.wikipedia.org/wiki/Mean_motion#Formulae
    '''
    no = np.sqrt(4.0 * np.pi ** 2 * (altitude + r_earth) ** 3 / mu).to(u.day)
    # unit of no is sqrt(km^3)*s/sqrt(m^3) == 31622.776601683792 [s]
    return 1 / no


   
def get_sat_orbits(epoch, inclination, right_asc, mean_anomaly, mean_motion):   
 
    #Generate TLE strings from orbital parameters.
    def checksum(line):
        s = 0
        for c in line[:-1]:
            if c.isdigit():
                s += int(c)
            if c == "-":
                s += 1
        return '{:s}{:1d}'.format(line[:-1], s % 10)

    tle0 = 'Dummy'
    tle1 = checksum('1 00001U 20001A   {:14.8f}  .00000000  00000-0  50000-4 0    0X'.format(epoch))
    N = len(inclination)
    sat_list = [None] * N
    sat_nr = 8000
    for j in range(N):
      tle2 = checksum(
        '2 00001 {:8.4f} {:8.4f} 0001000   0.0000 {:8.4f} '
        '{:11.8f}    0X'.format(
            inclination[j], right_asc[j],
            mean_anomaly[j], mean_motion[j]
        ))
      sat = readtle(tle0,tle1,tle2) 
      sat_list[j] = sat
    return sat_list

def get_observer(o_name):
  
  telescope = Site(name=o_name)
  observer = Observer()
  observer.lat = telescope.latitude_rad
  observer.lon = telescope.longitude_rad
  observer.elevation = telescope.height
  return observer

def get_time_in_pyephem_format(time_mjd):
    # Date and time in PyEphem are stored using floating point numbers, that denote the number of days passed since 1899/12/31 12:00:00 UTC
    # which is 1.50195e+04
    mjd_eph_date_diff = 15019.5
    time_eph = time_mjd - mjd_eph_date_diff
    
    return time_eph

    
def angularSeparation(long1, lat1, long2, lat2):
    """
    angle between 2 points
    """
    ## haversine distance 
    #how far apart two points on the sky are 
    t1 = np.sin(lat2/2.0 - lat1/2.0)**2
    t2 = np.cos(lat1)*np.cos(lat2)*np.sin(long2/2.0 - long1/2.0)**2
    _sum = t1 + t2

    if np.size(_sum) == 1:
        if _sum < 0.0:
            _sum = 0.0
    else:
        _sum = np.where(_sum < 0.0, 0.0, _sum)

    return 2.0*np.arcsin(np.sqrt(_sum))

def make_table(title, N_sat, trace_w, p_pix, p_ccd, n_tr, l_tr, o_time, sun_az, sun_alt, o_az,o_alt):
  
  # header/title rows
  T = np.array(["Num satellites: {:d}".format(N_sat)])
  T = np.vstack((T,np.array(["Observation date/time: {:s}".format(o_time.to_value('iso'))])))
  T = np.vstack((T,np.array(["Sun az/alt: ({:.0f},{:.0f})".format(sun_az, sun_alt)])))
  T = np.vstack((T,["##"]))
  T = np.vstack((T,[title]))  
  T = np.vstack((T,["##"]))  
  T = np.hstack((T,np.full((np.size(T,axis=0), np.size(p_pix,axis=1))," ")))
  col_name = ["camera data / (observation az/alt)"]
  empty_row = ["__________________________________"]  
  for ii in range(0,np.size(o_az)):
    col_name = np.append(col_name, ["({:.0f},{:.0f})".format(o_az[ii],o_alt[ii])])
    empty_row = np.append(empty_row, " ")
  T = np.vstack((T,col_name)) 
  T = np.vstack((T,empty_row)) 
  # data
  row_name_tr_w = []
  for tr_w in trace_w:
    row_name_tr_w  = np.append(row_name_tr_w,["% pix (trace width = {:d})".format(tr_w)]) 
  tbl = np.hstack((row_name_tr_w[:,None], np.round(p_pix,decimals=2)))
  tbl = np.vstack((tbl,np.hstack((['% ccd'], np.round(p_ccd,decimals=2)))))
  tbl = np.vstack((tbl,np.hstack((['num traces'], np.round(n_tr,decimals=2)))))
  tbl = np.vstack((tbl,np.hstack((['length traces'], np.round(l_tr,decimals=2)))))
  #
  T = np.vstack((T,tbl))
    
  return T  


#
# Taken from https://github.com/lsst/rubin_sim.git, 
#   rubin_sim.utils package
#

class LSST_site_parameters(object):
    """
    This is a struct containing the LSST site parameters as defined in

    https://docushare.lsstcorp.org/docushare/dsweb/ImageStoreViewer/LSE-30

    (accessed on 4 January 2016)

    This class only exists for initializing Site with LSST parameter values.
    Users should not be accessing this class directly.
    """

    def __init__(self):
        self.longitude = -70.7494  # in degrees
        self.latitude = -30.2444  # in degrees
        self.height = 2650.0  # in meters
        self.temperature = 11.5  # in centigrade
        self.pressure = 750.0  # in millibars
        self.humidity = 0.4  # scale 0-1
        self.lapseRate = 0.0065  # in Kelvin per meter
        # the lapse rate was not specified by LSE-30;
        # 0.0065 K/m appears to be the "standard" value
        # see, for example http://mnras.oxfordjournals.org/content/365/4/1235.full


#
# Taken from https://github.com/lsst/rubin_sim.git, 
#   rubin_sim.utils package
#

class Site(object):
    """
    This class will store site information for use in Catalog objects.

    Defaults values are LSST site values taken from the Observatory System Specification
    document
    https://docushare.lsstcorp.org/docushare/dsweb/ImageStoreViewer/LSE-30
    on 4 January 2016

    Parameters
    ----------
    name : `str`, opt
        The name of the observatory. Set to 'LSST' for other parameters to default to LSST values.
    longitude : `float`, opt
        Longitude of the site in degrees.
    latitude : `float`, opt
        Latitude of the site in degrees.
    height : `float`, opt
        Height of the site in meters.
    temperature : `float`, opt
        Mean temperature in Centigrade
    pressure : `float`, opt
        Pressure for the site in millibars.
    humidity : `float`, opt
        Relative humidity (range 0-1).
    lapseRate : `float`, opt
        Change in temperature in Kelvins per meter
    """

    def __init__(
        self,
        name=None,
        longitude=None,
        latitude=None,
        height=None,
        temperature=None,
        pressure=None,
        humidity=None,
        lapseRate=None,
    ):

        default_params = None
        self._name = name
        if self._name == "LSST":
            default_params = LSST_site_parameters()

        if default_params is not None:
            if longitude is None:
                longitude = default_params.longitude

            if latitude is None:
                latitude = default_params.latitude

            if height is None:
                height = default_params.height

            if temperature is None:
                temperature = default_params.temperature

            if pressure is None:
                pressure = default_params.pressure

            if humidity is None:
                humidity = default_params.humidity

            if lapseRate is None:
                lapseRate = default_params.lapseRate

        if longitude is not None:
            self._longitude_rad = np.radians(longitude)
        else:
            self._longitude_rad = None

        if latitude is not None:
            self._latitude_rad = np.radians(latitude)
        else:
            self._latitude_rad = None

        self._longitude_deg = longitude
        self._latitude_deg = latitude
        self._height = height
        self._pressure = pressure

        if temperature is not None:
            self._temperature_kelvin = temperature + 273.15  # in Kelvin
        else:
            self._temperature_kelvin = None

        self._temperature_centigrade = temperature
        self._humidity = humidity
        self._lapseRate = lapseRate

        # Go through all the attributes of this Site.
        # Raise a warning if any are None so that the user
        # is not surprised when some use of this Site fails
        # because something that should have beena a float
        # is NoneType
        list_of_nones = []
        if self.longitude is None or self.longitude_rad is None:
            if self.longitude_rad is not None:
                raise RuntimeError(
                    "in Site: longitude is None but longitude_rad is not"
                )
            if self.longitude is not None:
                raise RuntimeError(
                    "in Site: longitude_rad is None but longitude is not"
                )
            list_of_nones.append("longitude")

        if self.latitude is None or self.latitude_rad is None:
            if self.latitude_rad is not None:
                raise RuntimeError("in Site: latitude is None but latitude_rad is not")
            if self.latitude is not None:
                raise RuntimeError("in Site: latitude_rad is None but latitude is not")
            list_of_nones.append("latitude")

        if self.temperature is None or self.temperature_kelvin is None:
            if self.temperature is not None:
                raise RuntimeError(
                    "in Site: temperature_kelvin is None but temperature is not"
                )
            if self.temperature_kelvin is not None:
                raise RuntimeError(
                    "in Site: temperature is None but temperature_kelvin is not"
                )
            list_of_nones.append("temperature")

        if self.height is None:
            list_of_nones.append("height")

        if self.pressure is None:
            list_of_nones.append("pressure")

        if self.humidity is None:
            list_of_nones.append("humidity")

        if self.lapseRate is None:
            list_of_nones.append("lapseRate")

        if len(list_of_nones) != 0:
            msg = "The following attributes of your Site were None:\n"
            for name in list_of_nones:
                msg += "%s\n" % name
            msg += "If you want these to just default to LSST values,\n"
            msg += "instantiate your Site with name='LSST'"
            warnings.warn(msg)

    def __eq__(self, other):

        for param in self.__dict__:
            if param not in other.__dict__:
                return False
            if self.__dict__[param] != other.__dict__[param]:
                return False

        for param in other.__dict__:
            if param not in self.__dict__:
                return False

        return True

    def __ne__(self, other):
        return not self.__eq__(other)

    @property
    def name(self):
        """
        observatory name
        """
        return self._name

    @property
    def longitude_rad(self):
        """
        observatory longitude in radians
        """
        return self._longitude_rad

    @property
    def longitude(self):
        """
        observatory longitude in degrees
        """
        return self._longitude_deg

    @property
    def latitude_rad(self):
        """
        observatory latitude in radians
        """
        return self._latitude_rad

    @property
    def latitude(self):
        """
        observatory latitude in degrees
        """
        return self._latitude_deg

    @property
    def temperature(self):
        """
        mean temperature in centigrade
        """
        return self._temperature_centigrade

    @property
    def temperature_kelvin(self):
        """
        mean temperature in Kelvin
        """
        return self._temperature_kelvin

    @property
    def height(self):
        """
        height in meters
        """
        return self._height

    @property
    def pressure(self):
        """
        mean pressure in millibars
        """
        return self._pressure

    @property
    def humidity(self):
        """
        mean humidity in the range 0-1
        """
        return self._humidity

    @property
    def lapseRate(self):
        """
        temperature lapse rate (in Kelvin per meter)
        """
        return self._lapseRate

#
# Taken from https://github.com/lsst/rubin_sim.git, 
#   rubin_sim.utils package
#

def gnomonic_project_toxy(RA1, Dec1, RAcen, Deccen):
    """Calculate x/y projection of RA1/Dec1 in system with center at RAcen, Deccen.
    Input radians. Grabbed from sims_selfcal"""
    # also used in Global Telescope Network website
    cosc = np.sin(Deccen) * np.sin(Dec1) + np.cos(Deccen) * np.cos(Dec1) * np.cos(
        RA1 - RAcen
    )
    x = np.cos(Dec1) * np.sin(RA1 - RAcen) / cosc
    y = (
        np.cos(Deccen) * np.sin(Dec1)
        - np.sin(Deccen) * np.cos(Dec1) * np.cos(RA1 - RAcen)
    ) / cosc
    return x, y

#
# Taken from https://github.com/lsst/rubin_sim.git, 
#   rubin_sim.utils package
#
def _raDec2Hpid(nside, ra, dec, **kwargs):
    """
    Assign ra,dec points to the correct healpixel.

    Parameters
    ----------
    nside : int
        Must be a value of 2^N.
    ra : np.array
        RA values to assign to healpixels. Radians.
    dec : np.array
        Dec values to assign to healpixels. Radians.

    Returns
    -------
    hpids : np.array
        Healpixel IDs for the input positions.
    """
    lat = np.pi / 2.0 - dec
    hpids = hp.ang2pix(nside, lat, ra, **kwargs)
    return hpids

#
# Taken from https://github.com/lsst/rubin_sim.git, 
#   rubin_sim.utils package
#
def hpid2RaDec(nside, hpids, **kwargs):
    """
    Correct for healpy being silly and running dec from 0-180.

    Parameters
    ----------
    nside : int
        Must be a value of 2^N.
    hpids : np.array
        Array (or single value) of healpixel IDs.

    Returns
    -------
    raRet : float (or np.array)
        RA positions of the input healpixel IDs. In degrees.
    decRet : float (or np.array)
        Dec positions of the input healpixel IDs. In degrees.
    """
    ra, dec = _hpid2RaDec(nside, hpids, **kwargs)
    return np.degrees(ra), np.degrees(dec)

#
# Taken from https://github.com/lsst/rubin_sim.git, 
#   rubin_sim.utils package
#
def _hpid2RaDec(nside, hpids, **kwargs):
    """
    Correct for healpy being silly and running dec from 0-180.

    Parameters
    ----------
    nside : int
        Must be a value of 2^N.
    hpids : np.array
        Array (or single value) of healpixel IDs.

    Returns
    -------
    raRet : float (or np.array)
        RA positions of the input healpixel IDs. In radians.
    decRet : float (or np.array)
        Dec positions of the input healpixel IDs. In radians.
    """

    lat, lon = hp.pix2ang(nside, hpids, **kwargs)
    decRet = np.pi / 2.0 - lat
    raRet = lon

    return raRet, decRet