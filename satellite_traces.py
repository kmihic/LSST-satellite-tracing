import numpy as np
import healpy as hp
from sat_utils import _raDec2Hpid, hpid2RaDec, get_pos_Sun, angularSeparation
import ccd_mask as cm
import math
import observatory as ob
import strlnk_constellation as sl_constellation
import camera_image as cam
from astropy.time import Time
import sys
import matplotlib.pyplot as plt
  
  
def get_image(observation_time_mjd, epoch, observation_ra=np.nan, observation_dec=np.nan, observation_az=np.nan, observation_alt=np.nan,
              trace_width = 10, exposure_time = 30., sat_mult = 1, max_speed=1., rnd_seed = 1234, camera_mode='pix',
              plot_image=False, plot_image_filename='',):
   
  if camera_mode == 'pix':
    c_mode = 0
  elif camera_mode == 'ccd':
    c_mode = 1
  elif camera_mode == 'traces':
    c_mode = 2 
  else:
    sys.exit('Allowed camera modes: pix, ccd or traces')
  # create a camera object
  if c_mode > 0:
    camera = cam.Camera(init_masks=False)
  else:
    camera = cam.Camera()
  N_pix = camera.get_camera_pix_num()
  N_ccd = camera.get_camera_ccd_num()
  # get telescope/camera radii (field of view)           
  t_fov_rad = camera.get_fov_radii()
  # Exposure time given in seconds, convert to day
  e_time = exposure_time/(3600.*24.)
   # Define perimeter around telescope based on max speed of sats and 
  t_peri = math.radians(max_speed * exposure_time) +t_fov_rad# speed given in degree/s   
  # create the observatory object and initialize the satellite constellation
  observatory = ob.Observatory(epoch, sl_constellation.get_constellation, sat_mult, o_name='LSST', rnd_seed = rnd_seed)
  # update sat locations to the (current) observation time
  observatory.set_observation_time(observation_time_mjd)
  observatory.update_sat_locations()
  # get telescope positions in ra/dec, if needed
  if np.isfinite(observation_az) & np.isfinite(observation_alt):
    t_ra, t_dec = observatory.get_observation_radec(math.radians(observation_az), math.radians(observation_alt), observation_time_mjd)  
  elif np.isfinite(observation_ra) & np.isfinite(observation_dec):
    t_ra = math.radians(observation_ra)
    t_dec = math.radians(observation_dec)
  else:
    sys.exit('Must define observation coordinates in either ra/dec or az/alt')
  # allocate space for the output/working vars    
  N_sat = observatory.get_num_sat()
  ang_dist = np.empty(N_sat, dtype=np.double)
  
  # make an observation
  s_ra, s_dec = observatory.get_sat_last_known_pos_radec()
  s_eclipsed = observatory.get_sat_is_eclipsed()
  # traces can be fully ot partially eclipsed. For the later we can have either start or end pt in dark
  # with the equal prob. Given that we do not calc mid-points, the best we can do is to take half of the part-traces
  # (if a trace starts visable, keep it regardless if it ends in a shade)
  ix_v = (s_eclipsed==False)
  # get the angle between satelites and the telescope
  for ix_sat in range(0, N_sat):
    ang_dist[ix_sat] = angularSeparation(t_ra, t_dec, s_ra[ix_sat], s_dec[ix_sat])
  # filter out sats not visible and outside of the perimeter
  ix_p = (ang_dist <= t_peri)
  ix = ix_p & ix_v
  close_by_sat_ix = np.where(ix)[0]
  if np.size(close_by_sat_ix)>0:
    # find end positions
    e_ra, e_dec, e_eclipsed, *_ = observatory.get_sat_location_at_time(observation_time_mjd+e_time,close_by_sat_ix) 
    if c_mode == 0:
      # take a snapshot and process the image
      camera.take_image_pix(s_ra[close_by_sat_ix],s_dec[close_by_sat_ix],e_ra,e_dec,t_ra, t_dec, 
             trace_width, plot_image = plot_image, plot_filename_wccd = plot_image_filename)   
      num_pix = camera.get_traces_pix_num()
      perc_pix = num_pix * 100./N_pix  
      num_traces = camera.get_traces_num()
      len_traces = camera.get_traces_len()
      num_ccd = camera.get_traces_ccd_num()
      perc_ccd = num_ccd * 100./N_ccd
      return num_pix, perc_pix, num_ccd, perc_ccd, num_traces, len_traces, N_sat
    elif c_mode == 1:
      camera.take_image_ccd(s_ra[close_by_sat_ix],s_dec[close_by_sat_ix], e_ra, e_dec, t_ra, t_dec, 
             plot_image = plot_image, plot_filename = plot_image_filename)
      num_traces = camera.get_traces_num()
      len_traces = camera.get_traces_len()
      num_ccd = camera.get_traces_ccd_num()
      perc_ccd = num_ccd * 100./N_ccd
      ccd_id = camera.get_traces_ccd_id()
      traces_per_ccd = camera.get_traces_per_ccd()
      return num_ccd, perc_ccd, traces_per_ccd, ccd_id, num_traces, len_traces, N_sat
    else:
      camera.take_fov_traces(s_ra[close_by_sat_ix],s_dec[close_by_sat_ix], e_ra, e_dec, t_ra, t_dec, plot_image = plot_image, plot_filename = plot_image_filename)
      num_traces = camera.get_traces_num()
      len_traces = camera.get_traces_len()
      return num_traces, len_traces, N_sat 
  

def get_traces(observation_start_time_mjd, epoch, observation_time_min, plot_step_sec=1., sat_mult=1, nside=128, rnd_seed = 1234,
                    az_min_deg = 0., az_max_deg=360., alt_min_deg=0., alt_max_deg=90., 
                    alt_min_view = 0., max_mollview_col_range = -1,
                    plot_image=False, plot_filename='', simple_traces=False):

  # Plot steps are given in seconds, convert to day
  step = plot_step_sec/(3600.*24.)
  # Observation time given in minutes, convert to day
  o_time = observation_time_min/(60.*24.)
  # Min/max azimuth/altitude given in degrees, convert to radians
  az_min = np.radians(az_min_deg)
  az_max = np.radians(az_max_deg)
  alt_min = np.radians(alt_min_deg)
  alt_max = np.radians(alt_max_deg)
  # create the observatory object and initialize the satellite constellation
  observatory = ob.Observatory(epoch, sl_constellation.get_constellation, sat_mult, o_name='LSST', rnd_seed = rnd_seed)
  sun_az, sun_alt, *_ = np.degrees(get_pos_Sun(observation_start_time_mjd))
  
  # Get traces
  azimuth, altitudes  = observatory.get_traces (observation_start_time_mjd, o_time, step, az_min, az_max, alt_min, alt_max)
  N_sat = observatory.get_num_sat()
  if plot_image:  
    # want to see the sky above the observer, set ra==az, dec==alt  
    # transform the coordinates to healpixels
    hpids = _raDec2Hpid(nside, azimuth,altitudes)
    # prepare mollview: darken all outside the fov   
    N = hp.nside2npix(nside)
    result = np.zeros(N)
    ra,dec = hpid2RaDec(nside, np.arange(N))
    # mark pixels of satelites' traces
    if simple_traces:
      result[hpids] += 1
      mx=1
    else: 
      # highlight different "speeds" (multiple az/alt map to the same heatpixel(ss)
      for ii in hpids:
        result[ii] +=1
      if max_mollview_col_range <=0:
        mx=np.nanmax(result)
      else:
        mx = max_mollview_col_range   
    # remove traces below min visible altitude
    result[(dec<=alt_min_view)]=0       
    # hide anything below the horizont
    result[(dec<=0)] = np.nan
    title='%i sats, sun az,alt: %.1f, %.2f degrees\nObservation time: %.1f minutes\nMin visible altitude: %.1f degrees' % (N_sat, sun_az, sun_alt, observation_time_min, alt_min_view)    
    hp.mollview(result, rot=(0,90,0), max=mx)
    plt.rcParams.update({"text.usetex" : True})
    plt.title(title,fontsize=12)
    plt.axis('auto')
    plt.gca().set_aspect('equal', adjustable='box')
    if len(plot_filename)>0 & ~plot_filename.isspace():
        plt.savefig(plot_filename,bbox_inches='tight') 
    #hp.mollzoom(result, rot=(0,90,0),  max=10)
  else:
    result=np.empty((0,1))
    
  return  azimuth, altitudes, N_sat, sun_az, sun_alt, result

