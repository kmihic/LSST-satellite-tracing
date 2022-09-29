
import sys
import numpy as np
import math
import observatory as ob
import strlnk_constellation as sl_constellation
import camera_image as cam
from astropy.time import Time
import sat_utils as su
from tabulate import tabulate
   
def get_trace_info_fix_az_sun(sat_mult, traces_width, rnd_seed=[123], observation_time_mjd=59863.0097259146, epoch=22050.1, 
      print_latex=False, save_results=False, plot_image=False, plot_filename=''):
  
  rnd_seed = np.array(rnd_seed) 
  n_sats_x = np.array(sat_mult) # for starling constelation 1x = 11k
  trace_w = np.array(traces_width) # in camera pixels
  o_time = Time(observation_time_mjd,format='mjd')
  # define observations' coordinates
  observations_alt = np.concatenate([np.arange(20,90,10),np.arange(90,10,-10)])
  # Find current azimuth of Sun  
  # (recall: input to get_trace_info() is in degrees)
  sun_az, sun_alt, *_ = np.degrees(su.get_pos_Sun(observation_time_mjd))
  observations_az =  np.concatenate([np.ones(8)*sun_az,np.ones(7)*(sun_az+180.)])%360  
  

  # run the experiments
  N = np.size(n_sats_x)
  # for each number of satellites
  for ix_ns in range(0,N):
    p_pix, p_ccd, num_traces, len_traces, N_sat = get_images(observations_az, observations_alt, observation_time_mjd, epoch, #
        traces_width = trace_w, sat_mult = n_sats_x[ix_ns],  rnd_seed = rnd_seed, plot_image=False, plot_filename=plot_filename )   
    p_pix_avg = np.mean(p_pix[:,:,:], axis=2)
    p_pix_std = np.std(p_pix[:,:,:], axis=2)
    p_ccd_avg = np.mean(p_ccd[:,:], axis=1)
    p_ccd_std = np.std(p_ccd[:,:], axis=1)
    n_tr_avg = np.mean(num_traces[:,:], axis=1)
    n_tr_std = np.std(num_traces[:,:], axis=1)
    l_tr_avg = np.mean(len_traces[:,:], axis=1)
    l_tr_std = np.std(len_traces[:,:], axis=1)
    # make tables
    title = "Sample mean\nNum experiments: {:d}\n".format(np.size(rnd_seed))
    T_avg = su.make_table(title, N_sat, trace_w, p_pix_avg, p_ccd_avg, n_tr_avg, l_tr_avg, o_time, sun_az, sun_alt,observations_az, observations_alt)
    title = "Sample standard deviation\nNum experiments: {:d}\n".format(np.size(rnd_seed))
    T_std = su.make_table(title, N_sat, trace_w, p_pix_std, p_ccd_std, n_tr_std, l_tr_std, o_time, sun_az, sun_alt,observations_az, observations_alt)
    print (tabulate(T_avg,tablefmt="plain"))#,tablefmt="latex",headers="firstrow"))
    print("-------------------------------------------------------------------------------------")
    print (tabulate(T_std,tablefmt="plain"))
    print("-------------------------------------------------------------------------------------")
    if ix_ns == 0:
      T_out_a=np.empty([N,np.shape(T_avg)[0],np.shape(T_avg)[1]],dtype='<U46')
      T_out_s=np.empty([N,np.shape(T_avg)[0],np.shape(T_avg)[1]],dtype='<U46')
    T_out_a[ix_ns,:,:] = T_avg  
    T_out_s[ix_ns,:,:] = T_std 
    if save_results:  
      filename = "az_sunpos_alt_20_90_20_nsat_{:d}_rawdata".format(N_sat)
      np.savez(filename, p_pix=p_pix, p_ccd=p_ccd, 
           num_traces=num_traces, len_traces=len_traces)
      filename = "az_sunpos_alt_20_90_20_nsat_{:d}_mean_std".format(N_sat)
      np.savez(filename, table_mean=T_avg, table_std=T_std)
      
  if print_latex:
    print ("####################################### LATEX ######################################")
    for ix_ns in range(0,N):
      print (tabulate(T_out_a[ix_ns,:,:],tablefmt="latex"))
      print (tabulate(T_out_a[ix_ns,:,:],tablefmt="latex")) 
    
     
def get_trace_info_fix_alt(sat_mult, traces_width, observation_alt, rnd_seed=[123], observation_time_mjd=59863.0097259146, epoch=22050.1, 
    print_latex=False, save_results=False, plot_image=False, plot_filename=''):
  
  rnd_seed=np.array(rnd_seed)
  n_sats_x = np.array(sat_mult) # for starling constelation 1x = 11k
  trace_w = np.array(traces_width) # in camera pixels
  o_time = Time(observation_time_mjd,format='mjd')  
  sun_az, sun_alt, *_ = np.degrees(su.get_pos_Sun(observation_time_mjd))
  # define observations' coordinates
  observations_az = np.arange(0,370,10)
  observations_alt = np.ones(np.size(observations_az))*observation_alt
  # run the experiments
  # for each number of satellites
  N = np.size(n_sats_x)
  for ix_ns in range(0,N):
    p_pix, p_ccd, num_traces, len_traces, N_sat = get_images(observations_az, observations_alt, observation_time_mjd, epoch, #
        traces_width = trace_w, sat_mult = n_sats_x[ix_ns],  rnd_seed = rnd_seed, plot_image=plot_image, plot_filename=plot_filename)   
    p_pix_avg = np.mean(p_pix[:,:,:], axis=2)
    p_pix_std = np.std(p_pix[:,:,:], axis=2)
    p_ccd_avg = np.mean(p_ccd[:,:], axis=1)
    p_ccd_std = np.std(p_ccd[:,:], axis=1)
    n_tr_avg = np.mean(num_traces[:,:], axis=1)
    n_tr_std = np.std(num_traces[:,:], axis=1)
    l_tr_avg = np.mean(len_traces[:,:], axis=1)
    l_tr_std = np.std(len_traces[:,:], axis=1)
    # make tables
    title = "Sample mean\nNum experiments: {:d}\n".format(np.size(rnd_seed))
    T_avg = su.make_table(title, N_sat, trace_w, p_pix_avg, p_ccd_avg, n_tr_avg, l_tr_avg, o_time, sun_az, sun_alt,observations_az, observations_alt)
    title = "Sample standard deviation\nNum experiments: {:d}\n".format(np.size(rnd_seed))
    T_std = su.make_table(title, N_sat, trace_w, p_pix_std, p_ccd_std, n_tr_std, l_tr_std, o_time, sun_az, sun_alt,observations_az, observations_alt)
    
    
    print (tabulate(T_avg,tablefmt="plain"))#,tablefmt="latex",headers="firstrow"))
    print("-------------------------------------------------------------------------------------")
    print (tabulate(T_std,tablefmt="plain"))
    print("-------------------------------------------------------------------------------------")
    if ix_ns == 0:
      T_out_a=np.empty([N,np.shape(T_avg)[0],np.shape(T_avg)[1]],dtype='<U46')
      T_out_s=np.empty([N,np.shape(T_avg)[0],np.shape(T_avg)[1]],dtype='<U46')
    T_out_a[ix_ns,:,:] = T_avg  
    T_out_s[ix_ns,:,:] = T_std     
    if save_results:  
      filename = "az_0_360_alt_20_nsat_{:d}_rawdata".format(N_sat)
      np.savez(filename, p_pix=p_pix, p_ccd=p_ccd, 
           num_traces=num_traces, len_traces=len_traces)
      filename = "az_0_360_alt_20_nsat_{:d}_mean_std".format(N_sat)
      np.savez(filename, table_mean=T_avg, table_std=T_std)
      
  if print_latex:
    print ("####################################### LATEX ######################################")
    for ix_ns in range(0,N):
      print (tabulate(T_out_a[ix_ns,:,:],tablefmt="latex"))
      print (tabulate(T_out_a[ix_ns,:,:],tablefmt="latex"))      


def get_images(observations_abs_poz_az, observations_abs_poz_alt, observation_time_mjd, epoch, traces_width = [10], exposure_time = 30., 
          sat_mult = 1, max_speed=1., plot_image=False, plot_filename='', rnd_seed = [1234]):
    
  # create camera object
  camera = cam.Camera()
  N_pix = camera.get_camera_pix_num()
  N_ccd = camera.get_camera_ccd_num()
  # convert input to radians (recall: PyEphem’s convention: a float point number is radians, while a string is interpreted as degrees.)
  o_az = np.radians(observations_abs_poz_az)
  o_alt = np.radians(observations_abs_poz_alt)
  # get telescope/camera radii (field of view)           
  t_fov_rad = camera.get_fov_radii()
  # Exposure time given in seconds, convert to day
  e_time = exposure_time/(3600.*24.)
   # Define perimeter around telescope based on max speed of sats and 
  t_peri = math.radians(max_speed * exposure_time) +t_fov_rad# speed given in degree/s   
  # create the observatory object and initialize the satellite constellation
  observatory = ob.Observatory(epoch, sl_constellation.get_constellation, sat_mult, o_name='LSST')
  # get telescope positions in ra/dec
  t_ra, t_dec = observatory.get_observations_radec(o_az, o_alt, observation_time_mjd)  
  # allocate space for the output/working vars  
  rnd_seeds = np.array(rnd_seed)
  N_sat = observatory.get_num_sat()
  ang_dist = np.empty(N_sat, dtype=np.double)
  N_ra = np.size(t_ra)  
  N_dec = np.size(t_dec)
  if (N_ra != N_dec):
    sys.exit("RA and DEC arrays must be of the same length")
  trace_w = np.array(traces_width)  
  N_tr = np.size(trace_w)
  N_rnd = np.size(rnd_seed)
  p_pix = np.zeros([N_tr,N_ra, N_rnd], dtype=np.double)  
  p_ccd = np.zeros([N_ra,N_rnd], dtype=np.double)
  num_traces = np.zeros([N_ra,N_rnd], dtype=np.int32)
  len_traces = np.zeros([N_ra,N_rnd], dtype=np.double)
  
  # for each rnd_seed
  for ix_rnd in range(0,N_rnd):
    observatory.reset_constellation(rnd_seeds[ix_rnd])
    observatory.set_observation_time(observation_time_mjd) 
    observatory.update_sat_locations()
    s_ra, s_dec = observatory.get_sat_last_known_pos_radec()
    s_eclipsed = observatory.get_sat_is_eclipsed()
    # traces can be fully ot partially eclipsed. For the later we can have either start or end pt in dark
    # with the equal prob. Given that we do not calc mid-points, the best we can do is to take half of the part-traces
    # (if a trace starts visable, keep it regardless if it ends in a shade)
    ix_v = (s_eclipsed==False)
    # for each coordinate
    for ix_c in range(0,N_ra):
      # get the angle between satelites and the telescope
      for ix_sat in range(0, N_sat):
        ang_dist[ix_sat] = su.angularSeparation(t_ra[ix_c], t_dec[ix_c], s_ra[ix_sat], s_dec[ix_sat])
      # filter out sats not visible and outside of the perimeter
      ix_p = (ang_dist <= t_peri)
      ix = ix_p & ix_v
      close_by_sat_ix = np.where(ix)[0]
      if np.size(close_by_sat_ix)>0:
        # find end positions
        e_ra, e_dec, e_eclipsed, e_az, e_alt = observatory.get_sat_location_at_time(observation_time_mjd+e_time,close_by_sat_ix) 
        # for each trace width
        for ix_tr in range(0,N_tr):        
          # take a snapshot and process the image
          camera.take_image_pix(s_ra[close_by_sat_ix],s_dec[close_by_sat_ix],e_ra,e_dec,t_ra[ix_c], t_dec[ix_c], 
               trace_w[ix_tr], plot_image = plot_image, plot_filename_wccd = plot_filename)   
          n_pix = camera.get_traces_pix_num()
          # put results into arrays for post-processing
          # [trace_w, coord, rnd]
          p_pix[ix_tr, ix_c, ix_rnd] = n_pix * 100./N_pix  
        # not dependent on the trace width:          
        n_tr = camera.get_traces_num()
        l_tr = camera.get_traces_len()
        n_ccd = camera.get_traces_ccd_num()
      #[coord, rnd]
      p_ccd[ix_c, ix_rnd] = n_ccd * 100./N_ccd
      num_traces[ix_c, ix_rnd] = n_tr
      len_traces[ix_c, ix_rnd] = l_tr
         
  return p_pix, p_ccd, num_traces, len_traces, N_sat
  

 



