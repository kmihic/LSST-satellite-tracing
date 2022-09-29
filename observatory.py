
import numpy as np
import math
import sat_utils as su


class Observatory(object):

  def __init__(self, epoch, get_constellation, sizeup_x, rnd_seed=1234, o_name='LSST'):
  
    self.epoch = epoch
    self.__init_constellation(get_constellation, rnd_seed, sizeup_x)
    self.observer = su.get_observer(o_name)
    self.num_sat = len(self.sat_list)    
    # initialize arrays
    self.alt = np.empty(self.num_sat, dtype=np.double)
    self.az = np.empty(self.num_sat, dtype=np.double)
    self.ra = np.empty(self.num_sat, dtype=np.double)
    self.dec = np.empty(self.num_sat, dtype=np.double)
    self.eclipsed = np.empty(self.num_sat, dtype=bool)
    self.all_sat = np.arange(0,self.num_sat, dtype=np.int32)
    self.curr_time = -1#su.get_time_in_pyephem_format(curr_time_mjd)   
     
  
  def __init_constellation(self, get_constellation, rnd_seed, sizeup_x):
    
    self.sizeup_x = sizeup_x
    self.constellation = get_constellation
    inclination, right_asc, mean_anomaly, mean_motion = self.constellation(*[self.sizeup_x, rnd_seed])      
    self.sat_list = su.get_sat_orbits(self.epoch, inclination, right_asc, mean_anomaly, mean_motion)
    
    
  def reset_constellation(self, rnd_seed):
    
    self.__init_constellation(self.constellation, rnd_seed, self.sizeup_x)
    
    
  def update_sat_locations(self):
    
    #c_time = su.get_time_in_pyephem_format(c_time)
    #self.observer.date = c_time
    for ix in self.all_sat:
      self.sat_list[ix].compute(self.observer)
      self.alt[ix] = self.sat_list[ix].alt
      self.az[ix] = self.sat_list[ix].az
      self.ra[ix] = self.sat_list[ix].ra
      self.dec[ix] = self.sat_list[ix].dec
      self.eclipsed[ix] = self.sat_list[ix].eclipsed      
    
    
  def get_sat_location_at_time(self, c_time, sat_ix):
    
    c_time = su.get_time_in_pyephem_format(c_time)
    self.observer.date = c_time
    N = np.size(sat_ix)
    ra = np.empty(N, dtype=np.double)
    dec = np.empty(N, dtype=np.double)
    az = np.empty(N, dtype=np.double)
    alt = np.empty(N, dtype=np.double)
    eclipsed = np.empty(N, dtype=np.double)
    cnt = 0
    for ix in sat_ix:
      self.sat_list[ix].compute(self.observer)
      ra[cnt] = self.sat_list[ix].ra
      dec[cnt] = self.sat_list[ix].dec
      az[cnt] = self.sat_list[ix].az
      alt[cnt] = self.sat_list[ix].alt
      eclipsed[cnt] = self.sat_list[ix].eclipsed
      cnt +=1
    self.observer.date = self.curr_time
      
    return ra, dec, eclipsed, az, alt
  
  def set_observation_time(self, time_mjd):    
    
    self.curr_time = su.get_time_in_pyephem_format(time_mjd)  # convert to pyephem datetime
    self.observer.date = self.curr_time    
    
  # Input: az/alt array in degrees  
  # Output: ra, dec
  def get_observations_radec(self, t_az, t_alt, time_mjd):  
  
    N = np.size(t_az)
    t_ra = np.empty(N,dtype=np.double)
    t_dec = np.empty(N,dtype=np.double)
    c_time = su.get_time_in_pyephem_format(time_mjd)
    self.observer.date = c_time
    for ii in range(0,N):
      t_ra[ii], t_dec[ii] = self.observer.radec_of(t_az[ii], t_alt[ii])
    self.observer.date = self.curr_time
    
    return t_ra, t_dec
  
  # Input: az/alt in degrees  
  # Output: ra, dec
  def get_observation_radec(self, t_az, t_alt, time_mjd):  
  
    c_time = su.get_time_in_pyephem_format(time_mjd)
    self.observer.date = c_time
    t_ra, t_dec = self.observer.radec_of(t_az, t_alt)
    self.observer.date = self.curr_time
    
    return t_ra, t_dec
        
  def get_sat_last_known_pos_radec(self):
    
    return self.ra,self.dec
  
  def get_sat_last_known_pos_azalt(self):
    
    return self.az,self.alt
      
  def get_sat_is_eclipsed(self):
  
    return self.eclipsed
    
  def get_num_sat(self):
  
    return self.num_sat 
      
  def get_traces (self,start_time, duration, time_step,  az_min, az_max, alt_min, alt_max, p_vis = 0.3):
    
    N = math.ceil(duration/time_step)+1
    # preallocate memory assuming we'll see only p_vis% of the satellites
    NM = max(int(p_vis*N*self.num_sat),N)
    altitudes=np.empty(NM, dtype=np.double)
    azimuth=np.empty(NM, dtype=np.double)
    cnt = 0
    curr_sz = NM
    c_time = su.get_time_in_pyephem_format(start_time)
    idx_n = np.arange(c_time, c_time+duration+time_step, time_step)
    idx_m = range(self.num_sat)

    for s_time in idx_n:
      self.observer.date = s_time
      for sat_j in self.sat_list:
        sat_j.compute(self.observer)
        in_coord = (sat_j.az >= az_min) & (sat_j.az <= az_max) & (sat_j.alt >= alt_min) & (sat_j.alt <= alt_max) 
        if ((sat_j.eclipsed == False) & in_coord):
          altitudes[cnt] = sat_j.alt
          azimuth[cnt] = sat_j.az
          cnt += 1
        if cnt == curr_sz:
          altitudes = np.concatenate((altitudes,np.empty(NM,dtype=np.double)))
          azimuth = np.concatenate((azimuth,np.empty(NM,dtype=np.double)))
          curr_sz = len(altitudes)
              
    # sizedown the arrays if needed
    rm = curr_sz-cnt
    if rm > 0:
      altitudes = altitudes[:-rm]
      azimuth = azimuth[:-rm]
      
    return azimuth, altitudes 
    


#
#





















