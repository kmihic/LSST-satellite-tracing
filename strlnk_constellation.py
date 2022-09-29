import numpy as np
from astropy import units as u
from sat_utils import satellite_mean_motion

def init_starlink_pos(sizeup_x):
  
    altitudes = np.array([550, 1110, 1130, 1275, 1325, 345.6, 340.8, 335.9]) #in km
    inclinations = np.array([53.0, 53.8, 74.0, 81.0, 70.0, 53.0, 48.0, 42.0]) #in deg
    nplanes = np.array([72, 32, 8, 5, 6, 2547, 2478, 2493])
    sats_per_plane = np.array([22, 50, 50, 75, 75, 1, 1, 1])
    
        
   # altitudes = np.array([550, 1110])
   # inclinations = np.array([53.0, 53.8])
   # nplanes = np.array([10, 3])
   #sats_per_plane = np.array([2,1])
    
    M = len(altitudes)
    N = sizeup_x * M
    if sizeup_x <= 1:
      return altitudes, inclinations, nplanes, sats_per_plane    
    else:
        new_altitudes = np.empty(N,dtype=np.double)
        new_inclinations = np.empty(N,dtype=np.double)
        new_nplanes = np.empty(N,dtype=np.uint32)
        new_sat_pp = np.empty(N,dtype=np.uint32)
        cnt = 0
        for i in range(sizeup_x): #np.arange(sizeup_x,dtype=np.uint32):
          alt_i = i*20
          inc_i = i*3
          for j in range(M):
            new_altitudes[cnt] = altitudes[j]+alt_i
            new_inclinations[cnt] = inclinations[j]+inc_i
            new_nplanes[cnt] = nplanes[j]
            new_sat_pp[cnt] = sats_per_plane[j]
            cnt +=1
    return new_altitudes, new_inclinations, new_nplanes, new_sat_pp
    

def get_constellation(sizeup_x, seed=1234):  
    
    # want to have reproducable simulations
    # setting rnd seed the new and proper way.
    # Docs for np.random.seed, the description reads: This is a convenience, legacy function.
    # NumPy calls the global random seed which may affect other packages
    rng = np.random.default_rng(seed)    
    alt, inc, nplanes, sat_per_plane = init_starlink_pos(sizeup_x)
    N = len(alt)
    idx = np.arange(N,dtype=np.uint32)
    M = np.dot(nplanes, sat_per_plane)
    cnt = 0    
    mean_motion = np.empty(M,dtype=np.double)
    inclination = np.empty(M,dtype=np.double)
    right_asc = np.empty(M,dtype=np.double)
    mean_anomaly = np.empty(M,dtype=np.double)
    for i in idx:
      if sat_per_plane[i] == 1:
        # random placement for lower orbits
        mas = rng.uniform(0, 360, nplanes[i])
        raans = rng.uniform(0, 360, nplanes[i])
      else: 
        mas = np.linspace(0.0, 360.0, sat_per_plane[i], endpoint=False)
        mas += rng.uniform(0, 360, 1)
        raans = np.linspace(0.0, 360.0, nplanes[i], endpoint=False)
        mas, raans = np.meshgrid(mas, raans)
        mas, raans = mas.flatten(), raans.flatten()

      mm = satellite_mean_motion(alt[i]*u.km)
      sat_inc = inc[i]
      for j in range(len(mas)):#np.arange(len(mas)):
        mean_motion[cnt] = mm.value
        inclination[cnt] = sat_inc
        right_asc[cnt] = raans[j]
        mean_anomaly[cnt] = mas[j]
        cnt +=1
          
    return inclination, right_asc, mean_anomaly, mean_motion 
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
