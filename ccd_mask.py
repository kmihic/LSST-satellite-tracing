from scipy import sparse
import numpy as np
import matplotlib.pyplot as plt
import math
import sys

import time as tm


 # CCD grid
 #   "dark" areas between ccd's (i.e. no sensors) constitute a mask
 #
 # Requirements
 #   The actual delivered ccds slightly differ
 #   (imaging area width for x and y axis differ by a small amount)
 #   but for the purpose of this code those
 #   small deviances can be ignored
 #
 # Camera spec:
 # 5x5 raft
 # a raft:
 #   width 126.5 mm
 #   pitch 127 mm
 #   3x3 ccd sensors
 # a sensor
 #   width 42 mm
 #   imaging area width 40 mm
 #   pitch 42.25 mm
 #   1 pix == 10 um
 #
 # Assumptions: all dimensions are a multiple of the pix size

def get_masks(pix_per_cell_coarse):
  
  mask_field_fine, mask_field_coarse, field_fine_n, corner_gap = mask_ccd(pix_per_cell_coarse)
  
  return mask_field_fine, mask_field_coarse, corner_gap 
  

def mask_ccd(pix_per_cell_coarse, pixel_w = 10e-6, image_w = 40e-3, sensor_w = 42e-3,
      raft_w = 126.5e-3, raft_p = 127e-3, n_sensors = 3, n_rafts = 5):
        
  # Construct the coarse mask
  pixel_w_c = pixel_w * pix_per_cell_coarse
  sensor, n_masked = get_sensor(pixel_w_c, image_w, sensor_w, 1, 0)
  raft = get_raft(n_sensors, pixel_w_c, raft_w, sensor, 1)  
  fp_coarse = get_field(n_rafts, pixel_w_c, raft_p, raft,1)[0]
  n_used = n_masked * pix_per_cell_coarse;
  # Construct the fine mask
  sensor = get_sensor(pixel_w, image_w, sensor_w, 0,n_used )[0]  
  raft = get_raft(n_sensors, pixel_w, raft_w, sensor, 0)
  corner_gap = raft.get_shape()[0]
  fp_fine, fp_fine_out_ring_len = get_field(n_rafts, pixel_w, raft_p, raft,0)
  
  return fp_fine.tocsr(), fp_coarse.tocsr(), fp_fine_out_ring_len, corner_gap
  
#
# construct a sensor
#
def get_sensor(pixel_w, image_w, sensor_w, is_coarse, n_used_pix):

  i_pix = math.floor(image_w / pixel_w);
  s_pix = math.floor(sensor_w / pixel_w);  
  # number of no pixels (dark) per side
  # we have image sensors currounded by a "dark" frame
  d_pix = math.floor((s_pix - i_pix)/2);
  if is_coarse:
    d_pix = d_pix-1;
      
  sensor = sparse.lil_matrix((s_pix,s_pix), dtype=np.uint8)
  # mark cells with no imaging pixels
  if n_used_pix <= 0:
    sensor[0:d_pix,:] = 1
    sensor[s_pix-d_pix:s_pix,:] = 1
    sensor[:, 0:d_pix] = 1
    sensor[:, s_pix-d_pix:s_pix] = 1  
  else:
    # remove pixels already used in the coarse mask
    sensor[n_used_pix:d_pix,n_used_pix:s_pix-n_used_pix] = 1
    sensor[s_pix-d_pix:s_pix-n_used_pix,n_used_pix:s_pix-n_used_pix] = 1
    sensor[n_used_pix:s_pix-n_used_pix, n_used_pix:d_pix] = 1
    sensor[n_used_pix:s_pix-n_used_pix, s_pix-d_pix:s_pix-n_used_pix] = 1  
    
  return sensor.tocoo(), d_pix

  
def get_raft(n_sensors, pixel_w, raft_w, sensor, is_coarse):

  s_pix = sensor.get_shape()[0]
  r_pix = math.floor(raft_w / pixel_w)
  # "dark" area between the sensors
  d_p = (raft_w / pixel_w - n_sensors*s_pix)/(n_sensors-1);
  d_pix = math.floor(d_p);
  if d_pix != d_p:
    print('Change num pixels for coarse mask. Getting fractional space between sensors within raft\n')
    sys.exit()
  if is_coarse:
    d_sliver = sparse.coo_matrix(np.ones([s_pix,d_pix]),dtype=np.uint8)
  else:
    d_sliver = sparse.coo_matrix((s_pix,d_pix),dtype=np.uint8)
  rp = sensor
  for ii in range(n_sensors-1):
    rp = sparse.hstack([rp, d_sliver, sensor])
  if is_coarse:
    d_sliver = sparse.coo_matrix(np.ones([r_pix,d_pix]),dtype=np.uint8)
  else:
    d_sliver = sparse.coo_matrix((r_pix,d_pix),dtype=np.uint8)
  rp = rp.transpose()
  raft = rp
  for ii in range(n_sensors-1):
    raft = sparse.hstack([raft, d_sliver, rp])
 
  return raft


def get_field(n_rafts, pixel_w, raft_p, raft, is_coarse):
 
  nnz_r = raft.getnnz()
  r_pix = raft.get_shape()[0]
  # total field width 
  rp_pix = math.floor(raft_p / pixel_w)
  # "dark" area between the rafts  
  d_pix = rp_pix - r_pix 
  # no rafts on the corners
  raft_empty = sparse.coo_matrix(raft.get_shape(),dtype=np.uint8)  
  fp_pix_o = (n_rafts-2) * r_pix + (n_rafts-1) * d_pix
  # top/bottom rafts
  #fp_pix_o = (n_rafts-2) * r_pix + (n_rafts-1) * d_pix
  if is_coarse:
    d_sliver = sparse.coo_matrix(np.ones([r_pix,d_pix]),dtype=np.uint8)
  else:
    d_sliver = sparse.coo_matrix((r_pix,d_pix),dtype=np.uint8)
  fp_o = raft_empty
  for ii in range(n_rafts-2):
    fp_o = sparse.hstack([fp_o, d_sliver, raft])
  fp_o = sparse.hstack([fp_o, d_sliver, raft_empty])  
  # center rafts
  fp_c = raft
  for ii in range(n_rafts-1):
    fp_c = sparse.hstack([fp_c, d_sliver, raft])
  # assemble the field of view  
  f_pix = n_rafts * r_pix + (n_rafts-1) * d_pix;
  if is_coarse:
    d_sliver = sparse.coo_matrix(np.ones([d_pix,f_pix]),dtype=np.uint8)
  else:
    d_sliver = sparse.coo_matrix((d_pix,f_pix),dtype=np.uint8)
  fp = fp_o
  for ii in range(n_rafts-2):
    fp = sparse.vstack([fp, d_sliver, fp_c])
  fp = sparse.vstack([fp, d_sliver, fp_o])   

  return fp, fp_pix_o
  