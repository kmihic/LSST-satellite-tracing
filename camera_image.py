from scipy import sparse
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import math
import ccd_mask as cm
import sat_utils as su


class Camera:

  def __init__(self, fast_processing=True, init_masks = True,
    fov_radii_deg=1.75, pix_per_cell_coarse = 5, num_ccd_w = 15, num_corner_ccd_w = 3, num_pix_per_ccd_w = 4e3):
    
    self.fast = fast_processing
    self.pix_per_cell_coarse = pix_per_cell_coarse    
    if init_masks:
      self.mask_f_f, self.mask_f_c, self.corner_gap = cm.get_masks(pix_per_cell_coarse)      
      self.__create_scale_down_operators()
      self.camera_image_size_pix = self.mask_f_f.get_shape()[0]
    self.num_ccd_w = num_ccd_w 
    self.corner_ccd_gap = num_corner_ccd_w
    self.field_radii = math.radians(fov_radii_deg)
    # get num of usable camera pixels
    # estimate loss due to telescope optics radii (otherwise we'll just remove corner ccd's from the total num!)
    #  estimation done by summing up dark pixels from radial and field masks
    #  note: radial mask is not used (any more) as line segments already start/end within a circle with
    #        radius field_radii. The get all the masks call cm.get_masks_all()
        # N_pix_outside_radii_c=self.mask_r_c.count_nonzero() - self.mask_r_c.multiply(self.mask_f_c).count_nonzero()
        # N_pix_outside_radii_f=self.mask_r_f.count_nonzero() - self.mask_r_f.multiply(self.mask_f_f).count_nonzero()
        # N_pix_outside_radii = N_pix_outside_radii_f + N_pix_outside_radii_c*pix_per_cell_coarse
        # N_ccd_camera_total = num_ccd_w**2 - 4*num_corner_ccd_w**2
        # N_pix_camera_total = int(N_ccd_camera_total * num_pix_per_ccd_w**2)
        # N_pix_camera_usable = N_pix_camera_total - N_pix_outside_radii
    
    self.num_pix_camera = 2968583000
    self.num_ccd_camera = num_ccd_w**2 - 4*num_corner_ccd_w**2
    self.n_trace_pix = -1
    self.n_trace_ccd = -1
    self.n_traces = -1
    self.l_traces = -1
    self.ccd_id = np.array([-1],dtype=np.int16)    
    self.ccd_trace_num = np.array([-1],dtype=np.uint8)   
  #return camera parameters
  def get_camera_pix_num(self):
    
    return  self.num_pix_camera
    
  def get_camera_ccd_num(self):
    
    return  self.num_ccd_camera
    
  def get_fov_radii(self):
    
    return self.field_radii
  
  # return satellite "damage" 
  # Total number of pixels
  def get_traces_pix_num(self):
    
    return self.n_trace_pix 
    
  # Total number of ccd's     
  def get_traces_ccd_num(self):
    
    return self.n_trace_ccd
    
  # ID's of affected ccd's
  def get_traces_ccd_id(self):
    
    return self.ccd_id
    
  # Number of traces per ccd 
  def get_traces_per_ccd(self):
    
    return self.ccd_trace_num
    
  # Total number of traces
  def get_traces_num(self):
    
    return self.n_traces
  
  # Total trace length
  def get_traces_len(self):
    
    return self.l_traces
  
  def take_fov_traces(self, s_ra, s_dec, e_ra, e_dec, t_ra, t_dec,plot_image = False, plot_filename = ''):
    
    # not produced by this funct call, set it to some imposible value
    self.ccd_id = np.array([-1],dtype=np.uint8)  
    self.ccd_trace_num = np.array([-1],dtype=np.uint8)
    self.n_trace_pix = -1
    self.n_trace_ccd = -1 
    # find intersections
    Ax, Ay, Bx, By, k = self.__get_intersection(s_ra, s_dec, e_ra, e_dec, t_ra, t_dec, self.field_radii, plot_intersect=plot_image, 
          plot_filename=plot_filename)       
    self.n_traces = np.size(Ax)
    self.l_traces = np.sum(np.sqrt((Ax-Bx)**2 + (Ay-By)**2))    
    
    return
  
  def take_image_ccd(self, s_ra, s_dec, e_ra, e_dec, t_ra, t_dec,
     plot_image = False, plot_image_title = 'Camera image with satellite traces (ccd level)',
     plot_filename = ''):
    
    # not produced by this funct call, set it to some imposible value
    self.n_trace_pix = -1
    # find num tracses and total length
    # Find intersections of line (A,B) segments and fov
    Ax, Ay, Bx, By, k = self.__get_intersection(s_ra, s_dec, e_ra, e_dec, t_ra, t_dec, self.field_radii)       
    self.n_traces = np.size(Ax)
    self.l_traces = np.sum(np.sqrt((Ax-Bx)**2 + (Ay-By)**2))    
    if self.n_traces == 0:
      self.n_trace_ccd = 0   
      self.ccd_id = np.array([],dtype=np.uint8)   
      self.ccd_trace_num = np.array([],dtype=np.uint8)  
      return 
    # get affected ccds
    Q = self.__get_traces_image(Ax, Ay, Bx, By, k, 1, 2*self.field_radii, self.num_ccd_w, self.corner_ccd_gap,
        eliminate_duplicates=True, plot_overlay=plot_image, plot_title=plot_image_title,plot_filename= plot_filename) 
    # number of affected ccds
    self.n_trace_ccd = Q.count_nonzero()
    # get indices of the affected ccds
    # first row, first ccd starts with index 0
    # recall that tere is no ccd's at corners (3x3 in size)
    indices = Q.indices
    ix = Q.indptr
    N = Q.get_shape()[0]
    ccd_ix = np.empty(np.size(indices),dtype=np.uint8)
    ccd_loss_top = 2*self.corner_ccd_gap**2
    N_row_bottom = N-self.corner_ccd_gap
    # it has to be a more elegant way to code this, but for now it's fine
    for row in range(0,N):
      if (row < self.corner_ccd_gap):
        col_ix = row*N + indices[ix[row]:ix[row+1]]
        col_ix -= 2*self.corner_ccd_gap*row+self.corner_ccd_gap
        ccd_ix[ix[row]:ix[row+1]] = col_ix
      elif(row >= N_row_bottom):   
        col_ix = row*N + indices[ix[row]:ix[row+1]]
        col_ix -= 2*self.corner_ccd_gap*(row-N_row_bottom) + self.corner_ccd_gap + ccd_loss_top
        ccd_ix[ix[row]:ix[row+1]] = col_ix
      else:
        col_ix = row*N + indices[ix[row]:ix[row+1]] - ccd_loss_top
        ccd_ix[ix[row]:ix[row+1]] = col_ix
    # id(s) of affected ccds    
    self.ccd_id = ccd_ix
    # num traces per ccd
    self.ccd_trace_num = Q.data
    
    return
  
    
  def take_image_pix(self, s_ra, s_dec, e_ra, e_dec, t_ra, t_dec, trace_width, 
            plot_image = False, plot_image_title = 'Camera image with satellite traces (pixel level)', 
            plot_filename_fov = '', plot_filename_wccd = '',
            plot_intersect = False, plot_intersect_filename = ''):    
    
    # not produced by this funct call, set it to some imposible value
    self.ccd_id = np.array([-1],dtype=np.uint8)  
    self.ccd_trace_num = np.array([-1],dtype=np.uint8)   
    # find num tracses and total length
    # Find intersections of line (A,B) segments and fov
    Ax, Ay, Bx, By, k = self.__get_intersection(s_ra, s_dec, e_ra, e_dec, t_ra, t_dec, self.field_radii, plot_intersect, plot_filename=plot_intersect_filename)       
    self.n_traces = np.size(Ax)
    self.l_traces = np.sum(np.sqrt((Ax-Bx)**2 + (Ay-By)**2))    
    if self.n_traces == 0:
      self.n_trace_ccd = 0
      self.n_trace_pix = 0      
      return 
    # find num of affected ccds
    field_w = 2*self.field_radii
    self.n_trace_ccd =  self.__get_traces_image(Ax, Ay, Bx, By, k, 1, field_w, self.num_ccd_w, self.corner_ccd_gap).count_nonzero() 
    # find affected pixels   
    im_size = int(self.camera_image_size_pix/self.pix_per_cell_coarse)
    tr_width = int(trace_width/self.pix_per_cell_coarse)
    corner_gap = int(self.corner_gap/self.pix_per_cell_coarse)
    Q_c = self.__get_traces_image(Ax, Ay, Bx, By, k, tr_width, field_w, im_size, corner_gap)
    if self.fast:
      Q = self.__apply_masks_approx(Q_c)
    else:
      im_size = self.camera_image_size_pix
      corner_gap = self.corner_gap
      tr_width = trace_width    
      Q_f = self.__get_traces_image(Ax, Ay, Bx, By, k, tr_width, field_w, im_size, corner_gap)
      Q = self.__apply_masks_exact(Q_f,Q_c)
    if plot_image:
      plt.rcParams.update({"text.usetex" : True})
      plt.figure() 
      plt.title(plot_image_title,fontsize=12)  
      plt.spy(Q,ms=.005,c='r',origin='lower')  
      radii=np.shape(Q)[0]/2
      theta = np.linspace(0, 2*np.pi, 100) 
      x = radii*np.cos(theta)
      y = radii*np.sin(theta)
      plt.plot(x+radii,y+radii,linewidth = 3.5, c = 'y')
      if len(plot_filename_fov)>0 & ~plot_filename_fov.isspace():
        plt.savefig(plot_filename_fov,bbox_inches='tight')
      plt.spy(self.mask_f_f, ms=.005,c='y',origin='lower') 
      if len(plot_filename_wccd)>0 & ~plot_filename_wccd.isspace():
        plt.savefig(plot_filename_wccd,bbox_inches='tight')
    self.n_trace_pix = Q.count_nonzero()
  
  def __create_scale_down_operators(self):
    
    m, n = self.mask_f_f.get_shape()
    L1=sparse.eye(m, dtype=np.uint8,format='csr')
    L2=sparse.eye(round(m/self.pix_per_cell_coarse), dtype=np.uint8,format='csr')
    R1=sparse.eye(n, dtype=np.uint8,format='csr')
    R2=sparse.eye(round(n/self.pix_per_cell_coarse), dtype=np.uint8,format='csr')
  
    self.L=su.repelem(L2,1,self.pix_per_cell_coarse) * L1
    self.R=(su.repelem(R2,1,self.pix_per_cell_coarse) * R1)
        
  # Returns intersections of line segments (start/end points) and fov. 
  # (all points in the nonnegative orthant)
  def __get_intersection(self, s_ra, s_dec, e_ra, e_dec, t_ra, t_dec, fov_r, plot_intersect=False, plot_title='Satellite traces', plot_filename=''):  
    
    # convert to cartesian system
    # In theory, the projection could introduce some distortion, say a satellite that just grazed 
    # the focal plane actually didn't (or vice versa). 
    # Ignoring them here, with a hope not much error is introduced.
    # Note that the field of view is already in 2D, so it can be applied directly without projecting it  
    ax, ay = su.gnomonic_project_toxy(s_ra, s_dec, t_ra, t_dec)
    bx, by = su.gnomonic_project_toxy(e_ra, e_dec, t_ra, t_dec)
             
    N = np.size(ax)    
    # get lin equ. for each segmet
    dx = bx - ax
    dy = by - ay
    # np.divide(N/0) returns 1 or inf, depends on numpy version
    # i.e. not a way to go! Doing it the old fashion way
    k = np.ones(N, dtype = np.double) * np.inf
    l = np.ones(N, dtype = np.double) * np.inf
    ix_no_vert = (dx!=0)
    k[ix_no_vert] = dy[ix_no_vert]/dx[ix_no_vert]
    l[ix_no_vert] = -k[ix_no_vert]*ax[ix_no_vert] + ay[ix_no_vert] 
    ix_vert = ~ix_no_vert    
    # get intersections
    #
    lx = np.ones(N, dtype = np.double) * np.inf 
    ly = np.ones(N, dtype = np.double) * np.inf 
    rx = np.ones(N, dtype = np.double) * np.inf 
    ry = np.ones(N, dtype = np.double) * np.inf    
    # find the line point closest to the circle center
    # Ax+By+c=0 => (x_0=-Ac/(A^2+B^2), y_0=-Bc/(A^2+B^2))
    #              d_0^2 = c^2/(A^2+B^2)
    # y = kx +l => A=k, B=-1, c=l  
    x0 = np.ones(N, dtype = np.double) * np.inf 
    y0 = np.ones(N, dtype = np.double) * np.inf 
    d0_sq = np.ones(N, dtype = np.double) * np.inf
    k_sq = k[ix_no_vert]**2
    ab_sq = (k_sq+1)
    x0[ix_no_vert] = -k[ix_no_vert]*l[ix_no_vert]/ab_sq
    y0[ix_no_vert] = l[ix_no_vert]/ab_sq
    x0[ix_vert] = ax[ix_vert]
    y0[ix_vert] = 0; 
    l_sq = l[ix_no_vert]**2
    d0_sq[ix_no_vert] = l_sq/ab_sq 
    d0_sq[ix_vert] = np.abs(ax[ix_vert]**2)
    # interested only in lines that cut the circle (i.e. two intersection points)
    # which happens if d0 < radius   
    r_sq = fov_r**2    
    has_two_pts = (d0_sq < r_sq) 
    if(np.size(has_two_pts) > 0):
      # find the distance from circle-line intersection and d_0
      d_sq = r_sq - d0_sq
      # "move" the origin to x0,y0 and find intersections coordinates
      # x_sq = d_sq*B^2/(A^2+B^2), y_sq = d_sq*A^2/(A^2+B^2)      
      ix = (has_two_pts & ix_no_vert)
      if np.any(ix):
        K = np.sqrt(d_sq[ix]/ab_sq[ix])
        # left pt
        lx[ix] = x0[ix] - K
        ly[ix] = y0[ix] -k[ix] * K
        # right pt
        rx[ix] = x0[ix] + K      
        ry[ix] = y0[ix] + k[ix] * K
      # handle vertical lines
      ix = (has_two_pts & ix_vert) 
      if np.any(ix):      
        K = np.sqrt(d_sq[ix])
        lx[ix] = x0[ix] 
        ly[ix] = y0[ix] - K
        ry[ix] = y0[ix] + K
        rx[ix] = x0[ix] 
    # put all together    
    x = np.array([ax, bx, lx, rx])
    y = np.array([ay, by, ly, ry])
    # now we should have 4 points per line 
    # sort and choose the inner two  
    ix_s=np.argsort(x,axis=0)  # recall: np returns row based indices sorting, unlike matlan which is Fortran like
    m = np.size(x,axis=0)
    n = np.size(x,axis=1)
    ix_arr = np.array([ix_s[1,:],np.arange(0,n)])
    ix = np.ravel_multi_index(ix_arr, (m,n))  
    Ax = np.ravel(x)[ix]
    Ay = np.ravel(y)[ix]
    ix_arr = np.array([ix_s[2,:],np.arange(0,n)])
    ix = np.ravel_multi_index(ix_arr, (m,n))
    Bx = np.ravel(x)[ix]
    By = np.ravel(y)[ix]
    # handle vertical lines    
    x = x[:,ix_vert]
    if np.size(x)>0:
      y = y[:,ix_vert]
      ix_s=np.argsort(y,axis=0)  
      m = np.size(x,axis=0)
      n = np.size(x,axis=1)
      ix_arr = np.array([ix_s[1,:],np.arange(0,n)])
      ix = np.ravel_multi_index(ix_arr, (m,n))
      Ax[ix_vert] = np.ravel(x)[ix]
      Ay[ix_vert] = np.ravel(y)[ix]
      ix_arr = np.array([ix_s[2,:],np.arange(0,n)])
      ix = np.ravel_multi_index(ix_arr, (m,n))
      Bx[ix_vert] = np.ravel(x)[ix]
      By[ix_vert] = np.ravel(y)[ix]
    # remove lines not cutting the circle
    # we may have a false positives, i.e. the line segment does not cut, while its extension into a line does
    ix = has_two_pts & ix_no_vert
    Lx = np.minimum(Ax, Bx)
    Rx = np.maximum(Ax, Bx)
    seg_no_vert_ok = ~(((ax<=Lx) & (bx<=Lx)) | ((Rx<=ax) & (Rx<=bx))) & ix
    ix = has_two_pts & ix_vert
    Dy = np.minimum(Ay,By)
    Uy = np.maximum(Ay,By)
    seg_vert_ok = ~(((ay<=Dy) & (by<=Dy)) | ((Uy<=ay) & (Uy<=by))) & ix
    seg_ok = seg_no_vert_ok | seg_vert_ok
    Ax = Ax[seg_ok]
    Ay = Ay[seg_ok]
    Bx = Bx[seg_ok]
    By = By[seg_ok]
    k = k[seg_ok]
    # plot processed
    if plot_intersect:
      plt.figure()
      su.plot_intersect(Ax,Ay,Bx,By,-fov_r,-fov_r,fov_r,fov_r,plot_title,fov_r)
      if len(plot_filename)>0 & ~plot_filename.isspace():
        plt.savefig(plot_filename)#,bbox_inches='tight')
    # move the poins to the nonnegative orthant
    Ax += fov_r
    Ay += fov_r
    Bx += fov_r
    By += fov_r
    return Ax, Ay, Bx, By, k   
  
  # assumption: input points are cartesian and within camera field of view  
  def __get_traces_image(self, Ax, Ay, Bx, By, k, tr_width, field_w, N, corner_gap, eliminate_duplicates=False,
     plot_overlay = False, plot_title='Camera image with satellite traces',plot_filename = ''): 
    
    # normalize the coordinates
    dxy = field_w / N;
    Ax = Ax / dxy
    Ay = Ay / dxy
    Bx = Bx / dxy
    By = By / dxy 
    # evaluate across x-axis 
    if tr_width > 1:
      w = math.floor(tr_width / 2)
      W = np.arange(-w, -w + tr_width)
      ax = np.repeat(Ax, repeats=tr_width)
      bx = np.repeat(Bx, repeats=tr_width)
      k = np.repeat(k, repeats=tr_width)
      ww = np.tile(W, (np.size(Ay), 1))
      ay = ww + Ay[:, None]
      ay = ay.flatten()
      by = ww + By[:, None]
      by = by.flatten()  
      aay = np.repeat(Ay, repeats=tr_width)
      bby = np.repeat(By, repeats=tr_width) 
      aax = ax
      bbx = bx
    else:
      ax = Ax 
      ay = Ay
      aax = Ax
      aay = Ay
      bx = Bx 
      by = By 
      bbx = Bx
      bby = By 
    M = np.size(ax)
    x1 = np.tile(np.arange(0., N), (M, 1))
    y1 = x1 * k[:, None] + (ay - ax * k)[:, None]
    # limit axis to be inside of [a,b]
    ix_x1 = np.empty((M,N), dtype=np.bool)
    ix_y1 = np.empty((M,N), dtype=np.bool)
    for ii in range(0, M):
      # can get parts of thick traces outside of fov
      min_x = max((math.floor(min(aax[ii], bbx[ii]))), 0)
      max_x = min(max(aax[ii], bbx[ii]), N)
      min_y = max(math.floor(min(aay[ii], bby[ii])),0)
      max_y = min(math.ceil(max(aay[ii], bby[ii])), N)
      ix_x1[ii,:] = (x1[ii,:] >= min_x) & (x1[ii,:] <= max_x)
      ix_y1[ii,:] = (y1[ii,:] <= max_y) & (y1[ii,:] >=  min_y)
    
    # handle lines that intesect corners (of a "cell")
    k_neg = k < 0
    y1[~k_neg,:] = np.floor(y1[~k_neg,:]) 
    y1[k_neg,:] = np.ceil(y1[k_neg,:]) -1        
    # evaluate across y-axis
    # (otherwise we may not catch all cells for traces cutting vertically stack cells)
    if tr_width > 1:  # VERIFY
      ay = np.repeat(Ay, repeats=tr_width)
      by = np.repeat(By, repeats=tr_width)    
      ww = np.tile(W, (np.size(Ax), 1))
      ax = ww + Ax[:, None]
      ax = ax.flatten()
      bx = ww + Bx[:, None]
      bx = bx.flatten()     
    y2 = np.tile(np.arange(0., N), (M, 1))
    x2 = y2 / k[:, None] + (ax - ay / k)[:, None]
    y2[k_neg] -=1
    ix_x2 = np.empty((M,N), dtype=np.bool)
    ix_y2 = np.empty((M,N), dtype=np.bool)
    for ii in range(0, M):
      # can get parts of thick traces outside of fov
      min_y = max(math.floor(min(aay[ii], bby[ii])),0)
      max_y = min(max(aay[ii], bby[ii]),N)
      max_x = max(math.ceil(max(aax[ii], bbx[ii])),N)
      min_x = max(math.floor(min(aax[ii], bbx[ii])),0)
      ix_y2[ii,:] = (y2[ii,:] >= min_y) & (y2[ii,:] <= max_y)
      ix_x2[ii,:] = (x2[ii,:] <= max_x) & (x2[ii,:] >= min_x)
    x2[k_neg,:] = np.floor(x2[k_neg,:]) 
    x2[~k_neg,:] = np.ceil(x2[~k_neg,:]) -1
    
    # draw traces
    
    # the above procedure can produce "doubles", i.e. cells that are identified by both
    # x-y and y-x parts. In general, that is not a problem if we ask only to id cells that have at least one
    # trace over them (as it is a case fro pixels). For CCD's, we may be interested in number of lines that cross each
    # scipy.sparse.csr construction ...duplicate (i,j) entries will be summed together. This facilitates efficient construction of finite element matrices..." (see documentation)
    # As CSR format is used by this code, we need to remove "duplicates", using lil_matrix.
    # Note: this is much slower and more memory intensive process than the direct method
    
    ix1 = ix_x1 & ix_y1      
    ix2 = ix_x2 & ix_y2
    if eliminate_duplicates:
      ix = np.hstack((ix1, ix2))
      x = np.hstack((x1,x2))
      y = np.hstack((y1,y2))
      Q = sparse.lil_matrix((N,N), dtype=np.int8)
      for ii in range(0,M):
        q = sparse.lil_matrix((N,N), dtype=np.int8)
        q[y[ii,ix[ii,:]], x[ii,ix[ii,:]]] = 1
        Q += q
      # remo2ve trace parts that overlap areas with no ccd  
      Q[0:corner_gap,N-corner_gap:N]=0
      Q[0:corner_gap,0:corner_gap]=0
      Q[N-corner_gap:N, 0:corner_gap]=0
      Q[N-corner_gap:N,N-corner_gap:N,]=0  
      Q = Q.tocsr()
    else:        
      X = np.append(x1[ix1],x2[ix2])
      Y = np.append(y1[ix1],y2[ix2])
      # remo2ve trace parts that overlap areas with no ccd
      ix_ul = (X < corner_gap) & (Y < corner_gap)
      ix_ll = (X < corner_gap) & (Y >= N - corner_gap)
      ix_ur = (X >= N - corner_gap) & (Y < corner_gap)
      ix_lr = (X >= N - corner_gap) & (Y >= N - corner_gap)
      ix = ix_ul | ix_ll | ix_ur | ix_lr
      X = X[~ix]
      Y = Y[~ix]
      Q = sparse.csr_matrix((np.ones(np.size(X)), (Y.astype(dtype=np.uint32), 
         X.astype(dtype=np.uint32))), shape=(N, N), dtype=np.uint8)
    
    if plot_overlay:
      plt.rcParams.update({"text.usetex" : True})
      plt.figure() 
      q=Q.astype(dtype=np.float).A
      plt.imshow(q, cmap='gray_r', origin='lower',aspect='equal', extent=[0,N, 0,N] )
      su.plot_intersect(Ax,Ay,Bx,By,0,0,N,N, plot_title,N/2., offset=N/2,clr='r', linewidth=3)
      su.plot_intersect(Ax,Ay,Bx,By,0,0,N,N, plot_title,N/2., offset=N/2,clr='y', linewidth=2)
      hm = plt.pcolor(q, cmap='gray_r')
      plt.colorbar(hm)
      ax = plt.gca()
      rect = Rectangle((0,0),corner_gap, corner_gap,linewidth=1,edgecolor='r',facecolor='none')
      ax.add_patch(rect)
      rect = Rectangle((N-corner_gap,0),corner_gap, corner_gap,linewidth=1,edgecolor='r',facecolor='none')
      ax.add_patch(rect)
      rect = Rectangle((N-corner_gap,N-corner_gap),corner_gap, corner_gap,linewidth=1,edgecolor='r',facecolor='none')
      ax.add_patch(rect)
      rect = Rectangle((0,N-corner_gap),corner_gap, corner_gap,linewidth=1,edgecolor='r',facecolor='none')
      ax.add_patch(rect)  
      if len(plot_filename)>0 & ~plot_filename.isspace():
        plt.savefig(plot_filename,bbox_inches='tight')
  
    return Q

  def __apply_masks_exact(self, Q, q):  
  
    # apply the coarse mask
    q = q - q.multiply(self.mask_f_c)
    if q.count_nonzero()==0:
      return q
    # scale up the resolution  
    q = su.repelem(q,self.pix_per_cell_coarse,self.pix_per_cell_coarse)
    # use q as a mask on fine traces
    Q = Q.multiply(q)
    #apply the fine sensor mask
    Q=Q-Q.multiply(self.mask_f_f)
  
    return Q
    
  def __apply_masks_approx(self, q):  
    
    # apply the coarse mask
    q = q - q.multiply(self.mask_f_c)
    # scale up the resolution  
    if q.count_nonzero()==0:
      return q
    Q = su.repelem(q,self.pix_per_cell_coarse,self.pix_per_cell_coarse) 
    # apply the fine sensor mask
    Q=Q-Q.multiply(self.mask_f_f)
    
  
    return Q
    
