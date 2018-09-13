
import h5py
import numpy as np

import os
from nonLinCorr import *

import sys
run = int(sys.argv[1])

def unflatten_shots(flat_shots,mask):
    num_shots = flat_shots.shape[0]
    shape = mask.shape
    flat_mask = mask.reshape(shape[0]*shape[1]
                            )
    shots = np.zeros( (num_shots,shape[0]*shape[1]), dtype = flat_shots.dtype)
    shots[:, flat_mask ] = flat_shots
    
    return shots.reshape( (num_shots,shape[0],shape[1]))

# load coefs
f=h5py.File('/reg/d/psdm/cxi/cxilr6716/results/flatfield_calibration/copper_cali_coefs.h5','r')
mask=np.load('/reg/d/psdm/cxi/cxilr6716/results/masks/basic_psana_mask.npy')

coef_keys=[kk for kk in f.keys() if kk.startswith('coefs')]

# calibrate some single protein shots
single_int_dir = '/reg/d/psdm/cxi/cxilr6716/scratch/flatfield_calibration/flat_det_imgs/'
f_imgs = h5py.File(os.path.join(single_int_dir,'fullImgs_run%d.h5'%run))

single_shots = f_imgs['flat_img'][:]
single_shots_int = single_shots.mean(-1)
selection_mask = (single_shots_int>1.3)*(single_shots_int<95.7)
single_shots=single_shots[selection_mask]
single_shots_int=single_shots_int[selection_mask]

print ("calibrating %d shots..."%single_shots_int.size)

f_out = h5py.File('/reg/d/psdm/cxi/cxilr6716/results/flatfield_calibration/solution_shots_cali2.h5','w')


for kk in coef_keys:
    
    cn = f[kk].value
    c = lambda(i): polyVal(cn,i)
    deg=cn.shape[0]-1
    print ("poly fit degree %d"%deg)
    calibrated_shots = single_shots*c(single_shots_int)
    # unflatten with mask and save the results
    f_out.create_dataset('cali_single_shots_%d'%deg,
                         data=unflatten_shots( calibrated_shots, mask) )

f_out.close()
print("done!")