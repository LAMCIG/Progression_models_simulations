import numpy as np

import matplotlib
# %matplotlib qt5 


import pandas as pd
# import seaborn as sns
from typing import Tuple, Optional, List

from visbrain.gui import Brain
from visbrain.objects import BrainObj
from visbrain.io import download_file
from visbrain.objects import ColorbarObj

def visualize_brain_region_statistics(region_statistics_dataframe, 
                                      colormap = None,
                                      cbar_string = None,
                                      p_val_threshold = 0.05):
    
    if colormap is None:
        colormap = 'coolwarm'

    if cbar_string is None:
        cbar_string = 'Effect Size'

    Sig_ROIs_L = []
    Sig_ROIs_R = []

    D_val_L = []
    D_val_R = []

    
    for ind in region_statistics_dataframe.index:
        
        name = ind.split('_')
        hemi = name[0]
        ROI = name[1]
        
        stats = region_statistics_dataframe[region_statistics_dataframe.index==ind]
        p = float(stats['P-value'])
        d = float(stats['FX_size'])
        
        if(p < p_val_threshold):
            if(hemi=='L'):      #Left hemisphere - need to split hemispheres for display
                D_val_L.append(d)
                Sig_ROIs_L.append(ROI)
            if(hemi=='R'):      #Right hemisphere - need to split hemispheres for display
                D_val_R.append(d)
                Sig_ROIs_R.append(ROI)


    file_L = 'lh.aparc.annot'
    file_R = 'rh.aparc.annot'
    
    path_to_file_L = download_file(file_L, astype='example_data')
    path_to_file_R = download_file(file_R, astype='example_data')


    b_obj = BrainObj('inflated', hemisphere='both', translucent=False,
                    cblabel=cbar_string, cbtxtsz=4.)

    if colormap == 'coolwarm':
        c_max = np.max([np.max(np.abs(D_val_L)),np.max(np.abs(D_val_R))])
        c_min = -1.0*c_max

    c_max = np.max([np.max(D_val_L),np.max(D_val_R)])
    c_min = np.min([np.min(D_val_L),np.min(D_val_R)])

    b_obj.parcellize(path_to_file_L, clim=(c_min, c_max), hemisphere='left', select=Sig_ROIs_L, data=D_val_L,
                    cmap=colormap)

    b_obj.parcellize(path_to_file_R, clim=(c_min, c_max), hemisphere='right', select=Sig_ROIs_R, data=D_val_R,
                    cmap=colormap)

    vb = Brain(brain_obj=b_obj)
    vb.rotate(custom=(-100., 10.))  
    vb.show()


