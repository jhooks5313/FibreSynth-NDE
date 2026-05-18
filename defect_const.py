# -*- coding: utf-8 -*-
"""
Created on Thu Jan 29 11:50:17 2026

@author: JP Hooks
"""

import numpy as np
#image/grid defaults
IMG_H = 512
IMG_W = 512

#background image settings
USE_BACKGROUND_IMAGES = True       # set True to use real images instead of procedural textures
BACKGROUND_IMAGE_DIR  = r"C:\path\to\background_images"  # folder containing grayscale fibre images
# Angle is parsed from filenames containing '_ang0_', '_ang90_', '_ang45_', '_ang-45_'.
DX = 1.0  # mm/pixel
ALLOWED_ANGLES_DEG = [0, 90, 45, -45]
DEFAULT_RNG_SEED = 60

#severity ranges
SEVERITY_MIN = 0.1
SEVERITY_MAX = 0.98

#noise levels
GLOBAL_NOISE_STD_MIN = 0.005
GLOBAL_NOISE_STD_MAX = 0.04

#defect counts/sizes
MIN_DEFECTS = 0
MAX_DEFECTS = 3
DELT_POLY_VERTICES = 6-12

#delam defaults
DELAM_RADIUS_MIN = 20
DELAM_RADIUS_MAX = 160

#crack defaults
CRACK_LENGTH_MIN = 15
CRACK_LENGTH_MAX = 220
CRACK_THICKNESS = 1
N_CRACK_MIN = 1
N_CRACK_MAX = 5

#porosity defaults
R_MIN = 3
R_MAX = 9
N_CYL = np.random.randint(low=8, high=30)
POROSITY_LVL = np.random.uniform(low=0.83, high=0.96) #use in place of N_CYL is desired
PHI_MAX = 15
THETA_MAX = 5
POROSITY_LENGTH = 100
POR_SEED = 18
POR_SEV = 0.7
