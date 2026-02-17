# -*- coding: utf-8 -*-
"""
Created on Thu Jan 29 22:37:55 2026

@author: JP Hooks
"""

from scipy.ndimage import gaussian_filter
import numpy as np
import matplotlib.pyplot as plt
from build_defects import make_porosity, make_delamination, make_crack, generate_clean
from helpers import *
from defect_const import *

rng = np.random.default_rng(DEFAULT_RNG_SEED)
print(DEFAULT_RNG_SEED)

def apply_porosity(rng=None):
    if rng is None:
        rng = np.random.default_rng(DEFAULT_RNG_SEED)
    fiber, ang_deg = generate_clean(rng=rng)
    porosity_mask = make_porosity(rng=rng, align_angle=ang_deg)
    result_porosity = porosity_mask.copy()
    result_porosity[porosity_mask == 1] = fiber[porosity_mask == 1]
    return result_porosity, porosity_mask, fiber, ang_deg

def apply_delam(rng=None):
    if rng is None:
        rng = np.random.default_rng(DEFAULT_RNG_SEED)
    fiber, ang_deg = generate_clean(rng=rng)
    delam_mask = make_delamination(rng=rng)
    result_delam = delam_mask.copy()
    result_delam[delam_mask == 1] = fiber[delam_mask == 1]
    return result_delam, delam_mask, fiber, ang_deg

def apply_crack(rng=None):
    if rng is None:
        rng = np.random.default_rng(DEFAULT_RNG_SEED)
    fiber, ang_deg = generate_clean(rng=rng)
    crack_mask = make_crack(rng=rng, angle=ang_deg)
    crack_mask = one2zero(mask=crack_mask)
    result_crack = crack_mask.copy()
    result_crack[crack_mask == 1] = fiber[crack_mask == 1]
    return result_crack, crack_mask, fiber, ang_deg




