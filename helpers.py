# -*- coding: utf-8 -*-
"""
Created on Thu Jan 29 22:09:56 2026

@author: JP Hooks
"""

import uuid, inspect, random, math
from typing import Tuple, List
import numpy as np
from math import ceil, sqrt
from scipy.ndimage import rotate, gaussian_filter
import matplotlib.pyplot as plt
from skimage import exposure
from skimage.draw import polygon, disk
from defect_const import *

def _uid(prefix): #create unique defect identifier for metadata
    return f"{prefix}_{uuid.uuid4().hex[:8]}"

def deg2rad(d):
    return float(d)*np.pi/180

def rad2deg(r):
    return float(r)*180/np.pi

def clip(val, lower, upper):
    return min(upper, max(val, lower))

def rand_lines_mask(n_cracks=None, angle=ALLOWED_ANGLES_DEG, rng=None):
    if rng is None:
        rng = np.random.default_rng()
    shape = (H_IMG,W_IMG)
    mask = np.zeros(shape, dtype=np.uint8)
    if n_cracks is None: n_cracks = rng.integers(1,5)
    for _ in range(n_cracks):
        x, y = rng.integers(0, W_IMG), rng.integers(0, H_IMG)
        length = rng.integers(CRACK_LENGTH_MIN, CRACK_LENGTH_MAX)
        ang_jit = angle +- rng.uniform(0.0,0.25)
        x2 = int(x + length * np.cos(ang_jit))
        y2 = int(y + length * np.sin(ang_jit))
        cv2.line(mask, (x, y), (x2, y2), 255, thickness=1)
    return mask

def crop(array, out_size):
    H, W = array.shape
    assert H >= out_size and W >= out_size
    start_r = (H - out_size) // 2
    start_c = (W - out_size) // 2
    return array[start_r:start_r+out_size, start_c:start_c+out_size]

def rand_ang_steps(steps: int, irregularity: float, rng=None):
    if rng is None:
        rng = np.random.default_rng()
    angles = []
    lower = (2*np.pi/steps) - irregularity
    upper = (2*np.pi/steps) + irregularity
    cumsum = 0
    for i in range(steps):
        angle = rng.uniform(lower, upper)
        angles.append(angle)
        cumsum += angle
    cumsum /= 2*np.pi
    for i in range(steps): angles[i] /= cumsum
    return angles

def one2zero(mask):
    ones = np.count_nonzero(mask)
    total = mask.size
    if ones < total/2: 
        mask = 1 - mask
    else:
        mask = mask
    return mask

def gen_polygon(center, avg_radius, num_vertices, irregularity=0.4, spikiness=0.08, rng=None):
    if rng is None:
        rng = np.random.default_rng()
    if irregularity < 0 or irregularity > 1:
        raise ValueError("Irregularity must be between 0 and 1.")
    if spikiness < 0 or spikiness > 1:
        raise ValueError("Spikiness must be between 0 and 1.")
    irregularity *= 2 * np.pi / num_vertices
    spikiness *= avg_radius
    angle_steps = rand_ang_steps(num_vertices, irregularity)
    points = []
    angle = rng.uniform(0, 2*np.pi)
    for i in range(num_vertices):
        radius = clip(rng.normal(avg_radius, spikiness), 0, 2*avg_radius)
        point = (center[0] + radius * np.cos(angle), center[1] + radius * np.sin(angle))
        points.append(point)
        angle += angle_steps[i]
    return points
        
def pad_array(array, pad_val=0):
    H, W = array.shape
    assert H == W
    diag = ceil(sqrt(2)*H)
    pad_total = max(0, diag-H)
    pad_before = pad_total//2 
    pad_after = pad_total - pad_before
    padded = np.pad(array, ((pad_before, pad_after), (pad_before, pad_after)),
                    mode='constant', constant_values=pad_val)
    return padded, pad_before, pad_after

def rotate_mask(mask, angle_deg, out_size=IMG_H):
    m = mask.copy()
    padded, pad_before, pad_after = pad_array(m, pad_val=1)
    rotated = rotate(padded, angle_deg, reshape=True, order=0, mode='constant', cval=1)
    cropped = crop(rotated, out_size)
    bin_mask = (cropped > 0.5).astype(np.uint8)
    return bin_mask

def fibre_map(shape, ang_deg=None, ang_jit=0.02, rng=None): #edit ang_deg to desired fibre angle
    if rng is None:
        rng = np.random.default_rng()
    H, W = shape
    if ang_deg is None:
        ang_deg = int(rng.choice(ALLOWED_ANGLES_DEG))
    if ang_deg not in ALLOWED_ANGLES_DEG:
        raise ValueError("chosen_angle_deg must be one of " + str(ALLOWED_ANGLES_DEG))
    base_ang = deg2rad(ang_deg) % np.pi
    noise = gaussian_filter(rng.normal(scale=ang_jit, size=(H, W)).astype(np.float32), sigma=max(1.0, min(H, W) / 128.0))
    theta = (noise + np.full((H, W), base_ang, dtype=np.float32)) % np.pi
    return theta, ang_deg

def fibre_texture(shape, fibre_map, sigma_along=70.0, sigma_across=1.5, contrast=0.5, randomness=0.025, rng=None, pad_factor=1.3):
    if rng is None:
        rng = np.random.default_rng()
    H, W = shape
    maxdim = int(max(H, W) * pad_factor)
    if maxdim % 2 == 1:
        maxdim += 1
    noise = rng.normal(size=(maxdim, maxdim)).astype(np.float32)
    theta = float(np.mean(fibre_map))
    angle_deg = -np.degrees(theta)
    rot_noise = rotate(noise, angle_deg, reshape=False, order=1, mode='reflect')
    smoothed = gaussian_filter(rot_noise, sigma=(sigma_along, sigma_across), mode='reflect')
    rot_back = rotate(smoothed, -angle_deg, reshape=False, order=1, mode='reflect')
    cy = maxdim // 2
    cx = maxdim // 2
    y0 = int(cy - H//2)
    x0 = int(cx - W//2)
    texture_raw = rot_back[y0:y0+H, x0:x0+W]
    texture = texture_raw - float(np.min(texture_raw))
    mx = float(np.max(texture)) + 1e-12
    texture = texture / mx
    texture = 0.5 + contrast * (texture - 0.5)
    mx = float(np.max(texture)) + 1e-12
    noise_mult = rng.normal(scale=randomness, size=(H, W)).astype(np.float32)
    texture = np.clip(texture + noise_mult * texture, 0.0, 1.0)
    texture = np.clip(texture, 0.28, 0.78)
    return texture.astype(np.float32)




    
