# -*- coding: utf-8 -*-
"""
Created on Thu Jan 29 11:23:38 2026

@author: JP Hooks
"""

import numpy as np
import uuid, inspect, random, cv2, re, glob
from scipy.stats import gamma
from scipy.ndimage import rotate
from skimage.draw import polygon
import matplotlib.pyplot as plt
from matplotlib.path import Path
from defect_const import *
from helpers import *
try: import porespy as ps
except Exception:
    print('Porespy is not installed. Please install using conda-forge in your venv.')
    exit(1)

def _load_background_image(rng):
    """Pick a random image from BACKGROUND_IMAGE_DIR, resize to (IMG_H, IMG_W),
    normalise to [0, 1] float32, and parse the fibre angle from the filename.
    Filename angle tokens: _ang0_, _ang90_, _ang45_, _ang-45_  (degrees).
    Returns (texture, ang_deg)."""
    exts = ("*.png", "*.jpg", "*.jpeg", "*.tif", "*.tiff", "*.bmp")
    paths = []
    for ext in exts:
        paths.extend(glob.glob(str(BACKGROUND_IMAGE_DIR) + "/" + ext))
    if not paths:
        raise FileNotFoundError(
            f"No images found in BACKGROUND_IMAGE_DIR: {BACKGROUND_IMAGE_DIR}"
        )
    idx = int(rng.integers(0, len(paths)))
    img_path = paths[idx]
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise IOError(f"cv2 could not read image: {img_path}")
    if img.shape != (IMG_H, IMG_W):
        img = cv2.resize(img, (IMG_W, IMG_H), interpolation=cv2.INTER_LINEAR)
    texture = img.astype(np.float32) / 255.0
    texture = np.clip(texture, 0.0, 1.0)
    # parse angle from filename, e.g. _ang90_ or _ang-45_
    m = re.search(r"_ang(-?\d+)_", img_path)
    if m:
        ang_deg = int(m.group(1))
        if ang_deg not in ALLOWED_ANGLES_DEG:
            ang_deg = 0
    else:
        ang_deg = random.choice(ALLOWED_ANGLES_DEG)
    return texture, ang_deg

def generate_clean(shape=(IMG_H, IMG_W), rng=None):
    if rng is None:
        rng = np.random.default_rng(DEFAULT_RNG_SEED)
    if USE_BACKGROUND_IMAGES:
        return _load_background_image(rng)
    theta_map, ang_deg = fibre_map(shape, rng=rng)
    texture = fibre_texture(shape, theta_map, rng=rng)
    texture = np.clip(texture, 0.0, 1.0).astype(np.float32)
    sign = random.choice([1,-1])
    if ang_deg == 0:
        texture = rotate(texture, (sign * 90), reshape=False, order=1, mode='reflect')
    else: texture = texture
    return texture, ang_deg

def make_porosity(shape=(IMG_H,IMG_W,1), severity=POR_SEV, n=N_CYL, 
                  phi_max=PHI_MAX, theta_max=THETA_MAX, length=POROSITY_LENGTH,
                  rng=None, seed=None, pad=0, align_angle=None):
    if rng is None:
        rng=np.random.default_rng()
    big_h, big_w = IMG_H, IMG_W
    shape_alt = (big_h,big_w,1)
    dist = gamma(a=1.99, loc=0.00, scale=3.0)
    r = dist.rvs(size=n, random_state=rng)
    while np.any((r < R_MIN) | (r > R_MAX)):
        bad = (r < R_MIN) | (r > R_MAX)
        r[bad] = dist.rvs(size=bad.sum(), random_state=rng)
    r_dist = np.round(r).astype(int)
    por_mask = np.zeros((big_h, big_w), dtype=bool)
    
    for i in range(n):
        r_i = r_dist[i]
        seed_i = int(rng.integers(0, 2**31 - 1))
        vol = ps.generators.cylinders(shape=shape_alt, r=r_i, ncylinders=2, 
                phi_max=phi_max, theta_max=theta_max, length=length, seed=seed_i)
        vol = np.asarray(vol)
        porosity_def_i = np.any(vol, axis=2) #convert to 2d by middle z-slice
        por_mask ^= porosity_def_i
    porosity_mask = one2zero(mask=por_mask)
    porosity_mask = rotate_mask(mask=porosity_mask, angle_deg=align_angle)
    return porosity_mask.astype(np.float32)

def make_delamination(shape=(IMG_H, IMG_W), n=None, center=None, avg_radius=None, rng=None):
    if rng is None:
        rng=np.random.default_rng()
    if center is None: #random center
        cx = rng.integers(int(0.2*IMG_W), int(0.7*IMG_W))
        cy = rng.integers(int(0.2*IMG_H), int(0.7*IMG_H))
        center = (cx,cy)
    if avg_radius is None:
        avg_radius = rng.uniform(low=DELAM_RADIUS_MIN, high=DELAM_RADIUS_MAX)
    if n is None: 
        n = rng.integers(8,16)
    poly_mask = np.zeros(shape, dtype=bool)
    pts = gen_polygon(center=center, avg_radius=avg_radius, num_vertices=n)
    xs = [p[0] for p in pts]
    ys = [p[1] for p in pts]
    rr, cc = polygon(ys, xs, shape=shape)
    poly_mask[rr,cc] = 1.0
    delam_mask = one2zero(mask=poly_mask)
    return delam_mask.astype(np.float32)

def make_crack(n_cracks=None, angle=None, rng=None):
    if rng is None:
        rng=np.random.default_rng()
    shape = (IMG_H,IMG_W)
    mask = np.zeros(shape, dtype=np.uint8)
    if angle == 0: angle = 90
    elif angle == 90: angle = 0
    angle = deg2rad(angle)
    if n_cracks is None: n_cracks = rng.integers(1,5)
    for _ in range(n_cracks):
        x, y = rng.integers(0, IMG_W), rng.integers(0, IMG_H)
        length = rng.integers(CRACK_LENGTH_MIN, CRACK_LENGTH_MAX)
        x2 = int(x + length * np.cos(angle))
        y2 = int(y + length * np.sin(angle))
        crack_mask = cv2.line(mask, (x, y), (x2, y2), 255, thickness=2)
    return crack_mask.astype(np.float32)
    

'''
u = make_delamination()
print(type(u))
print(u)

plt.figure(figsize=(4,4))
plt.imshow(u, cmap='gray', vmin=0, vmax=1)
plt.title("delam_mask")
plt.axis('off')
plt.show()

'''

    

    



















    
    