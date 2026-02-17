"""
Created on Mon Feb 16 13:54:59 2026
@author: JP Hooks
Description: 
"""

import os
from pathlib import Path
import numpy as np
import imageio
import csv

from apply_defect import apply_porosity, apply_delam, apply_crack
from defect_const import DEFAULT_RNG_SEED

OUT_DIR = Path("F:\\Data\\synthetic_data")
N_PER_DEFECT = 750
IMG_EXT = ".png"
DEFECTS = {
    "porosity": apply_porosity,
    "delam": apply_delam,
    "crack": apply_crack,
}

def ensure_dirs(base: Path, defect_name: str):
    p_clean = base / defect_name / "clean"
    p_def = base / defect_name / "defected"
    p_mask = base / defect_name / "mask"
    p_clean.mkdir(parents=True, exist_ok=True)
    p_def.mkdir(parents=True, exist_ok=True)
    p_mask.mkdir(parents=True, exist_ok=True)
    return p_clean, p_def, p_mask

def save_uint8_image(arr, path: Path):
    a = np.asarray(arr)
    if a.dtype != np.uint8:
        if a.max() <= 1.0:
            a = (np.clip(a, 0.0, 1.0) * 255.0).round().astype(np.uint8)
        else:
            a = np.clip(a, 0, 255).astype(np.uint8)
    imageio.imwrite(str(path), a)
def main():
    rng_master = np.random.default_rng(DEFAULT_RNG_SEED)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    metadata_path = OUT_DIR / "metadata.csv"

    # Create CSV and write header
    with open(metadata_path, mode="w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow([
            "filename",
            "defect_type",
            "seed",
            "fiber_angle_deg"
        ])

        for defect_name, apply_fn in DEFECTS.items():
            print(f"Generating {N_PER_DEFECT} samples for: {defect_name}")
            p_clean, p_def, p_mask = ensure_dirs(OUT_DIR, defect_name)

            for i in range(N_PER_DEFECT):
                seed_i = int(rng_master.integers(0, 2**31 - 1))
                rng = np.random.default_rng(seed_i)

                defected_img, mask, clean_img, ang_deg = apply_fn(rng=rng)

                clean = np.asarray(clean_img).astype(np.float32)
                defimg = np.asarray(defected_img).astype(np.float32)
                mask = np.asarray(mask)

                mask_u8 = (mask > 0).astype(np.uint8) * 255

                clean_name = f"pristine_{defect_name}_{seed_i}{IMG_EXT}"
                defect_name_file = f"defect_{defect_name}_{seed_i}{IMG_EXT}"
                mask_name = f"mask_{defect_name}_{seed_i}{IMG_EXT}"

                clean_path = p_clean / clean_name
                def_path = p_def / defect_name_file
                mask_path = p_mask / mask_name

                save_uint8_image(clean, clean_path)
                save_uint8_image(defimg, def_path)
                save_uint8_image(mask_u8, mask_path)

                # Write metadata row
                writer.writerow([
                    str(clean_path.relative_to(OUT_DIR)),
                    0,
                    "none",
                    seed_i,
                    ang_deg
                ])
                writer.writerow([
                    str(def_path.relative_to(OUT_DIR)),
                    1,
                    defect_name,
                    seed_i,
                    ang_deg
                ])

                if (i + 1) % 50 == 0:
                    print(f"  {i+1}/{N_PER_DEFECT} complete")

    print("Dataset generation complete.")
    print("Metadata saved to:", metadata_path.resolve())
    
if __name__ == "__main__":
    main()

