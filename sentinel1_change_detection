
"""
InSAR change detection from Sentinel‑1 SAR KMZ files.

Input  : KMZ_PREV, KMZ_CURR – paths to the earlier and later KMZ files
Output : KMZ_OUT            – a new KMZ containing the change overlay (diff_overlay.png)

White pixels in the overlay denote a significant increase in radar backscatter
between the two acquisitions, which can indicate newly built structures or
other surface changes.
"""

import cv2
import os
import shutil
import tempfile
import zipfile
import numpy as np
import xml.etree.ElementTree as ET

# -----------------------------------------------------------------------------
# Parameters
# -----------------------------------------------------------------------------
KMZ_PREV = "path/to/first_SAR_image.kmz"
KMZ_CURR = "path/to/second_SAR_image.kmz"
KMZ_OUT  = "path/to/result.kmz"
THRESHOLD = 0.6  # ignore differences smaller than this value (0–1 range)

# -----------------------------------------------------------------------------
# Helper functions
# -----------------------------------------------------------------------------

def load_overlay_from_kmz(kmz_path: str):
    """Return (image, png_name) for the first PNG found inside *kmz_path*."""
    with zipfile.ZipFile(kmz_path, "r") as kmz:
        png_files = [n for n in kmz.namelist() if n.lower().endswith(".png")]
        if not png_files:
            raise ValueError(f"No PNG overlay found in {kmz_path}")
        png_name = png_files[0]
        with kmz.open(png_name) as f:
            img = cv2.imdecode(np.frombuffer(f.read(), np.uint8), cv2.IMREAD_GRAYSCALE)
            img = img.astype(np.float32) / 255.0  # scale to 0–1
    return img, png_name


def compute_change_mask(img_prev: np.ndarray,
                        img_curr: np.ndarray,
                        threshold: float = 0.3) -> np.ndarray:
    """Return a binary mask (uint8) where white marks significant positive change."""
    diff = img_curr - img_prev
    mask = np.zeros_like(diff, dtype=np.uint8)
    mask[diff >= threshold] = 255
    return mask


def save_png(img: np.ndarray, path: str) -> None:
    cv2.imwrite(path, img)


def update_kml_icon(kml_path: str, new_image_name: str) -> None:
    """Point every <GroundOverlay> icon in *kml_path* to *new_image_name*."""
    ET.register_namespace('', 'http://www.opengis.net/kml/2.2')
    ns = {'kml': 'http://www.opengis.net/kml/2.2'}

    tree = ET.parse(kml_path)
    root = tree.getroot()

    for overlay in root.findall('.//kml:GroundOverlay', ns):
        icon = overlay.find('kml:Icon', ns)
        if icon is not None:
            href = icon.find('kml:href', ns)
            if href is not None:
                href.text = new_image_name
        # Ensure an opaque colour tag exists so white is not treated as transparent
        if overlay.find('kml:color', ns) is None:
            color = ET.Element('{http://www.opengis.net/kml/2.2}color')
            color.text = 'ffffffff'  # AABBGGRR – fully opaque white
            overlay.insert(1, color)

    tree.write(kml_path, encoding="utf-8", xml_declaration=True)


def create_kmz_with_diff(base_kmz: str,
                         diff_png: str,
                         kmz_out: str,
                         overlay_name: str = "diff_overlay.png") -> None:
    """Clone *base_kmz* → *kmz_out* and replace its overlay with *diff_png*."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        # 1. Unzip the original KMZ
        with zipfile.ZipFile(base_kmz, "r") as kmz:
            kmz.extractall(tmp_dir)

        # 2. Insert the new overlay and update KML references
        shutil.copy(diff_png, os.path.join(tmp_dir, overlay_name))
        update_kml_icon(os.path.join(tmp_dir, "doc.kml"), overlay_name)

        # 3. Remove any other PNG overlays to avoid Google Earth caching issues
        for fname in os.listdir(tmp_dir):
            if fname.endswith(".png") and fname != overlay_name:
                os.remove(os.path.join(tmp_dir, fname))

        # 4. Zip everything back into a fresh KMZ
        with zipfile.ZipFile(kmz_out, "w", zipfile.ZIP_DEFLATED) as new_kmz:
            for root, _, files in os.walk(tmp_dir):
                for file_name in files:
                    abs_path = os.path.join(root, file_name)
                    rel_path = os.path.relpath(abs_path, tmp_dir)
                    new_kmz.write(abs_path, rel_path)

# -----------------------------------------------------------------------------
# Main execution block
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    # 1. Load SAR amplitude overlays from both KMZ files
    img_prev, _ = load_overlay_from_kmz(KMZ_PREV)
    img_curr, _ = load_overlay_from_kmz(KMZ_CURR)

    # 2. Ensure both images have identical dimensions
    if img_prev.shape != img_curr.shape:
        img_curr = cv2.resize(img_curr, (img_prev.shape[1], img_prev.shape[0]))

    # 3. Compute the change mask with the chosen threshold
    change_mask = compute_change_mask(img_prev, img_curr, threshold=THRESHOLD)

    # 4. Save the mask to a temporary PNG and build the output KMZ
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp_png:
        save_png(change_mask, tmp_png.name)
        create_kmz_with_diff(KMZ_CURR, tmp_png.name, KMZ_OUT)

    print(f"Done! New KMZ written to: {KMZ_OUT}")
