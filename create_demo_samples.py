"""
LeafGuard AI - Demo Sample Image Generator
Generates realistic test images for quick hackathon evaluation and automated test suites:
1. sample_healthy.jpg       -> Accepted healthy potato leaf
2. sample_early_blight.jpg  -> Accepted early blight diseased leaf
3. sample_blurry.jpg        -> Rejected due to blur (Laplacian variance < 60)
4. sample_dark.jpg          -> Rejected due to extreme darkness
5. sample_ood_object.jpg    -> Rejected due to Out-of-Distribution features (non-leaf object)
"""

import os
import cv2
import numpy as np

import config

SAMPLES_DIR = os.path.join(config.BASE_DIR, 'static', 'images', 'samples')
os.makedirs(SAMPLES_DIR, exist_ok=True)


def generate_samples():
    rng = np.random.default_rng(101)

    # 1. Base Potato Leaf canvas (Crisp, High detail)
    # 400x400 image with realistic leaf gradient, oval contour, and veining
    leaf = np.full((400, 400, 3), (245, 248, 245), dtype=np.uint8)  # soft off-white background

    # Leaf silhouette: ellipse
    center = (200, 200)
    axes = (140, 175)
    angle = 15

    # Draw vibrant green leaf body
    cv2.ellipse(leaf, center, axes, angle, 0, 360, (38, 155, 62), -1)  # BGR green

    # Add leaf texture and internal chlorophyll variation
    for _ in range(800):
        rx, ry = rng.integers(70, 330, 2)
        if ((rx - 200)**2 / 130**2 + (ry - 200)**2 / 165**2) < 0.85:
            color = (int(rng.integers(25, 45)), int(rng.integers(135, 185)), int(rng.integers(40, 75)))
            cv2.circle(leaf, (rx, ry), int(rng.integers(2, 6)), color, -1)

    # Main central vein
    cv2.line(leaf, (160, 40), (240, 360), (70, 190, 100), 4)

    # Lateral secondary veins
    for y in range(80, 320, 30):
        # Left side vein
        cv2.line(leaf, (int(160 + (y - 40) * 0.25), y), (int(100 + (y - 80) * 0.2), y - 25), (70, 185, 100), 2)
        # Right side vein
        cv2.line(leaf, (int(160 + (y - 40) * 0.25), y), (int(290 - (y - 80) * 0.15), y - 20), (70, 185, 100), 2)

    # Save 1: Healthy Leaf
    healthy_path = os.path.join(SAMPLES_DIR, 'sample_healthy.jpg')
    cv2.imwrite(healthy_path, leaf)
    print(f"[OK] Generated {healthy_path}")

    # Save 2: Early Blight Leaf (adds concentric brown necrotic spots)
    blight_leaf = leaf.copy()
    lesion_centers = [(150, 160), (240, 210), (190, 270)]
    for lc in lesion_centers:
        for r, col in [(28, (20, 60, 110)), (20, (15, 45, 85)), (12, (10, 30, 65)), (5, (5, 20, 45))]:
            cv2.circle(blight_leaf, lc, r, col, -1)
    early_blight_path = os.path.join(SAMPLES_DIR, 'sample_early_blight.jpg')
    cv2.imwrite(early_blight_path, blight_leaf)
    print(f"[OK] Generated {early_blight_path}")

    # Save 3: Blurry Leaf (Heavy Gaussian blur to trigger blur quality rejection)
    blurry_leaf = cv2.GaussianBlur(leaf, (45, 45), 0)
    blurry_path = os.path.join(SAMPLES_DIR, 'sample_blurry.jpg')
    cv2.imwrite(blurry_path, blurry_leaf)
    print(f"[OK] Generated {blurry_path}")

    # Save 4: Dark Leaf (Heavily underexposed to trigger darkness rejection)
    dark_leaf = (leaf.astype(np.float32) * 0.08).astype(np.uint8)
    dark_path = os.path.join(SAMPLES_DIR, 'sample_dark.jpg')
    cv2.imwrite(dark_path, dark_leaf)
    print(f"[OK] Generated {dark_path}")

    # Save 5: Out-of-Distribution Object (Mechanical wrench / industrial pattern, metallic blue/gray, sharp rectangular geometry)
    ood_img = np.full((400, 400, 3), (230, 230, 235), dtype=np.uint8)
    # Metallic wrench shaft
    cv2.rectangle(ood_img, (180, 80), (220, 320), (160, 160, 170), -1)
    cv2.circle(ood_img, (200, 90), 50, (140, 140, 155), -1)
    cv2.circle(ood_img, (200, 90), 22, (230, 230, 235), -1)
    # Chrome highlight
    cv2.line(ood_img, (190, 100), (190, 310), (220, 220, 230), 4)
    # Diagonal hazard stripes
    for i in range(0, 400, 40):
        cv2.line(ood_img, (i, 0), (i + 40, 400), (40, 40, 180), 3)

    ood_path = os.path.join(SAMPLES_DIR, 'sample_ood_object.jpg')
    cv2.imwrite(ood_path, ood_img)
    print(f"[OK] Generated {ood_path}")

    print("[SUCCESS] All 5 demo sample images created.")


if __name__ == '__main__':
    generate_samples()
