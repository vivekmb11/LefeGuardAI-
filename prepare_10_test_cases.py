"""
LeafGuard AI - Prepare the 10 Specific Verification Test Cases
1. Healthy potato leaf
2. Early blight potato leaf
3. Late blight potato leaf
4. Car image
5. Animal image
6. Mobile phone image
7. Coffee mug image
8. Other plant leaf
9. Blurry image
10. Dark image
"""

import os
import glob
import shutil
import cv2
import numpy as np

import config

TEST_DIR = os.path.join(config.BASE_DIR, "test_cases")
SAMPLES_DIR = os.path.join(config.BASE_DIR, "static", "images", "samples")
os.makedirs(TEST_DIR, exist_ok=True)
os.makedirs(SAMPLES_DIR, exist_ok=True)

POTATO_BASE = os.path.join(config.BASE_DIR, "datasets", "Potato", "PLD_3_Classes_256", "Testing")
ODD_BASE = os.path.join(config.BASE_DIR, "datasets", "ODD")


def setup_test_cases():
    print("Assembling 10 verification test cases...")

    # 1. Healthy potato leaf
    h_files = glob.glob(os.path.join(POTATO_BASE, "Healthy", "*.*"))
    if h_files:
        shutil.copyfile(h_files[0], os.path.join(TEST_DIR, "01_healthy_potato_leaf.jpg"))
        shutil.copyfile(h_files[0], os.path.join(SAMPLES_DIR, "sample_healthy.jpg"))
        print("[1] Healthy potato leaf ready.")

    # 2. Early blight potato leaf
    eb_files = glob.glob(os.path.join(POTATO_BASE, "Early_Blight", "*.*"))
    if eb_files:
        shutil.copyfile(eb_files[0], os.path.join(TEST_DIR, "02_early_blight_potato_leaf.jpg"))
        shutil.copyfile(eb_files[0], os.path.join(SAMPLES_DIR, "sample_early_blight.jpg"))
        print("[2] Early blight potato leaf ready.")

    # 3. Late blight potato leaf
    lb_files = glob.glob(os.path.join(POTATO_BASE, "Late_Blight", "*.*"))
    if lb_files:
        shutil.copyfile(lb_files[0], os.path.join(TEST_DIR, "03_late_blight_potato_leaf.jpg"))
        shutil.copyfile(lb_files[0], os.path.join(SAMPLES_DIR, "sample_late_blight.jpg"))
        print("[3] Late blight potato leaf ready.")

    # 4. Car image
    car_files = glob.glob(os.path.join(ODD_BASE, "car", "**", "*.jpg"), recursive=True)
    if car_files:
        shutil.copyfile(car_files[0], os.path.join(TEST_DIR, "04_car_image.jpg"))
        shutil.copyfile(car_files[0], os.path.join(SAMPLES_DIR, "sample_car.jpg"))
        print("[4] Car image ready.")

    # 5. Animal image
    animal_files = glob.glob(os.path.join(ODD_BASE, "animals", "**", "*.jpg"), recursive=True)
    if animal_files:
        shutil.copyfile(animal_files[0], os.path.join(TEST_DIR, "05_animal_image.jpg"))
        shutil.copyfile(animal_files[0], os.path.join(SAMPLES_DIR, "sample_animal.jpg"))
        print("[5] Animal image ready.")

    # 6. Mobile phone image (clean realistic object render, 400x400)
    phone_img = np.full((400, 400, 3), (240, 240, 245), dtype=np.uint8)
    # Phone body (metallic dark titanium)
    cv2.rectangle(phone_img, (130, 60), (270, 340), (45, 45, 50), -1)
    # Screen (glossy black with OLED lock-screen widget)
    cv2.rectangle(phone_img, (136, 68), (264, 332), (18, 18, 20), -1)
    cv2.circle(phone_img, (200, 85), 6, (80, 80, 90), -1)  # dynamic island
    cv2.rectangle(phone_img, (160, 140), (240, 165), (200, 200, 220), 1)  # clock mockup
    # Save phone
    phone_path = os.path.join(TEST_DIR, "06_mobile_phone_image.jpg")
    cv2.imwrite(phone_path, phone_img)
    cv2.imwrite(os.path.join(SAMPLES_DIR, "sample_phone.jpg"), phone_img)
    print("[6] Mobile phone image ready.")

    # 7. Coffee mug image (normal high-resolution ceramic mug, 400x400)
    mug_img = np.full((400, 400, 3), (250, 248, 245), dtype=np.uint8)
    # Ceramic mug body (warm terracotta/ceramic red, 180x220)
    cv2.rectangle(mug_img, (120, 120), (260, 320), (50, 80, 190), -1)
    cv2.ellipse(mug_img, (190, 120), (70, 20), 0, 0, 360, (70, 100, 210), -1)
    # Coffee surface inside top
    cv2.ellipse(mug_img, (190, 122), (62, 16), 0, 0, 360, (30, 40, 65), -1)
    # Mug handle
    cv2.ellipse(mug_img, (260, 210), (35, 45), 0, 270, 90, (50, 80, 190), 16)
    # Bottom rim
    cv2.ellipse(mug_img, (190, 320), (70, 15), 0, 0, 180, (40, 65, 160), -1)
    mug_path = os.path.join(TEST_DIR, "07_coffee_mug_image.jpg")
    cv2.imwrite(mug_path, mug_img)
    cv2.imwrite(os.path.join(SAMPLES_DIR, "sample_cup.jpg"), mug_img)
    print("[7] Coffee mug image ready.")

    # 8. Other plant leaf (e.g. Acer / Maple palmate red/orange leaf - non-potato morphology)
    other_leaf = np.full((400, 400, 3), (245, 248, 245), dtype=np.uint8)
    # Draw star-like palmate lobes (distinct from solanaceous oval potato leaves)
    center = (200, 210)
    # 5 distinct maple lobes
    lobes = [
        [(200, 70), (180, 150), (220, 150)],    # top lobe
        [(110, 140), (150, 180), (160, 220)],   # left upper lobe
        [(290, 140), (250, 180), (240, 220)],   # right upper lobe
        [(120, 260), (160, 240), (180, 260)],   # left lower lobe
        [(280, 260), (240, 240), (220, 260)]    # right lower lobe
    ]
    for pts in lobes:
        cv2.fillPoly(other_leaf, [np.array(pts)], (25, 60, 200))  # Autumn red maple
    cv2.circle(other_leaf, center, 45, (25, 60, 200), -1)
    # Maple veins
    cv2.line(other_leaf, (200, 320), (200, 80), (15, 35, 120), 3)
    cv2.line(other_leaf, (200, 210), (120, 150), (15, 35, 120), 2)
    cv2.line(other_leaf, (200, 210), (280, 150), (15, 35, 120), 2)
    other_leaf_path = os.path.join(TEST_DIR, "08_other_plant_leaf.jpg")
    cv2.imwrite(other_leaf_path, other_leaf)
    cv2.imwrite(os.path.join(SAMPLES_DIR, "sample_other_leaf.jpg"), other_leaf)
    print("[8] Other plant leaf ready.")

    # 9. Blurry image (Apply heavy Gaussian blur on real potato leaf)
    if h_files:
        raw_leaf = cv2.imread(h_files[0])
        blurry = cv2.GaussianBlur(raw_leaf, (45, 45), 0)
        cv2.imwrite(os.path.join(TEST_DIR, "09_blurry_image.jpg"), blurry)
        cv2.imwrite(os.path.join(SAMPLES_DIR, "sample_blurry.jpg"), blurry)
        print("[9] Blurry image ready.")

    # 10. Dark image (Apply heavy underexposure on real potato leaf)
    if h_files:
        raw_leaf = cv2.imread(h_files[0])
        dark = (raw_leaf.astype(np.float32) * 0.08).astype(np.uint8)
        cv2.imwrite(os.path.join(TEST_DIR, "10_dark_image.jpg"), dark)
        cv2.imwrite(os.path.join(SAMPLES_DIR, "sample_dark.jpg"), dark)
        print("[10] Dark image ready.")

    print("\nAll 10 test case images successfully prepared!")


if __name__ == '__main__':
    setup_test_cases()
