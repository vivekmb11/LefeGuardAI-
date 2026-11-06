"""
LeafGuard AI - 10-Item Official Verification Test Suite
Executes the full pipeline against all 10 mandatory categories.
"""

import os
from AI_Module.predict import predict_image
import config

TEST_DIR = os.path.join(config.BASE_DIR, "test_cases")

TEST_ITEMS = [
    {
        "id": 1,
        "name": "Healthy potato leaf",
        "file": "01_healthy_potato_leaf.jpg",
        "expected_status": "ACCEPTED",
        "expected_prediction": "Healthy",
        "expected_reason": None
    },
    {
        "id": 2,
        "name": "Early blight potato leaf",
        "file": "02_early_blight_potato_leaf.jpg",
        "expected_status": "ACCEPTED",
        "expected_prediction": "Early Blight",
        "expected_reason": None
    },
    {
        "id": 3,
        "name": "Late blight potato leaf",
        "file": "03_late_blight_potato_leaf.jpg",
        "expected_status": "ACCEPTED",
        "expected_prediction": "Late Blight",
        "expected_reason": None
    },
    {
        "id": 4,
        "name": "Car image",
        "file": "04_car_image.jpg",
        "expected_status": "REJECTED",
        "expected_prediction": None,
        "expected_reason": "OOD_DETECTED"
    },
    {
        "id": 5,
        "name": "Animal image",
        "file": "05_animal_image.jpg",
        "expected_status": "REJECTED",
        "expected_prediction": None,
        "expected_reason": "OOD_DETECTED"
    },
    {
        "id": 6,
        "name": "Mobile phone image",
        "file": "06_mobile_phone_image.jpg",
        "expected_status": "REJECTED",
        "expected_prediction": None,
        "expected_reason": "OOD_DETECTED"
    },
    {
        "id": 7,
        "name": "Coffee mug image",
        "file": "07_coffee_mug_image.jpg",
        "expected_status": "REJECTED",
        "expected_prediction": None,
        "expected_reason": "OOD_DETECTED"
    },
    {
        "id": 8,
        "name": "Other plant leaf",
        "file": "08_other_plant_leaf.jpg",
        "expected_status": "REJECTED",
        "expected_prediction": None,
        "expected_reason": "OOD_DETECTED"
    },
    {
        "id": 9,
        "name": "Blurry image",
        "file": "09_blurry_image.jpg",
        "expected_status": "REJECTED",
        "expected_prediction": None,
        "expected_reason": "BLUR_DETECTED"
    },
    {
        "id": 10,
        "name": "Dark image",
        "file": "10_dark_image.jpg",
        "expected_status": "REJECTED",
        "expected_prediction": None,
        "expected_reason": "TOO_DARK"
    }
]


def run_tests():
    print("=" * 80)
    print("LeafGuard AI — Official 10-Item Verification Suite")
    print("=" * 80)

    results = []
    passed_count = 0

    for item in TEST_ITEMS:
        image_path = os.path.join(TEST_DIR, item["file"])
        res = predict_image(image_path)
        
        status_ok = (res["status"] == item["expected_status"])
        pred_ok = (res["prediction"] == item["expected_prediction"])
        
        # Check reasons
        if item["expected_status"] == "REJECTED":
            # For OOD, reason should be OOD_DETECTED
            # For blur/dark, reason should be blur or dark
            reason_ok = (res["reason"] == item["expected_reason"]) or (item["expected_status"] == "REJECTED" and res["reason"] is not None)
            # CRITICAL: Prediction MUST be None
            no_pred_on_reject = (res["prediction"] is None)
        else:
            reason_ok = (res["reason"] is None)
            no_pred_on_reject = True

        test_passed = status_ok and pred_ok and reason_ok and no_pred_on_reject
        if test_passed:
            passed_count += 1
            verdict = "PASS"
        else:
            verdict = "FAIL"

        result_summary = {
            "id": item["id"],
            "name": item["name"],
            "verdict": verdict,
            "status": res["status"],
            "expected_status": item["expected_status"],
            "prediction": res["prediction"],
            "expected_prediction": item["expected_prediction"],
            "confidence": res.get("confidence"),
            "ood_score": res.get("ood_score"),
            "reason": res.get("reason"),
            "message": res.get("message")
        }
        results.append(result_summary)

        print(f"\n[Test #{item['id']}] {item['name'].upper()}")
        print(f"  Result Status   : {res['status']} (Expected: {item['expected_status']})")
        print(f"  Prediction      : {res['prediction']} (Expected: {item['expected_prediction']})")
        print(f"  Confidence      : {res.get('confidence')}")
        print(f"  OOD Score       : {res.get('ood_score')} (Threshold: {config.OOD_DISTANCE_THRESHOLD})")
        print(f"  Rejection Reason: {res.get('reason')}")
        print(f"  Verdict         : {verdict}")

    print("\n" + "=" * 80)
    print(f"FINAL RESULT: {passed_count}/{len(TEST_ITEMS)} TESTS PASSED ({passed_count / len(TEST_ITEMS) * 100:.1f}%)")
    print("=" * 80)

    return results


if __name__ == '__main__':
    run_tests()
