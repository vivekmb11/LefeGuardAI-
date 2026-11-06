"""
LeafGuard AI - Evaluation on Real Held-Out Test Splits and OOD Categories
"""

import os
import glob
from AI_Module.predict import predict_image
import config

test_dir = os.path.join(config.BASE_DIR, "datasets", "Potato", "PLD_3_Classes_256", "Testing")
ood_dir = os.path.join(config.BASE_DIR, "datasets", "ODD")

print("=" * 65)
print("1. EVALUATING IN-DISTRIBUTION POTATO LEAVES (HELD-OUT TEST SET)")
print("=" * 65)

potato_results = {"Healthy": [], "Early Blight": [], "Late Blight": []}
for folder, expected in [("Healthy", "Healthy"), ("Early_Blight", "Early Blight"), ("Late_Blight", "Late Blight")]:
    paths = glob.glob(os.path.join(test_dir, folder, "*.*"))[:15]
    for p in paths:
        res = predict_image(p)
        potato_results[expected].append(res)
        print(f"[{expected}] {os.path.basename(p)} -> Status: {res['status']}, Pred: {res['prediction']}, Conf: {res['confidence']}, OOD: {res['ood_score']}")

print("\n" + "=" * 65)
print("2. EVALUATING OUT-OF-DISTRIBUTION (OOD) OBJECT CATEGORIES")
print("=" * 65)

ood_results = {}
for cat in ["car", "animals", "cup", "random_objects"]:
    files = glob.glob(os.path.join(ood_dir, cat, "**", "*.*"), recursive=True)
    imgs = [f for f in files if f.lower().endswith(('.jpg', '.jpeg', '.png'))][:10]
    ood_results[cat] = []
    for p in imgs:
        res = predict_image(p)
        ood_results[cat].append(res)
        print(f"[{cat.upper()}] {os.path.basename(p)} -> Status: {res['status']}, Reason: {res['reason']}, Pred: {res['prediction']}, OOD: {res['ood_score']}")

print("\n" + "=" * 65)
print("SUMMARY REPORT")
print("=" * 65)

for k, v in potato_results.items():
    accepted = [r for r in v if r['status'] == 'ACCEPTED']
    correct = [r for r in accepted if r['prediction'] == k]
    acc_pct = len(correct) / len(v) * 100 if v else 0
    print(f"Potato {k}: {len(correct)}/{len(v)} correct ({acc_pct:.1f}%) | Accepted: {len(accepted)}/{len(v)}")

total_ood = sum(len(v) for v in ood_results.values())
rejected_ood = sum(sum(1 for r in v if r['status'] == 'REJECTED' and r['prediction'] is None) for v in ood_results.values())
print(f"OOD Rejection Rate: {rejected_ood}/{total_ood} ({(rejected_ood / total_ood * 100) if total_ood else 0:.1f}%)")
print("CRITICAL CHECK: Were any OOD samples assigned a disease prediction?")
violations = [r for v in ood_results.values() for r in v if r['prediction'] is not None]
print(f"Violations (disease predicted on OOD): {len(violations)}")
