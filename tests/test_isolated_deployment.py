"""Isolated Deployment Unit Test.

Simulates a clean Streamlit Cloud clone where cache/ and data/ are completely absent.
Verifies that app.py initializes, loads weights, processes sample images, and handles
custom images without any missing dependency or file errors.
"""

from pathlib import Path
from PIL import Image
import src.config

# Simulate clean cloud container: non-existent cache and data directories
src.config.PATHS["cache"] = Path("c:/simulated_empty_dir/cache")
src.config.PATHS["data"] = Path("c:/simulated_empty_dir/data")

import app

def test_isolated_app():
    print("[TEST 1] Loading system in isolated environment...")
    backbone, clf, T, means, cov, mean, std, thr_near_95, thr_far_95, thr_deploy = app.load_system()
    assert backbone is not None, "Backbone failed to initialize"
    assert clf is not None, "Classifier failed to initialize"
    assert abs(T - 0.9460) < 1e-3, f"Unexpected temperature {T}"
    assert abs(thr_deploy - (-3.6037)) < 1e-2, f"Unexpected deploy threshold {thr_deploy}"
    print(f"   -> System loaded successfully! T={T:.4f}, thr_deploy={thr_deploy:.4f}")

    print("[TEST 2] Testing sample image loading from figures/samples/...")
    samples = [
        "sample_known_airplane_0.png",
        "sample_known_cat_0.png",
        "sample_near_horse_0.png",
        "sample_far_ankle_boot_0.png",
    ]
    for fname in samples:
        sample_path = src.config.PATHS["figures"] / "samples" / fname
        assert sample_path.exists(), f"Missing sample file: {fname}"
        img = Image.open(sample_path).convert("RGB")
        res = app.classify_and_filter(img, backbone, clf, T, means, cov, mean, std, thr_deploy)
        print(f"   -> {fname:26s} | Pred: {res['pred']:10s} | Conf: {res['conf']*100:5.1f}% | Energy: {res['energy']:6.2f} | Rejected: {res['is_ood']}")

    print("[TEST 3] Testing arbitrary custom image upload handling...")
    custom_img = Image.new("RGB", (224, 224), color=(180, 120, 70))
    res_custom = app.classify_and_filter(custom_img, backbone, clf, T, means, cov, mean, std, thr_deploy)
    assert "pred" in res_custom and "energy" in res_custom and "is_ood" in res_custom
    print(f"   -> Custom Upload | Pred: {res_custom['pred']} | Energy: {res_custom['energy']:.2f}")

    print("\n[SUCCESS] ALL ISOLATED DEPLOYMENT TESTS PASSED 100%!")

if __name__ == "__main__":
    test_isolated_app()
