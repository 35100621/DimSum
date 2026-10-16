import torch
from victim.detector import VictimDetector


# CNN-Aug ResNet34 checkpoint
path = r"C:\Users\Jin Xuan\Desktop\FIT3183\DeepfakeBench\training\weights\cnnaug_best.pth"


# Load detector
detector = VictimDetector(
    deepfakebench_root=r"C:\Users\Jin Xuan\Desktop\FIT3183\DeepfakeBench",
    detector_name="resnet34",
    weights_path=path,
    device="cpu"
)

print("SUCCESS: CNN-Aug ResNet loaded")


# ==========================
# Test image
# ==========================

image_path = r"C:\Users\Jin Xuan\Desktop\FIT3183\DimSum\real.jpg"


result = detector.predict_path(image_path)


print("\n========== RESULT ==========")
print("Label:", result["label"])
print("REAL probability:", result["real_probability"])
print("FAKE probability:", result["fake_probability"])
print("============================")