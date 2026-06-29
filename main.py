import cv2
import torch
import numpy as np
import torch.nn.functional as F
from pathlib import Path

from classification import PaletteClassifier
from classification.visualizer import save_result_figure
from visualize_seg import save_seg_figure
from src.models.system_1_deeplabv3 import DeepLabV3


# ================= CONFIG =================
IMAGE_PATH = "image/test.jpg"
CHECKPOINT_PATH = "checkpoints/system_1_deeplabv3.pth"

OUT_SEG = "image/result_seg.png"
OUT_CLASS = "image/result_classification.png"

_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
_STD = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)


# ================= LOAD MODEL =================
def load_model():
    ckpt = torch.load(CHECKPOINT_PATH, map_location="cpu")

    model = DeepLabV3(num_classes=11)
    model.load_state_dict(ckpt.get("model", ckpt))
    model.eval()

    return model


# ================= SEGMENTATION =================
def predict_seg(model, bgr):
    h, w = bgr.shape[:2]

    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    resized = cv2.resize(rgb, (473, 473))

    x = torch.from_numpy(resized).permute(2, 0, 1).float() / 255.0
    x = (x - _MEAN) / _STD
    x = x.unsqueeze(0)

    with torch.no_grad():
        out = model(x)
        logits = out["out"] if isinstance(out, dict) else out

        logits = F.interpolate(
            logits,
            size=(h, w),
            mode="bilinear",
            align_corners=False,
        )

        seg = logits.argmax(1).squeeze(0).numpy().astype(np.uint8)

    return seg


# ================= MAIN =================
def main():
    Path("results").mkdir(exist_ok=True)

    print("[1] Load image")
    bgr = cv2.imread(IMAGE_PATH)

    if bgr is None:
        raise ValueError("Cannot read image")

    print("[2] Load model")
    model = load_model()

    print("[3] Run segmentation")
    seg_mask = predict_seg(model, bgr)

    print("[4] Save segmentation result")
    save_seg_figure(
        face_bgr=bgr,
        seg_mask=seg_mask,
        output_path=OUT_SEG
    )

    print("[5] Run classification")
    clf = PaletteClassifier(hair_label=10)
    result = clf.classify(bgr, seg_mask)

    print("[6] Save classification result")
    save_result_figure(
        face_bgr=bgr,
        result=result,
        output_path=OUT_CLASS
    )

    print("\n===== DONE =====")
    print("Segmentation image:", OUT_SEG)
    print("Classification image:", OUT_CLASS)


if __name__ == "__main__":
    main()