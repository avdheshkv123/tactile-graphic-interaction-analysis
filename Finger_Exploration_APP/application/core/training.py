from pathlib import Path
from datetime import datetime
import yaml
from ultralytics import YOLO
import os
import random
import cv2
import numpy as np
import torch


# ================== RUN FOLDER ==================
def create_run_folder():
    base_dir = Path(__file__).resolve().parents[2] / "output" / "runs"

    run_name = datetime.now().strftime("run_%Y%m%d_%H%M%S")
    run_folder = base_dir / run_name
    models_folder = run_folder / "models"
    samples_folder = run_folder / "cropped_samples"

    models_folder.mkdir(parents=True, exist_ok=True)
    samples_folder.mkdir(parents=True, exist_ok=True)

    return {
        "run_folder": str(run_folder),
        "models_folder": str(models_folder),
        "samples_folder": str(samples_folder)
    }


# ================== SAFE OBB CROP ==================
def crop_and_upright_obb(image, polygon):
    rect = cv2.minAreaRect(polygon.astype(np.float32))
    (cx, cy), (w, h), angle = rect

    if w < h:
        angle += 90

    M = cv2.getRotationMatrix2D((cx, cy), angle, 1.0)
    rotated = cv2.warpAffine(image, M, (image.shape[1], image.shape[0]))

    # clamp coordinates (IMPORTANT FIX)
    x = int(max(0, cx - w / 2))
    y = int(max(0, cy - h / 2))
    w = int(w)
    h = int(h)

    x_end = min(rotated.shape[1], x + w)
    y_end = min(rotated.shape[0], y + h)

    crop = rotated[y:y_end, x:x_end]

    return crop


# ================== IMAGE COLLECTION ==================
def collect_images(dataset_root):
    dataset_root = Path(dataset_root)

    train_path = dataset_root / "train" / "images"
    val_path = dataset_root / "val" / "images"

    image_extensions = {".jpg", ".jpeg", ".png", ".bmp"}

    image_paths = []

    if train_path.exists():
        image_paths += [p for p in train_path.glob("*") if p.suffix.lower() in image_extensions]

    if val_path.exists():
        image_paths += [p for p in val_path.glob("*") if p.suffix.lower() in image_extensions]

    print(f"📸 Found {len(image_paths)} images total")

    return image_paths


# ================== SAMPLE PROCESS ==================
def process_random_images(dataset_root, model, output_dir, num_samples=25):
    image_paths = collect_images(dataset_root)

    if len(image_paths) == 0:
        print("❌ No images found for sampling")
        return []

    samples = random.sample(image_paths, min(num_samples, len(image_paths)))

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    saved_paths = []

    for idx, img_path in enumerate(samples):
        img = cv2.imread(str(img_path))

        if img is None:
            print(f"⚠️ Skipping unreadable image: {img_path}")
            continue

        try:
            results = model(img)[0]
        except Exception as e:
            print(f"⚠️ YOLO failed on {img_path}: {e}")
            continue

        if results.obb is None or len(results.obb.xyxyxyxy) == 0:
            print(f"⚠️ No OBB detected: {img_path}")
            continue

        polygon = results.obb.xyxyxyxy[0].cpu().numpy()

        crop = crop_and_upright_obb(img, polygon)

        if crop is None or crop.size == 0:
            print(f"⚠️ Empty crop: {img_path}")
            continue

        save_path = output_dir / f"sample_{idx}.png"
        cv2.imwrite(str(save_path), crop)

        print(f"✅ Saved: {save_path}")

        saved_paths.append(str(save_path))

    return saved_paths


# ================== TRAIN + SAMPLE ==================
def train_model(
    dataset_yaml_path,
    base_model,
    epochs=100,
    batch_size=8,
    image_size=640,
    patience=20
):
    folders = create_run_folder()

    with open(dataset_yaml_path, "r") as f:
        data = yaml.safe_load(f)

    dataset_root = os.path.dirname(dataset_yaml_path)
    print(f"📂 Dataset root: {dataset_root}")

    data["path"] = dataset_root

    fixed_yaml_path = os.path.join(dataset_root, "temp_data.yaml")

    with open(fixed_yaml_path, "w") as f:
        yaml.dump(data, f)

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    print(f"⚙️ Using device: {device}")

    model = YOLO(base_model)

    # ================== TRAIN ==================
    results = model.train(
        data=fixed_yaml_path,
        epochs=epochs,
        imgsz=image_size,
        batch=batch_size,
        patience=patience,
        device=device,
        workers=2,
        project=folders["models_folder"],
        name="training_run",
        save=True,
        verbose=True
    )

    # ================== SAMPLE + CROP ==================
    print("\n🔍 Running sample extraction (OBB crop + rotation)...")

    saved_images = process_random_images(
        dataset_root=dataset_root,
        model=model,
        output_dir=folders["samples_folder"],
        num_samples=25
    )

    # ================== OUTPUT INFO ==================
    print("\n✅ TRAINING COMPLETE")
    print(f"📁 Run Folder: {folders['run_folder']}")
    print(f"📁 Model Folder: {folders['models_folder']}")
    print(f"📁 Cropped Samples Folder: {folders['samples_folder']}")
    print(f"🖼️ Samples Generated: {len(saved_images)}")

    return {
        "status": "success",
        "output_folder": folders["run_folder"],
        "model_folder": folders["models_folder"],
        "samples_folder": folders["samples_folder"],
        "num_samples_generated": len(saved_images),
        "results": results
    }