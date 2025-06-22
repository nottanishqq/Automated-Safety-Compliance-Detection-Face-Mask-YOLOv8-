import os
import shutil
import yaml
from pathlib import Path
from ultralytics import YOLO
import random
import torch

def create_dataset_yaml():
    """Create dataset.yaml file for YOLO training"""
    dataset_config = {
        'path': './dataset',  # dataset root dir
        'train': 'images/train',  # train images (relative to 'path')
        'val': 'images/val',  # val images (relative to 'path')
        'test': 'images/test',  # test images (optional)
        
        'nc': 2,  # number of classes
        'names': ['with_mask', 'without_mask']  # class names
    }
    
    with open('dataset.yaml', 'w') as f:
        yaml.dump(dataset_config, f, default_flow_style=False)
    
    return dataset_config

def organize_dataset():
    """Organize the dataset into train/val/test splits"""
    source_dir = Path('Labeled Mask Dataset/obj')
    dataset_dir = Path('dataset')
    
    # Create directory structure
    for split in ['train', 'val', 'test']:
        (dataset_dir / 'images' / split).mkdir(parents=True, exist_ok=True)
        (dataset_dir / 'labels' / split).mkdir(parents=True, exist_ok=True)
    
    # Get all image files
    image_files = list(source_dir.glob('*.jpg'))
    random.shuffle(image_files)
    
    # Split dataset: 70% train, 20% val, 10% test
    n_total = len(image_files)
    n_train = int(0.7 * n_total)
    n_val = int(0.2 * n_total)
    
    train_files = image_files[:n_train]
    val_files = image_files[n_train:n_train + n_val]
    test_files = image_files[n_train + n_val:]
    
    print(f"Dataset split: {len(train_files)} train, {len(val_files)} val, {len(test_files)} test")
    
    # Copy files to appropriate directories
    for split_name, files in [('train', train_files), ('val', val_files), ('test', test_files)]:
        for img_file in files:
            # Copy image
            shutil.copy2(img_file, dataset_dir / 'images' / split_name / img_file.name)
            
            # Copy corresponding label file
            label_file = img_file.with_suffix('.txt')
            if label_file.exists():
                shutil.copy2(label_file, dataset_dir / 'labels' / split_name / label_file.name)

def train_yolov8():
    """Train YOLOv8 model with data augmentation"""
    
    # Create dataset configuration
    create_dataset_yaml()
    
    # Organize dataset
    organize_dataset()
    
    # Initialize YOLOv8 model
    model = YOLO('yolov8n.pt')  # Load pretrained YOLOv8 nano model
    
    # Check if CUDA is available and set appropriate device and batch size
    if torch.cuda.is_available():
        device = 0  # Explicitly use GPU device 0
        batch_size = 8   # Reduced batch size for GTX 1650's 4GB VRAM
        print("✅ CUDA available - using GPU for training")
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    else:
        device = 'cpu'  # Use CPU if no GPU
        batch_size = 4   # Smaller batch size for CPU
        print("⚠️  CUDA not available - using CPU for training (this will be slower)")
    
    # Training configuration with data augmentation
    training_config = {
        'data': 'dataset.yaml',
        'epochs': 30,  # Reduced from 100 to 30 for faster training
        'imgsz': 640,
        'batch': batch_size,
        'device': device,  # Use appropriate device
        'workers': 4 if device == 'cpu' else 8,  # Fewer workers for CPU
        'patience': 15,  # Early stopping patience (reduced from 50)
        'save': True,
        'save_period': 5,  # Save every 5 epochs (reduced from 10)
        'cache': False,
        'project': 'face_mask_detection',
        'name': 'yolov8_face_mask',
        'exist_ok': True,
        
        # Data augmentation parameters
        'hsv_h': 0.015,  # HSV-Hue augmentation (fraction)
        'hsv_s': 0.7,    # HSV-Saturation augmentation (fraction)
        'hsv_v': 0.4,    # HSV-Value augmentation (fraction)
        'degrees': 0.0,  # Image rotation (+/- deg)
        'translate': 0.1,  # Image translation (+/- fraction)
        'scale': 0.5,    # Image scale (+/- gain)
        'shear': 0.0,    # Image shear (+/- deg)
        'perspective': 0.0,  # Image perspective (+/- fraction), range 0-0.001
        'flipud': 0.0,   # Image flip up-down (probability)
        'fliplr': 0.5,   # Image flip left-right (probability)
        'mosaic': 1.0,   # Image mosaic (probability)
        'mixup': 0.0,    # Image mixup (probability)
        'copy_paste': 0.0,  # Segment copy-paste (probability)
    }
    
    # Start training
    print("Starting YOLOv8 training with data augmentation...")
    print(f"Device: {device}, Batch size: {batch_size}")
    results = model.train(**training_config)
    
    # Print final metrics
    print("\n=== Training Results ===")
    print(f"Best mAP@0.5: {results.results_dict.get('metrics/mAP50(B)', 'N/A')}")
    print(f"Best mAP@0.5:0.95: {results.results_dict.get('metrics/mAP50-95(B)', 'N/A')}")
    print(f"Best Precision: {results.results_dict.get('metrics/precision(B)', 'N/A')}")
    print(f"Best Recall: {results.results_dict.get('metrics/recall(B)', 'N/A')}")
    
    return model, results

def validate_model(model_path):
    """Validate the trained model on test set"""
    model = YOLO(model_path)
    
    # Run validation
    results = model.val(data='dataset.yaml', split='test')
    
    print("\n=== Validation Results ===")
    print(f"mAP@0.5: {results.box.map50}")
    print(f"mAP@0.5:0.95: {results.box.map}")
    print(f"Precision: {results.box.mp}")
    print(f"Recall: {results.box.mr}")
    
    return results

if __name__ == "__main__":
    # Set random seed for reproducibility
    random.seed(42)
    
    # Train the model
    model, training_results = train_yolov8()
    
    # Validate the model
    best_model_path = 'face_mask_detection/yolov8_face_mask/weights/best.pt'
    if os.path.exists(best_model_path):
        validate_model(best_model_path)
    else:
        print(f"Best model not found at {best_model_path}") 