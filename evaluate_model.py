import os
import json
import yaml
from pathlib import Path
from ultralytics import YOLO
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from datetime import datetime

def load_training_results(project_path):
    """Load training results from YOLOv8 output"""
    results_file = Path(project_path) / 'results.csv'
    if results_file.exists():
        import pandas as pd
        return pd.read_csv(results_file)
    return None

def evaluate_model_performance(model_path, dataset_yaml):
    """Evaluate model performance and generate detailed metrics"""
    
    # Load model
    model = YOLO(model_path)
    
    # Run validation on test set
    print("Running validation on test set...")
    results = model.val(data=dataset_yaml, split='test', save_json=True)
    
    # Extract metrics
    metrics = {
        'mAP50': results.box.map50,
        'mAP50_95': results.box.map,
        'precision': results.box.mp,
        'recall': results.box.mr,
        'f1_score': 2 * (results.box.mp * results.box.mr) / (results.box.mp + results.box.mr) if (results.box.mp + results.box.mr) > 0 else 0
    }
    
    print("\n=== Model Performance Metrics ===")
    print(f"mAP@0.5: {metrics['mAP50']:.4f}")
    print(f"mAP@0.5:0.95: {metrics['mAP50_95']:.4f}")
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"Recall: {metrics['recall']:.4f}")
    print(f"F1-Score: {metrics['f1_score']:.4f}")
    
    return metrics, results

def generate_per_class_metrics(results):
    """Generate per-class performance metrics"""
    
    # Get confusion matrix
    conf_matrix = results.confusion_matrix.matrix
    
    # Per-class metrics
    class_names = ['with_mask', 'without_mask']
    per_class_metrics = {}
    
    for i, class_name in enumerate(class_names):
        if i < conf_matrix.shape[0]:
            tp = conf_matrix[i, i]
            fp = conf_matrix[i, :].sum() - tp
            fn = conf_matrix[:, i].sum() - tp
            
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
            
            per_class_metrics[class_name] = {
                'precision': precision,
                'recall': recall,
                'f1_score': f1,
                'true_positives': tp,
                'false_positives': fp,
                'false_negatives': fn
            }
    
    print("\n=== Per-Class Performance ===")
    for class_name, metrics in per_class_metrics.items():
        print(f"\n{class_name}:")
        print(f"  Precision: {metrics['precision']:.4f}")
        print(f"  Recall: {metrics['recall']:.4f}")
        print(f"  F1-Score: {metrics['f1_score']:.4f}")
    
    return per_class_metrics

def plot_training_curves(project_path):
    """Plot training curves from results"""
    
    results_file = Path(project_path) / 'results.csv'
    if not results_file.exists():
        print("Training results file not found. Skipping training curves plot.")
        return
    
    import pandas as pd
    df = pd.read_csv(results_file)
    
    # Create plots
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('YOLOv8 Training Curves', fontsize=16)
    
    # Loss curves
    axes[0, 0].plot(df['epoch'], df['train/box_loss'], label='Train Box Loss')
    axes[0, 0].plot(df['epoch'], df['val/box_loss'], label='Val Box Loss')
    axes[0, 0].set_title('Box Loss')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].legend()
    axes[0, 0].grid(True)
    
    axes[0, 1].plot(df['epoch'], df['train/cls_loss'], label='Train Cls Loss')
    axes[0, 1].plot(df['epoch'], df['val/cls_loss'], label='Val Cls Loss')
    axes[0, 1].set_title('Classification Loss')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Loss')
    axes[0, 1].legend()
    axes[0, 1].grid(True)
    
    # mAP curves
    axes[1, 0].plot(df['epoch'], df['metrics/mAP50(B)'], label='mAP@0.5')
    axes[1, 0].plot(df['epoch'], df['metrics/mAP50-95(B)'], label='mAP@0.5:0.95')
    axes[1, 0].set_title('mAP Metrics')
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('mAP')
    axes[1, 0].legend()
    axes[1, 0].grid(True)
    
    # Precision/Recall curves
    axes[1, 1].plot(df['epoch'], df['metrics/precision(B)'], label='Precision')
    axes[1, 1].plot(df['epoch'], df['metrics/recall(B)'], label='Recall')
    axes[1, 1].set_title('Precision & Recall')
    axes[1, 1].set_xlabel('Epoch')
    axes[1, 1].set_ylabel('Score')
    axes[1, 1].legend()
    axes[1, 1].grid(True)
    
    plt.tight_layout()
    plt.savefig('training_curves.png', dpi=300, bbox_inches='tight')
    plt.show()

def generate_evaluation_report(metrics, per_class_metrics, results):
    """Generate comprehensive evaluation report"""
    
    report = {
        'timestamp': datetime.now().isoformat(),
        'model_path': str(results.save_dir) if hasattr(results, 'save_dir') else 'Unknown',
        'overall_metrics': metrics,
        'per_class_metrics': per_class_metrics,
        'dataset_info': {
            'test_images': len(results.pred) if hasattr(results, 'pred') else 'Unknown',
            'classes': ['with_mask', 'without_mask']
        }
    }
    
    # Save report
    with open('evaluation_report.json', 'w') as f:
        json.dump(report, f, indent=2)
    
    print(f"\nEvaluation report saved to: evaluation_report.json")
    
    return report

def main():
    """Main evaluation function"""
    
    # Configuration
    model_path = 'face_mask_detection/yolov8_face_mask/weights/best.pt'
    dataset_yaml = 'dataset.yaml'
    project_path = 'face_mask_detection/yolov8_face_mask'
    
    # Check if model exists
    if not os.path.exists(model_path):
        print(f"Model not found at {model_path}")
        print("Please run training first: python train_yolov8.py")
        return
    
    # Check if dataset.yaml exists
    if not os.path.exists(dataset_yaml):
        print(f"Dataset configuration not found at {dataset_yaml}")
        print("Please run training first to generate dataset configuration")
        return
    
    print("Starting model evaluation...")
    
    # Evaluate model performance
    metrics, results = evaluate_model_performance(model_path, dataset_yaml)
    
    # Generate per-class metrics
    per_class_metrics = generate_per_class_metrics(results)
    
    # Generate evaluation report
    report = generate_evaluation_report(metrics, per_class_metrics, results)
    
    # Plot training curves if available
    plot_training_curves(project_path)
    
    print("\n=== Evaluation Complete ===")
    print("Files generated:")
    print("- evaluation_report.json: Detailed evaluation metrics")
    print("- training_curves.png: Training progress visualization")
    print("- results.json: YOLOv8 validation results")

if __name__ == "__main__":
    main() 