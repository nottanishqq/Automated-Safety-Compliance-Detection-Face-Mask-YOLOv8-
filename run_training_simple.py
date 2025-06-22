#!/usr/bin/env python3
"""
Simplified quick start script for face mask detection training and evaluation
(Skips dependency checks)
"""

import os
import sys
import subprocess
import time

def run_command(command, description):
    """Run a command and handle errors"""
    print(f"\n{'='*60}")
    print(f"🚀 {description}")
    print(f"{'='*60}")
    print(f"Running: {command}")
    
    try:
        result = subprocess.run(command, shell=True, check=True, capture_output=True, text=True)
        print("✅ Success!")
        if result.stdout:
            print("Output:")
            print(result.stdout)
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Error: {e}")
        if e.stdout:
            print("STDOUT:")
            print(e.stdout)
        if e.stderr:
            print("STDERR:")
            print(e.stderr)
        return False

def check_dataset():
    """Check if dataset exists"""
    dataset_path = "Labeled Mask Dataset/obj"
    if not os.path.exists(dataset_path):
        print(f"❌ Dataset not found at: {dataset_path}")
        return False
    
    # Count images
    image_files = [f for f in os.listdir(dataset_path) if f.endswith('.jpg')]
    print(f"✅ Dataset found with {len(image_files)} images")
    return True

def main():
    """Main execution function"""
    print("🎯 Face Mask Detection - Quick Start (Simplified)")
    print("="*60)
    
    # Check dataset
    if not check_dataset():
        sys.exit(1)
    
    # Step 1: Train the model
    print("\n📊 Step 1: Training YOLOv8 Model")
    if not run_command("python train_yolov8.py", "Training YOLOv8 model with data augmentation"):
        print("❌ Training failed. Please check the error messages above.")
        sys.exit(1)
    
    # Wait a moment for files to be written
    time.sleep(2)
    
    # Step 2: Evaluate the model
    print("\n📈 Step 2: Evaluating Model Performance")
    if not run_command("python evaluate_model.py", "Evaluating model and generating metrics"):
        print("⚠️  Evaluation failed, but training completed successfully.")
        print("You can still use the trained model.")
    
    # Step 3: Start the API (optional)
    print("\n🌐 Step 3: Starting Flask API")
    print("The API will be available at http://localhost:5000")
    print("Press Ctrl+C to stop the API")
    
    try:
        run_command("python app.py", "Starting Flask API server")
    except KeyboardInterrupt:
        print("\n👋 API stopped by user")
    
    print("\n🎉 Setup complete!")
    print("\nNext steps:")
    print("1. Test the API: python test_api.py")
    print("2. Use Docker: docker-compose up --build")
    print("3. Check results in face_mask_detection/ directory")

if __name__ == "__main__":
    main() 