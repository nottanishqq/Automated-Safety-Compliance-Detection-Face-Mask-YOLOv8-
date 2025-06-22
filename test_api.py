import requests
import base64
import json
from PIL import Image
import io
import os

def encode_image_to_base64(image_path):
    """Encode image to base64 string"""
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

def test_health_endpoint(base_url="http://localhost:5000"):
    """Test the health check endpoint"""
    try:
        response = requests.get(f"{base_url}/health")
        print("=== Health Check ===")
        print(f"Status Code: {response.status_code}")
        print(f"Response: {response.json()}")
        return response.status_code == 200
    except Exception as e:
        print(f"Health check failed: {e}")
        return False

def test_predict_endpoint(image_path, base_url="http://localhost:5000"):
    """Test the predict endpoint with base64 image"""
    try:
        # Encode image to base64
        image_base64 = encode_image_to_base64(image_path)
        
        # Prepare request
        payload = {
            "image": image_base64
        }
        
        # Make request
        response = requests.post(f"{base_url}/predict", json=payload)
        
        print(f"\n=== Prediction Results for {os.path.basename(image_path)} ===")
        print(f"Status Code: {response.status_code}")
        
        if response.status_code == 200:
            result = response.json()
            print(f"Success: {result['success']}")
            print(f"Total Detections: {result['total_detections']}")
            print(f"Image Shape: {result['image_shape']}")
            
            if result['detections']:
                print("\nDetections:")
                for i, detection in enumerate(result['detections']):
                    print(f"  Detection {i+1}:")
                    print(f"    Class: {detection['class_name']}")
                    print(f"    Confidence: {detection['confidence']:.3f}")
                    print(f"    Bounding Box: ({detection['bbox']['x1']}, {detection['bbox']['y1']}) to ({detection['bbox']['x2']}, {detection['bbox']['y2']})")
                    print(f"    Size: {detection['bbox']['width']} x {detection['bbox']['height']}")
            else:
                print("No detections found")
        else:
            print(f"Error: {response.text}")
            
        return response.status_code == 200
        
    except Exception as e:
        print(f"Prediction test failed: {e}")
        return False

def test_file_upload_endpoint(image_path, base_url="http://localhost:5000"):
    """Test the file upload endpoint"""
    try:
        # Prepare file upload
        with open(image_path, 'rb') as f:
            files = {'file': f}
            response = requests.post(f"{base_url}/predict_file", files=files)
        
        print(f"\n=== File Upload Results for {os.path.basename(image_path)} ===")
        print(f"Status Code: {response.status_code}")
        
        if response.status_code == 200:
            result = response.json()
            print(f"Success: {result['success']}")
            print(f"Total Detections: {result['total_detections']}")
            print(f"Image Shape: {result['image_shape']}")
            
            if result['detections']:
                print("\nDetections:")
                for i, detection in enumerate(result['detections']):
                    print(f"  Detection {i+1}:")
                    print(f"    Class: {detection['class_name']}")
                    print(f"    Confidence: {detection['confidence']:.3f}")
                    print(f"    Bounding Box: ({detection['bbox']['x1']}, {detection['bbox']['y1']}) to ({detection['bbox']['x2']}, {detection['bbox']['y2']})")
                    print(f"    Size: {detection['bbox']['width']} x {detection['bbox']['height']}")
            else:
                print("No detections found")
        else:
            print(f"Error: {response.text}")
            
        return response.status_code == 200
        
    except Exception as e:
        print(f"File upload test failed: {e}")
        return False

def test_multiple_images(base_url="http://localhost:5000"):
    """Test the API with multiple images from the dataset"""
    
    # Find test images
    dataset_path = "Labeled Mask Dataset/obj"
    if not os.path.exists(dataset_path):
        print(f"Dataset path not found: {dataset_path}")
        return
    
    # Get a few sample images
    image_files = [f for f in os.listdir(dataset_path) if f.endswith('.jpg')][:3]
    
    print(f"\n=== Testing with {len(image_files)} sample images ===")
    
    for image_file in image_files:
        image_path = os.path.join(dataset_path, image_file)
        print(f"\nTesting image: {image_file}")
        
        # Test both endpoints
        test_predict_endpoint(image_path, base_url)
        test_file_upload_endpoint(image_path, base_url)

def main():
    """Main test function"""
    
    base_url = "http://localhost:5000"
    
    print("Face Mask Detection API Test Client")
    print("=" * 50)
    
    # Test health endpoint
    if not test_health_endpoint(base_url):
        print("API is not running or not accessible. Please start the Flask server first.")
        return
    
    # Test with sample images
    test_multiple_images(base_url)
    
    print("\n" + "=" * 50)
    print("Testing complete!")

if __name__ == "__main__":
    main() 