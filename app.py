from flask import Flask, request, jsonify
from flask_cors import CORS
import cv2
import numpy as np
import base64
import io
from PIL import Image
from ultralytics import YOLO
import os
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)

# Load the trained model
MODEL_PATH = 'face_mask_detection/yolov8_face_mask/weights/best.pt'
model = None

def load_model():
    """Load the trained YOLOv8 model"""
    global model
    try:
        if os.path.exists(MODEL_PATH):
            model = YOLO(MODEL_PATH)
            logger.info(f"Model loaded successfully from {MODEL_PATH}")
        else:
            # Fallback to pretrained model if trained model doesn't exist
            model = YOLO('yolov8n.pt')
            logger.warning(f"Trained model not found at {MODEL_PATH}, using pretrained model")
    except Exception as e:
        logger.error(f"Error loading model: {e}")
        model = YOLO('yolov8n.pt')

def base64_to_image(base64_string):
    """Convert base64 string to PIL Image"""
    try:
        # Remove data URL prefix if present
        if ',' in base64_string:
            base64_string = base64_string.split(',')[1]
        
        # Decode base64 string
        image_data = base64.b64decode(base64_string)
        image = Image.open(io.BytesIO(image_data))
        return image
    except Exception as e:
        logger.error(f"Error converting base64 to image: {e}")
        return None

def image_to_base64(image):
    """Convert PIL Image to base64 string"""
    try:
        buffer = io.BytesIO()
        image.save(buffer, format='JPEG')
        img_str = base64.b64encode(buffer.getvalue()).decode()
        return img_str
    except Exception as e:
        logger.error(f"Error converting image to base64: {e}")
        return None

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'model_loaded': model is not None,
        'model_path': MODEL_PATH
    })

@app.route('/predict', methods=['POST'])
def predict():
    """Predict face masks in uploaded image"""
    try:
        # Check if model is loaded
        if model is None:
            return jsonify({'error': 'Model not loaded'}), 500
        
        # Get image data from request
        if 'image' not in request.json:
            return jsonify({'error': 'No image provided'}), 400
        
        # Convert base64 to image
        image = base64_to_image(request.json['image'])
        if image is None:
            return jsonify({'error': 'Invalid image format'}), 400
        
        # Convert PIL image to numpy array
        image_np = np.array(image)
        
        # Run inference
        results = model(image_np, conf=0.25, iou=0.45)  # Confidence and IoU thresholds
        
        # Process results
        detections = []
        for result in results:
            boxes = result.boxes
            if boxes is not None:
                for box in boxes:
                    # Get coordinates (normalized)
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    
                    # Get confidence and class
                    confidence = float(box.conf[0].cpu().numpy())
                    class_id = int(box.cls[0].cpu().numpy())
                    class_name = ['with_mask', 'without_mask'][class_id]
                    
                    # Convert to absolute coordinates
                    img_height, img_width = image_np.shape[:2]
                    x1_abs, y1_abs = int(x1), int(y1)
                    x2_abs, y2_abs = int(x2), int(y2)
                    
                    detection = {
                        'bbox': {
                            'x1': x1_abs,
                            'y1': y1_abs,
                            'x2': x2_abs,
                            'y2': y2_abs,
                            'width': x2_abs - x1_abs,
                            'height': y2_abs - y1_abs
                        },
                        'confidence': round(confidence, 3),
                        'class_id': class_id,
                        'class_name': class_name
                    }
                    detections.append(detection)
        
        # Prepare response
        response = {
            'success': True,
            'detections': detections,
            'total_detections': len(detections),
            'image_shape': {
                'width': image_np.shape[1],
                'height': image_np.shape[0]
            }
        }
        
        return jsonify(response)
        
    except Exception as e:
        logger.error(f"Error during prediction: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/predict_file', methods=['POST'])
def predict_file():
    """Predict face masks in uploaded file"""
    try:
        # Check if model is loaded
        if model is None:
            return jsonify({'error': 'Model not loaded'}), 500
        
        # Check if file is present
        if 'file' not in request.files:
            return jsonify({'error': 'No file provided'}), 400
        
        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400
        
        # Read image
        image_data = file.read()
        image = Image.open(io.BytesIO(image_data))
        image_np = np.array(image)
        
        # Run inference
        results = model(image_np, conf=0.25, iou=0.45)
        
        # Process results (same as predict endpoint)
        detections = []
        for result in results:
            boxes = result.boxes
            if boxes is not None:
                for box in boxes:
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    confidence = float(box.conf[0].cpu().numpy())
                    class_id = int(box.cls[0].cpu().numpy())
                    class_name = ['with_mask', 'without_mask'][class_id]
                    
                    x1_abs, y1_abs = int(x1), int(y1)
                    x2_abs, y2_abs = int(x2), int(y2)
                    
                    detection = {
                        'bbox': {
                            'x1': x1_abs,
                            'y1': y1_abs,
                            'x2': x2_abs,
                            'y2': y2_abs,
                            'width': x2_abs - x1_abs,
                            'height': y2_abs - y1_abs
                        },
                        'confidence': round(confidence, 3),
                        'class_id': class_id,
                        'class_name': class_name
                    }
                    detections.append(detection)
        
        response = {
            'success': True,
            'detections': detections,
            'total_detections': len(detections),
            'image_shape': {
                'width': image_np.shape[1],
                'height': image_np.shape[0]
            }
        }
        
        return jsonify(response)
        
    except Exception as e:
        logger.error(f"Error during file prediction: {e}")
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    # Load model on startup
    load_model()
    
    # Run Flask app
    app.run(host='0.0.0.0', port=5000, debug=False) 