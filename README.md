# Face Mask Detection with YOLOv8

A state-of-the-art face mask detection system using YOLOv8, featuring comprehensive training, evaluation, and deployment capabilities.

## 🎯 Features

- **YOLOv8 Training**: Fine-tuned YOLOv8 model for face mask detection
- **Data Augmentation**: Comprehensive augmentation pipeline for robust training
- **Performance Metrics**: Detailed mAP@0.5 and mAP@0.5:0.95 reporting
- **Flask API**: RESTful API for real-time inference
- **Docker Containerization**: Easy deployment with Docker
- **Comprehensive Evaluation**: Detailed performance analysis and visualization

## 📊 Dataset

The system uses a face mask dataset with YOLO format annotations:
- **Class 0**: with_mask
- **Class 1**: without_mask
- **Format**: `class_id center_x center_y width height` (normalized coordinates)

## 🚀 Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Train the Model

```bash
python train_yolov8.py
```

This will:
- Organize the dataset into train/val/test splits (70%/20%/10%)
- Create dataset configuration
- Train YOLOv8 with data augmentation
- Report mAP metrics automatically

### 3. Evaluate Model Performance

```bash
python evaluate_model.py
```

This generates:
- Detailed mAP@0.5 and mAP@0.5:0.95 metrics
- Per-class performance analysis
- Training curves visualization
- Comprehensive evaluation report

### 4. Run Flask API

```bash
python app.py
```

The API will be available at `http://localhost:5000`

### 5. Test the API

```bash
python test_api.py
```

## 📈 Data Augmentation Pipeline

The training includes the following augmentations:

- **HSV Augmentation**: Hue (±0.015), Saturation (±0.7), Value (±0.4)
- **Geometric Transformations**: Translation (±0.1), Scale (±0.5)
- **Flip Augmentation**: Horizontal flip (50% probability)
- **Mosaic Augmentation**: 100% probability for enhanced training

## 🔧 API Endpoints

### Health Check
```bash
GET /health
```

### Predict with Base64 Image
```bash
POST /predict
Content-Type: application/json

{
    "image": "base64_encoded_image_string"
}
```

### Predict with File Upload
```bash
POST /predict_file
Content-Type: multipart/form-data

file: image_file
```

### Response Format
```json
{
    "success": true,
    "detections": [
        {
            "bbox": {
                "x1": 100,
                "y1": 150,
                "x2": 300,
                "y2": 450,
                "width": 200,
                "height": 300
            },
            "confidence": 0.95,
            "class_id": 0,
            "class_name": "with_mask"
        }
    ],
    "total_detections": 1,
    "image_shape": {
        "width": 640,
        "height": 480
    }
}
```

## 🐳 Docker Deployment

### Build and Run with Docker Compose

```bash
# Build and start the container
docker-compose up --build

# Run in background
docker-compose up -d --build
```

### Manual Docker Build

```bash
# Build image
docker build -t face-mask-detection .

# Run container
docker run -p 5000:5000 face-mask-detection
```

## 📊 Performance Metrics

The system reports standard object detection metrics:

- **mAP@0.5**: Mean Average Precision at IoU threshold 0.5
- **mAP@0.5:0.95**: Mean Average Precision across IoU thresholds 0.5 to 0.95
- **Precision**: True Positives / (True Positives + False Positives)
- **Recall**: True Positives / (True Positives + False Negatives)
- **F1-Score**: Harmonic mean of Precision and Recall

## 📁 Project Structure

```
object-detection-from-scratch/
├── Labeled Mask Dataset/          # Original dataset
│   └── obj/                      # Images and annotations
├── train_yolov8.py               # Training script with augmentation
├── evaluate_model.py              # Model evaluation and metrics
├── app.py                        # Flask API server
├── test_api.py                   # API testing client
├── requirements.txt              # Python dependencies
├── Dockerfile                    # Docker configuration
├── docker-compose.yml            # Docker Compose setup
└── README.md                     # This file
```

## 🔍 Model Architecture

- **Base Model**: YOLOv8-nano (pretrained on COCO)
- **Input Size**: 640x640 pixels
- **Classes**: 2 (with_mask, without_mask)
- **Training**: 100 epochs with early stopping
- **Optimizer**: Adam with learning rate scheduling

## 🛠️ Configuration

### Training Parameters
- **Epochs**: 100
- **Batch Size**: 16
- **Image Size**: 640x640
- **Early Stopping**: 50 epochs patience
- **Device**: Auto (GPU if available)

### Inference Parameters
- **Confidence Threshold**: 0.25
- **IoU Threshold**: 0.45
- **Max Detections**: No limit

## 📝 Usage Examples

### Python Client Example

```python
import requests
import base64

# Encode image
with open("image.jpg", "rb") as f:
    image_base64 = base64.b64encode(f.read()).decode()

# Make prediction
response = requests.post("http://localhost:5000/predict", 
                        json={"image": image_base64})
result = response.json()

# Process results
for detection in result['detections']:
    print(f"Found {detection['class_name']} with confidence {detection['confidence']}")
```

### cURL Example

```bash
# Health check
curl http://localhost:5000/health

# File upload
curl -X POST -F "file=@image.jpg" http://localhost:5000/predict_file
```

## 🔧 Troubleshooting

### Common Issues

1. **CUDA Out of Memory**: Reduce batch size in `train_yolov8.py`
2. **Model Not Found**: Ensure training completed successfully
3. **API Connection Error**: Check if Flask server is running on port 5000
4. **Docker Build Fails**: Ensure Docker has sufficient memory and disk space

### Performance Optimization

- Use GPU for training (automatically detected)
- Adjust batch size based on available memory
- Use smaller model variant (yolov8n) for faster inference
- Enable model caching for repeated predictions

## 📄 License

This project is open source and available under the MIT License.

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📞 Support

For questions and support, please open an issue on the project repository.
