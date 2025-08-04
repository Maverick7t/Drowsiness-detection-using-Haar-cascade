# Drowsiness Detection using Haar Cascade and Deep Learning

A real-time drowsiness detection system that uses computer vision and deep learning to monitor driver alertness. The system detects when a person's eyes are closed for an extended period and triggers an audio alarm to prevent accidents caused by drowsy driving.

## Features

- **Real-time eye detection** using Haar Cascade classifiers
- **Deep learning classification** with InceptionV3-based model to distinguish between open and closed eyes
- **Audio alert system** that triggers when drowsiness is detected
- **Scoring system** that tracks consecutive closed-eye detections
- **Live video feed** with visual indicators for eye state and drowsiness score

## How It Works

1. **Face Detection**: Uses Haar Cascade to detect faces in the video stream
2. **Eye Detection**: Identifies eye regions within detected faces
3. **Eye State Classification**: A trained CNN model classifies each detected eye as open or closed
4. **Drowsiness Scoring**: Tracks consecutive closed-eye detections
5. **Alert System**: Plays an alarm sound when the drowsiness score exceeds the threshold

## Requirements

### Hardware
- Webcam or camera device
- Speakers or headphones for audio alerts

### Software Dependencies

```bash
# Core libraries
opencv-python>=4.5.0
tensorflow>=2.8.0
numpy>=1.19.0
pygame>=2.0.0

# Additional dependencies for training (optional)
Pillow>=8.0.0
matplotlib>=3.3.0
```

## Installation

1. **Clone the repository**
```bash
git clone https://github.com/your-username/drowsiness-detection-haar-cascade.git
cd drowsiness-detection-haar-cascade
```

2. **Create a virtual environment (recommended)**
```bash
python -m venv drowsiness_env
source drowsiness_env/bin/activate  # On Windows: drowsiness_env\Scripts\activate
```

3. **Install required packages**
```bash
pip install opencv-python tensorflow numpy pygame
```

4. **Set up the project structure**
```
Drowsiness-detection-using-Haar-cascade/
├── main.ipynb
├── Model training.ipynb
├── README.md
├── models/
│   └── model.h5          # Pre-trained model (you'll need to train this)
├── data/
│   ├── train/
│   │   ├── closed/       # Training images of closed eyes
│   │   └── open/         # Training images of open eyes
│   └── test/
│       ├── closed/       # Test images of closed eyes
│       └── open/         # Test images of open eyes
└── alarm.wav             # Audio file for drowsiness alert
```

## Dataset Preparation

1. **Collect eye images** or download a public dataset
2. **Organize images** into the following structure:
   - `data/train/closed/` - Images of closed eyes for training
   - `data/train/open/` - Images of open eyes for training
   - `data/test/closed/` - Images of closed eyes for testing
   - `data/test/open/` - Images of open eyes for testing

3. **Image requirements**:
   - Images should be cropped to show only the eye region
   - Recommended size: 80x80 pixels (images will be resized automatically)
   - Supported formats: PNG, JPG, JPEG

## Model Training

1. **Open the training notebook**
```bash
jupyter notebook "Model training.ipynb"
```

2. **Update file paths** in the notebook to match your system:
```python
# Update these paths in the training notebook
train_data_path = r'path/to/your/data/train'
test_data_path = r'path/to/your/data/test'
model_save_path = r'path/to/your/models/model.h5'
```

3. **Run all cells** to train the model
   - The model uses InceptionV3 as the base with transfer learning
   - Training typically takes 20-50 epochs depending on your dataset
   - The best model will be automatically saved to `models/model.h5`

## Usage

### Running the Drowsiness Detection System

1. **Update file paths** in `main.ipynb`:
```python
# Update these paths to match your system
model_path = r'path/to/your/models/model.h5'
alarm_path = r'path/to/your/alarm.wav'
```

2. **Add an alarm sound file**:
   - Place a `.wav` audio file in your project directory
   - Name it `alarm.wav` or update the path in the code

3. **Run the main notebook**:
```bash
jupyter notebook main.ipynb
```

4. **Execute all cells** to start the drowsiness detection system

### Using the System

- **Position yourself** in front of the camera with good lighting
- **Green rectangles** will appear around detected faces and eyes
- **Status display** shows:
  - "open" when eyes are detected as open
  - "closed" when eyes are detected as closed
  - Current drowsiness score
- **Alarm triggers** when the score exceeds 15 (consecutive closed-eye detections)
- **Press 'q'** to quit the application

## Configuration

### Adjustable Parameters

In `main.ipynb`, you can modify:

```python
# Detection sensitivity
scaleFactor = 1.2        # Face detection scale factor
minNeighbors = 3         # Face detection minimum neighbors

# Eye detection parameters
eye_scaleFactor = 1.1    # Eye detection scale factor
eye_minNeighbors = 1     # Eye detection minimum neighbors

# Drowsiness thresholds
closed_threshold = 0.30  # Threshold for classifying eyes as closed
open_threshold = 0.90    # Threshold for classifying eyes as open
alarm_threshold = 15     # Score threshold for triggering alarm

# Score adjustment
score_increment = 1      # Points added for closed eyes
score_decrement = 1      # Points subtracted for open eyes
```

## Troubleshooting

### Common Issues

1. **"No module named 'cv2'"**
   ```bash
   pip install opencv-python
   ```

2. **"Cannot load model.h5"**
   - Ensure you've trained the model first using `Model training.ipynb`
   - Check that the file path is correct

3. **"No sound from alarm"**
   - Verify the alarm.wav file exists and path is correct
   - Check system audio settings
   - Try a different audio format

4. **Poor detection accuracy**
   - Ensure good lighting conditions
   - Position camera at eye level
   - Adjust detection parameters
   - Retrain model with more diverse data

5. **Camera not working**
   ```python
   # Try different camera indices
   cap = cv2.VideoCapture(1)  # Instead of 0
   ```

### Performance Tips

- **Use GPU acceleration** for faster inference (requires CUDA-compatible GPU)
- **Adjust image resolution** for better performance vs accuracy trade-off
- **Optimize lighting** in your environment for better detection
- **Fine-tune thresholds** based on your specific use case

## Model Architecture

The system uses a transfer learning approach:

- **Base Model**: InceptionV3 (pre-trained on ImageNet)
- **Custom Layers**: 
  - Flatten layer
  - Dense layer (64 units, ReLU activation)
  - Dropout layer (0.5 rate)
  - Output layer (2 units, Softmax activation)

## Dataset Recommendations

For best results, use datasets containing:
- Various lighting conditions
- Different eye shapes and sizes
- Multiple ethnicities and ages
- Different angles and positions
- Both natural and deliberate eye closures

Popular datasets:
- CEW (Closed Eyes in the Wild)
- UTA-RLDD (Real-Life Drowsiness Dataset)
- Custom collected data specific to your use case

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/improvement`)
3. Commit your changes (`git commit -am 'Add new feature'`)
4. Push to the branch (`git push origin feature/improvement`)
5. Create a Pull Request


## Acknowledgments

- OpenCV for computer vision capabilities
- TensorFlow/Keras for deep learning framework
- InceptionV3 model architecture
- Contributors to open-source drowsiness detection research

## Future Improvements

- [ ] Add yawning detection
- [ ] Implement head pose estimation
- [ ] Mobile app integration
- [ ] Cloud-based model deployment
- [ ] Multi-person detection support
- [ ] Integration with vehicle systems

## Contact

For questions or support, please open an issue on GitHub or contact [your-email@example.com].
