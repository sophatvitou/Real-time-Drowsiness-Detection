# Drowsiness Detection System 

An AI-powered real-time drowsiness detection system that monitors eye states using deep learning and computer vision. This project uses a trained CNN model to classify eyes as open or closed to detect driver drowsiness in real-time.

## Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Project Structure](#project-structure)
- [Dataset](#dataset)
- [Installation](#installation)
- [Usage](#usage)
- [Model Details](#model-details)
- [Results](#results)
- [Future Improvements](#future-improvements)
- [License](#license)

## Overview

This project implements a real-time drowsiness detection system that:
- Detects faces and eyes using Haar Cascades
- Uses a pre-trained deep learning model to classify eye states (open/closed)
- Triggers alerts when prolonged eye closure is detected
- Works with live camera feed for real-time monitoring

**Use Cases:**
- Driver safety monitoring
- Workplace fatigue detection
- Student engagement tracking

## Features

- ✅ Real-time eye detection using OpenCV
- ✅ Deep learning-based eye state classification
- ✅ Configurable drowsiness threshold
- ✅ Pre-trained model included
- ✅ Jupyter notebooks for data exploration and training
- ✅ Labeled dataset with open/closed eye categories

## Project Structure

```
ai_tp_project/
│
├── clean_data.ipynb              # Data cleaning & preparation notebook
├── real_time_drowsiness.ipynb    # Real-time detection notebook
├── demo.py                       # Real-time camera demo script
│
├── drowsiness_eye_model.h5       # Pre-trained model
│
├── DataClean/                    # Cleaned and labeled dataset
│   ├── open/                     # Images of open eyes (1000+)
│   └── close/                    # Images of closed eyes (1000+)
│
├── mrlEyes_2018_01/              # Original dataset (37 subjects)
│   ├── annotation.txt
│   └── s0001/ to s0037/          # Subject folders with eye images
│
├── requirements.txt              # Python dependencies
└── README.md                     # This file
```

## Dataset

**Source:** MRLEyes2018 Dataset (37 subjects)
- **Total Images:** 1000+ labeled eye images
- **Categories:**
  - Open eyes: Images labeled with eye state '1'
  - Closed eyes: Images labeled with eye state '0'
- **Image Size:** 224×224 pixels (grayscale)
- **Format:** PNG

**Data Cleaning:**
The `clean_data.ipynb` notebook processes the raw dataset and organizes images into `DataClean/open/` and `DataClean/close/` folders based on eye state annotation.

## 🚀 Installation

### Prerequisites
- Python 3.8 or higher
- pip or conda

### Step 1: Clone the Repository
```bash
git clone https://github.com/yourusername/drowsiness-detection.git
cd drowsiness-detection
```

### Step 2: Create Virtual Environment (Recommended)
```bash
# Windows
python -m venv venv
venv\Scripts\activate

# macOS/Linux
python3 -m venv venv
source venv/bin/activate
```

### Step 3: Install Dependencies
```bash
pip install -r requirements.txt
```

## Usage

### Option 1: Real-Time Detection Demo
Run the camera-based demo:

```bash
python demo.py
```

**Controls:**
- Press `q` to quit the application
- The system will display:
  - Detected faces with bounding boxes
  - Detected eyes with predictions (OPEN/CLOSED)
  - Drowsiness status and frame counter

### Option 2: Jupyter Notebooks

#### Data Cleaning
```bash
jupyter notebook clean_data.ipynb
```
Use this to explore and prepare the raw dataset.

#### Real-Time Detection
```bash
jupyter notebook real_time_drowsiness.ipynb
```
Interactive notebook for experimenting with the detection pipeline.

## Model Details

**Architecture:**
- Model Type: Convolutional Neural Network (CNN)
- Input: 224×224×1 (grayscale eye images)
- Output: Binary classification (Open: 1, Closed: 0)
- File: `drowsiness_eye_model.h5` (Keras/TensorFlow format)

**Face Detection:**
- Haar Cascade: `haarcascade_frontalface_default.xml`
- Eye Detection: `haarcascade_eye.xml`

**Preprocessing:**
- Resize: 224×224 pixels
- Normalization: Divide by 255
- Input shape: (1, 224, 224, 1)

## 📈 Results

- **Accuracy:** Detects open/closed eye states with high precision
- **Real-Time Performance:** Processes video frames at ~20-30 FPS
- **Drowsiness Detection:** Triggers alert after 20+ consecutive frames of closed eyes

## 🔧 Configuration

Edit `demo.py` to customize:

```python
DROWSY_THRESHOLD = 20  # Frames before triggering alert (default: 20)
```

Increase for more tolerance, decrease for more sensitivity.

## 🚀 Future Improvements

- [ ] Add sound/visual alerts when drowsiness detected
- [ ] Implement eye movement tracking
- [ ] Add blink rate analysis
- [ ] Support multiple face detection
- [ ] Mobile app deployment
- [ ] Improve model accuracy with augmented dataset
- [ ] Add logging and analytics

## Technical Requirements

See `requirements.txt` for complete dependencies:
- **tensorflow** - Deep learning framework
- **opencv-python** - Computer vision library
- **numpy** - Numerical computing
- **pandas** - Data manipulation
- **matplotlib** - Visualization
- **scikit-learn** - ML utilities

## 🔗 References

- Dataset: [MRLEyes2018 - Gaze Tracking Dataset](http://mrl.cs.vsb.cz/eyedataset)
- OpenCV Documentation: https://docs.opencv.org/
- TensorFlow/Keras: https://www.tensorflow.org/

## ⚠️ Disclaimer

This system is for **educational and research purposes**. It should not be relied upon as the sole method for safety-critical applications like driver monitoring without additional validation and safety measures.

## 📄 License

This project is open source and available under the MIT License.

## 👥 Contributing

Contributions are welcome! Feel free to:
1. Fork the repository
2. Create a feature branch
3. Submit a pull request

## 📧 Contact

For questions or suggestions, please open an issue on GitHub.

---

**Last Updated:** 2026  
**Status:** ✅ Active Development
