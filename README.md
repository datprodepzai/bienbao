<br>
 
 <img src="Background.jpg" alt="Ảnh cô ấy" style="border: 5px solid #FF6347; border-radius: 10px; width: 300px; height: auto;">

# BienBao: Traffic Sign Recognition Using Deep Learning

## Overview

**BienBao** is a deep learning project focused on the automated recognition and classification of traffic signs. The project is developed as part of an academic exploration into computer vision and machine learning applications in intelligent transportation systems. Its potential usage spans across autonomous vehicles, traffic monitoring systems, and driver-assistance applications.

## Features

- Convolutional Neural Network (CNN) model for image-based classification.
- GRU (Gated Recurrent Unit) architecture for sequence-based learning.
- Real-time Graphical User Interface (GUI) for traffic sign prediction.
- Dataset of categorized traffic sign images with metadata.
- Support for training and testing on custom datasets.

## Project Structure

```
bienbao/
├── gui.py                      # GUI interface for testing traffic sign recognition
├── train.py                   # Training script for CNN model
├── train_GRU.py               # Training script for GRU-based model
├── my_model.h5                # Saved CNN model weights
├── traffic_classifier.h5      # Saved GRU model weights
├── Train.csv                  # Metadata for training dataset
├── Test.csv                   # Metadata for testing dataset
├── Meta.csv                   # Additional dataset metadata
├── train/                     # Training images
├── test/                      # Testing images
└── requirements.txt           # List of Python dependencies
```

## Installation

1. **Clone the repository**:

```bash
git clone https://github.com/datprodepzai/bienbao.git
cd bienbao
```

2. **Install dependencies**:

Make sure you have Python 3.x installed, then install the required libraries:

```bash
pip install -r requirements.txt
```

## Usage

### 1. Training the CNN Model

```bash
python train.py
```

### 2. Training the GRU Model

```bash
python train_GRU.py
```

### 3. Running the GUI

```bash
python gui.py
```

Use the GUI to load an image of a traffic sign and see the model’s prediction in real time.

## Dataset Description

The dataset consists of images of traffic signs categorized into classes. Each image is labeled and stored in structured folders (`train/` and `test/`). Metadata is provided in the form of CSV files:

- `Train.csv`: includes file paths and labels for training images.
- `Test.csv`: includes file paths and labels for test images.
- `Meta.csv`: includes class descriptions and additional annotations.

## Model Architectures

- **CNN (Convolutional Neural Network)**: Extracts spatial features from static images.
- **GRU (Gated Recurrent Unit)**: Captures temporal dynamics, particularly useful for sequential image inputs.

Both models are trained using supervised learning with cross-entropy loss and optimized using adaptive optimizers like Adam.

## Results

The models demonstrate high accuracy in recognizing traffic signs across multiple classes. GUI interaction offers instant feedback on predictions, allowing practical evaluation of the trained models.

## Future Work

- Integrate real-time webcam-based recognition.
- Augment dataset with diverse traffic sign images from different countries.
- Deploy the model in mobile or embedded systems.
- Enhance performance using ensemble learning or attention mechanisms.

## Contributing

Contributions are welcome. Please fork the repository and submit a pull request. For substantial changes, open an issue to discuss the proposed updates beforehand.

## Student Info
- Group ID: 7 
- Phạm Ngọc Phương Anh (22BI13037)
- Hoàng Quỳnh Anh (22BI13015)
- Phạm Chí Bách (22BI13055)
- Nguyễn Đức Dũng (22BI13106)

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## Acknowledgments

- TensorFlow and Keras open-source communities.
- Researchers and datasets that made this work possible.
- All contributors to the field of intelligent transportation systems.

---

Developed for academic and research purposes.
