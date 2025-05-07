<h1 style="color: #FF6347;">Chào mừng bé HQA đến với thế giới của anh</h1>
<p style="font-size: 24px; color: #FFD700;">（づ￣3￣）づ╭❤️～</p>

<br>

<img src="Background.jpg" alt="Ảnh cô ấy" style="border: 5px solid #FF6347; border-radius: 10px; width: 500px; height: auto;">


BienBao: Traffic Sign Recognition Using Deep Learning
Overview
BienBao is a machine learning project aimed at the automatic recognition and classification of traffic signs. By leveraging deep learning models, the project seeks to enhance the accuracy and efficiency of traffic sign detection, which is crucial for applications such as autonomous driving and advanced driver-assistance systems (ADAS).

Features
Deep Learning Models: Utilizes convolutional neural networks (CNNs) and gated recurrent units (GRUs) for image classification tasks.

Graphical User Interface (GUI): Provides an intuitive interface for users to interact with the model and visualize predictions.

Comprehensive Dataset: Includes a diverse set of traffic sign images for training and testing purposes.

Project Structure
The repository comprises the following key components:

train.py: Script for training the CNN model on the traffic sign dataset.

train_GRU.py: Script for training the GRU-based model.

gui.py: Implements the GUI for model interaction and visualization.

my_model.h5 and traffic_classifier.h5: Pre-trained model weights for CNN and GRU models, respectively.

train/ and test/: Directories containing training and testing images.

Train.csv and Test.csv: CSV files detailing the metadata for training and testing datasets.

Meta.csv: Contains additional metadata information for the dataset.

Installation
Clone the Repository:

bash
Sao chép
Chỉnh sửa
git clone https://github.com/datprodepzai/bienbao.git
cd bienbao
Install Dependencies:

Ensure you have Python 3.x installed. Install the required packages using pip:

bash
Sao chép
Chỉnh sửa
pip install -r requirements.txt
Usage
Training the Model:

To train the CNN model:

bash
Sao chép
Chỉnh sửa
python train.py
To train the GRU model:

bash
Sao chép
Chỉnh sửa
python train_GRU.py
Launching the GUI:

After training, launch the GUI to interact with the model:

bash
Sao chép
Chỉnh sửa
python gui.py
The GUI allows users to input traffic sign images and view the model's predictions in real-time.

Dataset
The dataset comprises various traffic sign images categorized into different classes. Each image is labeled and stored in the train/ and test/ directories. The accompanying CSV files (Train.csv, Test.csv, and Meta.csv) provide metadata, including image paths and corresponding labels.

Model Architecture
CNN Model: Designed to capture spatial hierarchies in images, making it suitable for recognizing traffic sign patterns.

GRU Model: Incorporates temporal dependencies, which can be beneficial if the dataset includes sequential data or video frames.

Both models are trained using supervised learning techniques and optimized using appropriate loss functions and optimizers to achieve high classification accuracy.

Results
The trained models demonstrate promising accuracy in classifying various traffic signs. The GUI facilitates easy testing and validation of the models on new images, providing immediate visual feedback on predictions.

Future Work
Dataset Expansion: Incorporate more diverse and extensive traffic sign images to improve model generalization.

Model Optimization: Experiment with different architectures and hyperparameters to enhance performance.

Real-time Deployment: Integrate the model into real-time systems for on-the-fly traffic sign recognition.

Contributing
Contributions are welcome! Please fork the repository and submit a pull request with your enhancements. For major changes, kindly open an issue first to discuss the proposed modifications.

License
This project is licensed under the MIT License. See the LICENSE file for details.

Acknowledgments
We extend our gratitude to the open-source community and contributors who have provided valuable resources and inspiration for this project.

For more details and to access the project, visit the GitHub repository.


