# Traffic Sign Detection and Recognition with Voice Alert System

Implementing a robust Traffic Sign Detection and Recognition system with integrated Voice Alert. Recognizes various signs, provides real-time visual feedback, and triggers voice alerts upon detection. Achieves high accuracy in detection and recognition, enhancing driver awareness without diverting attention from the road.

**Note: Please download the DFG dataset from the following link: [DFG Dataset](https://www.vicos.si/resources/dfg/)**

## Features

- Real-time detection and recognition of traffic signs.
- Integration of voice alert system for timely notifications to drivers.
- High accuracy in both detection and recognition tasks.
- Provides visual feedback on captured frames.
- Enhances driver awareness without distracting from the road.

## Requirements

### Software Requirements
- Operating System: Windows 10 or above
- Programming Language: Python 3.8.10 or above
- Libraries:
  - OpenCV for image processing.
  - Tkinter for GUI development.
  - PIL for image handling.
  - numpy for array manipulation.
  - tensorflow.keras.models for loading the pre-trained model.
  - pyttsx3 for text-to-speech conversion.
  - Ultralytics YOLO for efficient and accurate object detection.

### Hardware Requirements
- Processor: A CPU with a clock speed of 3 GHz or more and 4 or more cores, or GPU with CUDA version 11.8 or more, which speeds up model training and inference.
- Random Access Memory: 8GB or more RAM necessary to handle the data and model during training.
- Camera: To capture the video frames.

![Evaluation the model](https://raw.githubusercontent.com/SohailAkbar/Traffic-Sign-Detection-and-Recognition-with-Voice-Alert-System/main/Evaluation%20the%20model.png)

![Visualizing the dataset](https://raw.githubusercontent.com/SohailAkbar/Traffic-Sign-Detection-and-Recognition-with-Voice-Alert-System/main/visualising%20the%20dataset.png)



## Model Details

The Traffic Sign Detection and Recognition system utilizes two main models for its functionality:

1. **YOLO (You Only Look Once) Model**:
   - The YOLO model is employed for real-time object detection, specifically for detecting traffic signs within video frames. YOLO is known for its efficiency and accuracy in detecting objects in images and videos.
   - The pre-trained YOLO model file `traffic_det.pt` is loaded using the Ultralytics library. This model has been trained on a dataset containing various objects, including traffic signs, to accurately detect their presence and location in video frames.

2. **Convolutional Neural Network (CNN) Model**:
   - A CNN model is utilized for the recognition of specific traffic signs detected by the YOLO model. CNNs are well-suited for image classification tasks due to their ability to learn hierarchical features from input images.
   - The trained CNN model is loaded from the file `traffic_classifier.h5`. This model has been trained on a dataset of traffic sign images and is capable of classifying them into specific categories based on their visual features.

Both models work collaboratively to provide accurate and efficient traffic sign detection and recognition capabilities in real-time. The YOLO model locates the traffic signs within video frames, while the CNN model classifies the detected signs into specific categories, enabling the system to provide meaningful feedback to users.

### Usage

To use the trained model for traffic sign recognition in your project, simply load the model using a deep learning framework compatible with .h5 files (e.g., TensorFlow, Keras). Here's an example of how to load the model in Python using Keras:

```python
from tensorflow.keras.models import load_model
model = load_model('traffic_classifier.h5')
```

## Installation

1. **Install the dependencies**:

   - Open a command prompt or terminal.
   - Use pip to install the required libraries:

     ```
     pip install opencv-python
     pip install numpy
     pip install tensorflow
     pip install pyttsx3
     ```

   - Ensure that the paths to the model files (`traffic_det.pt` and `traffic_classifier.h5`) are correctly set in the code before running the system.

2. **Run the system**:

   - Navigate to the directory containing the project files.
   - Run the main Python script using the Python interpreter:

     ```
     python main.py
     ```

   This will start the Traffic Sign Detection and Recognition system with the integrated Voice Alert.

### Troubleshooting

If you encounter any issues while using the Traffic Sign Detection and Recognition system, consider the following troubleshooting steps:

- **Issue**: Error message indicating missing dependencies.
  - **Solution**: Ensure that all required libraries and dependencies are installed correctly. Refer to the software requirements section for installation instructions.

- **Issue**: Incorrect model file paths or errors related to model loading.
  - **Solution**: Double-check that the paths to the model files (`traffic_det.pt` and `traffic_classifier.h5`) are correctly set in the code. Ensure that the files exist in the specified locations.

- **Issue**: Unexpected behavior or errors during runtime.
  - **Solution**: Check for any error messages in the terminal or console output. Review the code logic and make sure that all components are functioning as intended. If necessary, refer to the documentation or seek assistance from the project contributors.

### Contributing

Contributions to the Traffic Sign Detection and Recognition project are welcome! Here's how you can contribute:

- Report any bugs or issues encountered while using the system by creating a detailed GitHub issue.
- Suggest new features or enhancements that could improve the functionality of the system.
- Fork the project, make changes or improvements, and submit a pull request for review.
- Follow the project's contribution guidelines and coding standards when submitting code changes.
- Participate in discussions, provide feedback, and help address open issues or pull requests.
