# Face Mask Detection

A real-time face mask detection system using OpenCV, TensorFlow, and Keras.

## Features

- Real-time video stream analysis
- Detection of faces and prediction of mask usage
- Bounding box and label display on the detected faces
- Logging of bounding box coordinates with timestamps

## Files

- `deploy.prototxt`: Defines the architecture of the face detector model
- `detect_mask.py`: Main script for running the face mask detection
- `mask_detector1K.model`: Trained face mask detector model

## Installation

1. Clone the repository:
    ```sh
    git clone https://github.com/username/face-mask-detection.git
    ```
2. Navigate to the project directory:
    ```sh
    cd face-mask-detection
    ```
3. (Optional) Create a virtual environment and activate it:
    ```sh
    python -m venv env
    source env/bin/activate  # On Windows use `env\Scripts\activate`
    ```
4. Install the required packages:
    ```sh
    pip install -r requirements.txt
    ```

## Usage

Run the face mask detection using the following command:
```sh
python detect_mask.py --face deploy.prototxt --model mask_detector1K.model
