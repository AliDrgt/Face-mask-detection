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
    ```

## Arguments

- `--face`: Path to the face detector model directory (default: `deploy.prototxt`)
- `--model`: Path to the trained face mask detector model (default: `mask_detector1K.model`)
- `--confidence`: Minimum probability to filter weak detections (default: `0.5`)

## How It Works

1. The script loads the serialized face detector model and the trained face mask detector model.
2. It initializes the video stream and starts capturing frames from the webcam.
3. For each frame, it detects faces and predicts whether they are wearing a mask or not.
4. Bounding boxes and labels are drawn on the frame, and the results are displayed in a window.
5. Bounding box coordinates are logged with timestamps for further analysis.



## Contributing

Contributions are welcome! Please feel free to submit a pull request or open an issue to discuss any changes.


## Acknowledgments

This project was created by Ali Durgut.
