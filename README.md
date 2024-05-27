# Parking Space Detection with Mask R-CNN

This project uses Mask R-CNN to detect available parking spaces in a video stream and sends an SMS alert when a parking space is found. The implementation leverages the pre-trained COCO model for object detection and Twilio for sending SMS notifications.

## Requirements

- **Python 3.6+**
- **Libraries:**
  - `numpy`
  - `opencv-python`
  - `twilio`
  - `mask-rcnn` (from the Matterport repository)
- **COCO pre-trained weights** (will be downloaded automatically if not present)
- **Twilio Account** (for SMS notifications)

## Installation

1. **Clone the repository:**

    ```bash
    git clone https://github.com/your_username/parking-space-detection.git
    cd parking-space-detection
    ```

2. **Install required Python packages:**

    ```bash
    pip install numpy opencv-python twilio
    pip install git+https://github.com/matterport/Mask_RCNN.git
    ```

3. **Set up your project directory structure:**

    ```
    parking-space-detection/
    ├── images/
    │   └── test_images/
    │       └── parking.mp4
    ├── logs/
    ├── mask_rcnn_coco.h5  # This will be downloaded automatically if not present
    ├── your_script.py     # Your Python script file
    ```

4. **Twilio Configuration:**

    Obtain the following credentials from your Twilio account:
    - `twilio_account_sid`
    - `twilio_auth_token`
    - `twilio_phone_number`
    - `destination_phone_number`

    Replace the placeholders in `your_script.py` with your actual Twilio credentials:

    ```python
    twilio_account_sid = 'YOUR_TWILIO_SID'
    twilio_auth_token = 'YOUR_TWILIO_AUTH_TOKEN'
    twilio_phone_number = 'YOUR_TWILIO_SOURCE_PHONE_NUMBER'
    destination_phone_number = 'THE_PHONE_NUMBER_TO_TEXT'
    ```

## Running the Project

1. **Ensure the video file `parking.mp4` is placed in the correct directory:**

    ```
    parking-space-detection/
    ├── images/
    │   └── test_images/
    │       └── parking.mp4
    ```

2. **Run the script:**

    ```bash
    python your_script.py
    ```

3. **Exit the script:**

    Press 'q' to quit the video display window.

## Notes

- **GPU Support:** If you have a GPU and want to leverage it for faster inference, ensure you have the necessary CUDA and cuDNN libraries installed and modify the Mask R-CNN configuration accordingly.
- **Video Source:** The script is currently set to process a video file (`parking.mp4`). To use a webcam instead, set `VIDEO_SOURCE = 0` in the script.

## Credits

- This project uses the [Mask R-CNN implementation](https://github.com/matterport/Mask_RCNN) by Matterport.
- Twilio is used for sending SMS notifications.

## License

This project is licensed under the MIT License. See the LICENSE file for details.

