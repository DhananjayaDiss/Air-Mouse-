# Hand Gesture Mouse Controller

Control your mouse cursor using hand gestures via your webcam. This Python application uses computer vision to track your hand movements and translate them into mouse actions.

## Features

- **Intuitive Gesture Control**: Move your cursor by pointing with your index finger
- **Precision Mode**: Slow down cursor movement by partially raising your pinky
- **Cursor Lock**: Lock the cursor position for precise clicking by fully raising your pinky
- **Click Actions**: Perform clicks by holding your pinky up
- **Double-Click Support**: Automatic detection of double-click gestures
- **Drag Mode**: Drag objects by keeping pinky up after clicking
- **Visual Feedback**: On-screen indicators show cursor state, click progress, and mode status

## Installation

### Prerequisites

- Python 3.6+
- OpenCV
- MediaPipe
- NumPy
- Autopy (for mouse control)

### Setup

1. Clone this repository:
```bash
git clone https://github.com/yourusername/hand-gesture-mouse.git
cd hand-gesture-mouse
```

2. Install the required packages:
```bash
pip install opencv-python mediapipe numpy autopy
```

## Usage

1. Run the MouseController script:
```bash
python mouse_controller.py
```

2. Position your hand in front of the webcam.

3. Use the following gestures:
   - **Move cursor**: Raise only your index finger
   - **Slow cursor**: Begin to raise your pinky while keeping index finger up
   - **Lock cursor**: Fully raise your pinky for precision clicking
   - **Click**: Keep pinky raised for a short duration
   - **Double-click**: After a click, perform another click within the timeframe
   - **Drag**: Keep pinky up after clicking to enter drag mode
   - **Exit drag**: Lower your pinky to release

4. Press 'q' to exit the application.

## Gesture Guide

| Gesture | Description | Visual Indicator |
|---------|-------------|------------------|
| Index finger up | Basic cursor movement | Green cursor |
| Pinky partially up | Slow cursor movement | Orange cursor outline |
| Pinky fully up | Lock cursor position | Orange cursor with "LOCKED" text |
| Pinky held up | Click action | Red flash + "CLICK!" text |
| Quick second click | Double-click | Purple flash + "DOUBLE CLICK!" text |
| Pinky stays up after click | Drag mode | Purple cursor with trail effect |

## Configuration

You can customize the controller by modifying the parameters when creating the `MouseController` instance:

```python
controller = MouseController(
    smoothing=6,     # Higher = smoother but more lag (default: 8)
    speed=1.2        # Higher = faster cursor movement (default: 1.5)
)
```

Additional parameters can be modified directly in the class:

```python
self.roi_margin_x = 100  # Horizontal margin for region of interest
self.roi_margin_y = 100  # Vertical margin for region of interest
self.click_delay = 0.3   # Minimum seconds between clicks
self.click_confirmation_time = 0.3  # Seconds to hold pinky up for click
self.double_click_threshold = 2.0  # Maximum seconds between clicks for double-click
```

## How It Works

The system uses the `HandDetector` class (based on MediaPipe) to detect and track hand landmarks in real-time from your webcam feed. The `MouseController` class then:

1. Maps the position of your index fingertip to screen coordinates
2. Detects various finger positions and gestures
3. Applies smoothing algorithms to prevent cursor jitter
4. Implements a state machine to handle different interaction modes
5. Provides visual feedback on the webcam feed

## Requirements

- **Hardware**:
  - Webcam
  - Computer with sufficient processing power for real-time vision

- **Dependencies**:
  - OpenCV: For image processing and webcam input
  - MediaPipe: For hand landmark detection
  - NumPy: For numerical operations
  - Autopy: For controlling the mouse cursor

## Troubleshooting

- **Jerky Cursor Movement**: Increase the `smoothing` parameter
- **Cursor Too Fast/Slow**: Adjust the `speed` parameter
- **Difficulty Clicking**: Adjust the `click_confirmation_time` or `click_delay`
- **Detection Issues**: Ensure adequate lighting and clear background

## Known Limitations

- Works best with a single hand in frame
- Requires reasonable lighting conditions
- Performance depends on your computer's processing power
- Some operating systems may restrict programmatic mouse control

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- Google MediaPipe team for the hand tracking solution
- OpenCV community for computer vision tools
- Autopy developers for the mouse control library
