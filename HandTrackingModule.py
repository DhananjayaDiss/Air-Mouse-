import cv2
import mediapipe as mp
import time


class HandDetector:
    """
    Basic Hand Detector module using MediaPipe.
    Detects hands and provides landmarks.
    """

    def __init__(self, mode=False, max_hands=2, detection_confidence=0.5, tracking_confidence=0.5):
        """
        Initialize the hand detector.

        Args:
            mode (bool): If True, detection runs on every frame
            max_hands (int): Maximum number of hands to detect
            detection_confidence (float): Minimum confidence for hand detection
            tracking_confidence (float): Minimum confidence for landmark tracking
        """
        self.mode = mode
        self.max_hands = max_hands
        self.detection_conf = detection_confidence
        self.tracking_conf = tracking_confidence

        # MediaPipe setup
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=self.mode,
            max_num_hands=self.max_hands,
            min_detection_confidence=self.detection_conf,
            min_tracking_confidence=self.tracking_conf
        )
        self.mp_draw = mp.solutions.drawing_utils

        # Results
        self.results = None

        # For FPS calculation
        self.prev_time = 0
        self.current_time = 0
        self.fps = 0

        # Reference for landmark indices
        '''
        HAND LANDMARKS:
        0: WRIST
        1-4: THUMB (1:CMC, 2:MCP, 3:IP, 4:TIP)
        5-8: INDEX FINGER (5:MCP, 6:PIP, 7:DIP, 8:TIP)
        9-12: MIDDLE FINGER (9:MCP, 10:PIP, 11:DIP, 12:TIP)
        13-16: RING FINGER (13:MCP, 14:PIP, 15:DIP, 16:TIP)
        17-20: PINKY (17:MCP, 18:PIP, 19:DIP, 20:TIP)
        '''

    def find_hands(self, img, draw=True, flip_type=True):
        """
        Process an image and detect hands.

        Args:
            img (numpy.ndarray): Input image (BGR format)
            draw (bool): Whether to draw landmarks and connections
            flip_type (bool): Flip image horizontally for mirror effect

        Returns:
            numpy.ndarray: Processed image with hand landmarks drawn (if draw=True)
        """
        # Make a copy to avoid modifying the original
        img_output = img.copy()

        # Flip image horizontally for a selfie-view display if needed
        if flip_type:
            img_output = cv2.flip(img_output, 1)

        # Convert to RGB
        img_rgb = cv2.cvtColor(img_output, cv2.COLOR_BGR2RGB)

        # Process the image
        self.results = self.hands.process(img_rgb)

        # Draw landmarks if detected
        if self.results and self.results.multi_hand_landmarks and draw:
            for hand_landmarks in self.results.multi_hand_landmarks:
                self.mp_draw.draw_landmarks(
                    img_output, hand_landmarks, self.mp_hands.HAND_CONNECTIONS)

        return img_output

    def find_positions(self, img):
        """
        Find positions of all hand landmarks from all detected hands.

        Args:
            img (numpy.ndarray): Input image

        Returns:
            list: List of hand data, where each hand is a list of landmark positions [id, x, y]
        """
        all_hands = []
        h, w, c = img.shape

        if self.results and self.results.multi_hand_landmarks:
            for hand_idx, hand_landmarks in enumerate(self.results.multi_hand_landmarks):
                # Get hand type
                hand_type = "Right"
                if self.results.multi_handedness:
                    hand_type = self.results.multi_handedness[hand_idx].classification[0].label

                # Get all landmarks for this hand
                hand_landmarks_list = []
                for lm_idx, landmark in enumerate(hand_landmarks.landmark):
                    px, py = int(landmark.x * w), int(landmark.y * h)
                    # Store [id, x, y, z, hand_type]
                    hand_landmarks_list.append([lm_idx, px, py, landmark.z, hand_type])

                all_hands.append(hand_landmarks_list)

        return all_hands

    def find_position(self, img, hand_no=0, draw=True, draw_radius=5):
        """
        Legacy method for backward compatibility.
        Find hand landmark positions of a specific hand.

        Args:
            img (numpy.ndarray): Input image
            hand_no (int): Index of the hand to track
            draw (bool): Whether to draw circles at landmark positions
            draw_radius (int): Radius of the circles

        Returns:
            list: List of [id, x, y] for each landmark
        """
        landmark_list = []

        if self.results and self.results.multi_hand_landmarks:
            if hand_no < len(self.results.multi_hand_landmarks):
                hand = self.results.multi_hand_landmarks[hand_no]

                for idx, landmark in enumerate(hand.landmark):
                    h, w, c = img.shape
                    px, py = int(landmark.x * w), int(landmark.y * h)
                    landmark_list.append([idx, px, py])

                    if draw:
                        cv2.circle(img, (px, py), draw_radius, (255, 0, 255), cv2.FILLED)

        return landmark_list

    def update_fps(self):
        """
        Calculate and update the frames per second.

        Returns:
            int: Current FPS value
        """
        self.current_time = time.time()
        fps = 1 / (self.current_time - self.prev_time) if self.prev_time > 0 else 0
        self.prev_time = self.current_time
        self.fps = int(fps)
        return self.fps

    def display_fps(self, img, position=(10, 30), font=cv2.FONT_HERSHEY_SIMPLEX,
                    scale=1, color=(0, 255, 0), thickness=2):
        """
        Display FPS on the image.

        Args:
            img (numpy.ndarray): Input image
            position (tuple): Position to display the FPS text
            font: OpenCV font type
            scale (float): Font scale
            color (tuple): RGB color
            thickness (int): Text thickness

        Returns:
            numpy.ndarray: Image with FPS displayed
        """
        cv2.putText(img, f"FPS: {self.fps}", position, font, scale, color, thickness)
        return img