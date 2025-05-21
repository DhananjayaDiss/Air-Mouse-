import cv2
import numpy as np
import autopy  # For mouse control (install with: pip install autopy)
import math
import time
from HandTrackingModule import HandDetector  # Your module


class MouseController:
    def __init__(self, smoothing=8, speed=1.5):
        # Initialize hand detector
        self.detector = HandDetector(detection_confidence=0.7, max_hands=1)

        # Screen dimensions
        self.screen_width, self.screen_height = autopy.screen.size()

        # Frame dimensions (to be determined from camera)
        self.frame_width, self.frame_height = 0, 0

        # Mouse control parameters
        self.smoothing_factor = smoothing  # Higher = smoother but more lag
        self.speed_factor = speed  # Higher = faster cursor movement

        # Region of interest (portion of frame used for cursor control)
        self.roi_margin_x = 100  # Margins from frame edge
        self.roi_margin_y = 100

        # Previous cursor positions for smoothing
        self.prev_positions = []

        # Click parameters
        self.clicking = False
        self.click_delay = 0.3  # seconds between clicks
        self.click_confirmation_time = 0.3  # Time to hold pinky up to confirm click
        self.last_click_time = 0
        self.lock_timeout = 1.5  # seconds before auto-unlocking if no click happens
        self.lock_start_time = 0

        # Mode control
        self.mouse_active = False
        self.click_mode = False
        self.drag_mode = False  # New: tracking drag mode
        self.drag_start_time = 0  # New: time when drag started

        # Cursor lock for clicking
        self.cursor_locked = False
        self.locked_position = (0, 0)  # Screen coordinates
        self.locked_frame_position = (0, 0)  # Frame coordinates for visualization

        # Tracking for pinky state
        self.pinky_was_down = True
        self.pinky_up_time = 0
        self.pinky_start_to_raise = False
        self.pinky_raise_level = 0  # 0 to 1, how much the pinky is raised
        self.pinky_kept_up_after_click = False  # New: tracking if pinky stays up after click

        # Double click tracking
        self.awaiting_second_click = False
        self.first_click_time = 0
        self.double_click_threshold = 2.0  # INCREASED time for double-click (now 2 seconds, more than 6x click confirmation)
        self.double_click_delay = 0.05  # delay between clicks in a double-click

    def check_index_finger_only(self, hand_landmarks):
        """
        Check if only the index finger is up (for mouse movement)

        Args:
            hand_landmarks: List of landmarks for a hand

        Returns:
            bool: True if only index finger is up, False otherwise
        """
        if len(hand_landmarks) < 21:
            return False

        # Get positions for fingertips and knuckles
        fingertips = [4, 8, 12, 16, 20]  # thumb, index, middle, ring, pinky tips
        knuckles = [2, 6, 10, 14, 18]  # corresponding knuckles/base joints

        # Check if each finger is up or down
        fingers_up = []

        # Special check for thumb
        thumb_tip = hand_landmarks[fingertips[0]]
        thumb_mcp = hand_landmarks[knuckles[0]]
        # Thumb is considered "up" if it's to the left/right of the MCP joint based on hand type
        if hand_landmarks[0][4] == "Right":  # Check hand type
            thumb_up = thumb_tip[1] < thumb_mcp[1]  # x-coordinate comparison
        else:
            thumb_up = thumb_tip[1] > thumb_mcp[1]
        fingers_up.append(thumb_up)

        # Check for other 4 fingers
        for i in range(1, 5):
            finger_tip = hand_landmarks[fingertips[i]]
            finger_pip = hand_landmarks[knuckles[i]]
            # Finger is up if tip's y-coordinate is less than PIP joint's y-coordinate
            fingers_up.append(finger_tip[2] < finger_pip[2])

        # Return true if ONLY index is up (fingers_up[1]) and others are down
        return not fingers_up[0] and fingers_up[1] and not fingers_up[2] and not fingers_up[3] and not fingers_up[4]

    def is_pinky_up(self, hand_landmarks):
        """
        Check if the pinky finger is up

        Args:
            hand_landmarks: List of landmarks for a hand

        Returns:
            bool: True if pinky finger is up, False otherwise
        """
        if len(hand_landmarks) < 21:
            return False

        # Get pinky fingertip and PIP joint
        pinky_tip = hand_landmarks[20]
        pinky_pip = hand_landmarks[18]

        # Pinky is up if its tip is above its PIP joint
        return pinky_tip[2] < pinky_pip[2]

    def is_pinky_starting_to_raise(self, hand_landmarks):
        """
        Check if the pinky finger is starting to raise (not fully up yet)

        Args:
            hand_landmarks: List of landmarks for a hand

        Returns:
            bool: True if pinky finger is starting to raise, False otherwise
        """
        if len(hand_landmarks) < 21:
            return False

        # Get pinky fingertip, DIP joint, and PIP joint
        pinky_tip = hand_landmarks[20]
        pinky_dip = hand_landmarks[19]
        pinky_pip = hand_landmarks[18]

        # Check if pinky tip is higher than DIP but not necessarily higher than PIP
        return pinky_tip[2] < pinky_dip[2]  # If tip is higher than DIP joint

    def get_pinky_raise_level(self, hand_landmarks):
        """
        Calculate how much the pinky is raised (0 to 1)

        Args:
            hand_landmarks: List of landmarks for a hand

        Returns:
            float: Level of pinky raise from 0 (down) to 1 (fully up)
        """
        if len(hand_landmarks) < 21:
            return 0

        # Get pinky landmarks
        pinky_tip = hand_landmarks[20]
        pinky_dip = hand_landmarks[19]
        pinky_pip = hand_landmarks[18]
        pinky_mcp = hand_landmarks[17]

        # Calculate level based on relative position to joints
        # Different calculation method - more reliable
        if pinky_tip[2] < pinky_pip[2]:  # Fully up - tip above PIP
            return 1.0
        elif pinky_tip[2] < pinky_dip[2]:  # Partially up - between DIP and PIP
            # Normalize between DIP and PIP
            total_range = pinky_dip[2] - pinky_pip[2]
            if total_range == 0:  # Avoid division by zero
                return 0.5
            current_pos = pinky_dip[2] - pinky_tip[2]
            return 0.5 + (current_pos / total_range) * 0.5
        else:  # Below DIP
            # Normalize between MCP and DIP
            total_range = pinky_mcp[2] - pinky_dip[2]
            if total_range == 0:  # Avoid division by zero
                return 0
            current_pos = pinky_mcp[2] - pinky_tip[2]
            return max(0, min(0.5, (current_pos / total_range) * 0.5))

    def get_smooth_position(self, x, y, slow_factor=1.0):
        """
        Apply smoothing to cursor movement using moving average
        with adjustable slow factor

        Args:
            x, y: Input coordinates
            slow_factor: Factor to slow down movement (1.0 = normal, higher = slower)

        Returns:
            tuple: Smoothed (x, y) coordinates
        """
        # If we have previous positions
        if self.prev_positions:
            # Get the last position
            last_x, last_y = self.prev_positions[-1]

            # Apply slow factor (move only a fraction of the way to the new position)
            adjusted_x = last_x + (x - last_x) / slow_factor
            adjusted_y = last_y + (y - last_y) / slow_factor
        else:
            # No previous positions, use as-is
            adjusted_x, adjusted_y = x, y

        # Add to history
        self.prev_positions.append((adjusted_x, adjusted_y))

        # Keep only the last N positions for moving average
        if len(self.prev_positions) > self.smoothing_factor:
            self.prev_positions.pop(0)

        # Calculate average position
        avg_x = sum(p[0] for p in self.prev_positions) / len(self.prev_positions)
        avg_y = sum(p[1] for p in self.prev_positions) / len(self.prev_positions)

        return avg_x, avg_y

    def map_to_screen(self, x, y):
        """Map frame coordinates to screen coordinates with bounds checking"""
        # Define active zone (ROI)
        roi_x1 = self.roi_margin_x
        roi_y1 = self.roi_margin_y
        roi_x2 = self.frame_width - self.roi_margin_x
        roi_y2 = self.frame_height - self.roi_margin_y

        # Ensure input coordinates are within ROI
        x = max(roi_x1, min(x, roi_x2))
        y = max(roi_y1, min(y, roi_y2))

        # Normalize coordinates to ROI
        x_ratio = (x - roi_x1) / (roi_x2 - roi_x1)
        y_ratio = (y - roi_y1) / (roi_y2 - roi_y1)

        # Map to screen coordinates
        screen_x = x_ratio * self.screen_width
        screen_y = y_ratio * self.screen_height

        return screen_x, screen_y

    def detect_click(self, hand_landmarks):
        """
        Detect click, double-click, and drag mode based on pinky finger being raised

        Args:
            hand_landmarks: List of landmarks for a hand

        Returns:
            tuple: (clicked, double_clicked) booleans
        """
        if len(hand_landmarks) < 21:
            return (False, False)

        # Check if index is up (required for any mouse action)
        index_tip = hand_landmarks[8]
        index_pip = hand_landmarks[6]
        index_up = index_tip[2] < index_pip[2]

        if not index_up:
            self.pinky_was_down = True
            self.pinky_start_to_raise = False
            self.cursor_locked = False
            self.drag_mode = False  # Exit drag mode if index goes down
            return (False, False)

        # Get pinky state
        pinky_up = self.is_pinky_up(hand_landmarks)
        pinky_level = self.get_pinky_raise_level(hand_landmarks)
        self.pinky_raise_level = pinky_level

        now = time.time()

        # Lock cursor when pinky level is high enough (don't change this part)
        if pinky_up and not self.cursor_locked:
            self.cursor_locked = True
            self.locked_position = (self.locked_screen_x, self.locked_screen_y)
            self.locked_frame_position = (self.locked_frame_x, self.locked_frame_y)
            self.lock_start_time = now

        # Auto-unlock if timeout is reached and no click/drag is happening
        if self.cursor_locked and not self.click_mode and not self.drag_mode:
            if now - self.lock_start_time > self.lock_timeout:
                self.cursor_locked = False

        # Unlock cursor when pinky goes completely down (preserve existing behavior)
        if pinky_level < 0.2 and self.cursor_locked and not self.click_mode and not self.drag_mode:
            self.cursor_locked = False

        # Detect a single click
        clicked = False
        double_clicked = False

        # Only allow click if we've seen the pinky go down before
        if pinky_up and self.pinky_was_down:
            self.pinky_was_down = False
            self.pinky_up_time = now
            self.click_mode = True

        # Click happens when pinky has been up for a short time
        if self.click_mode and pinky_up:
            # Visual indication but don't click yet
            duration = now - self.pinky_up_time

            # Check if pinky has been up long enough to trigger a click
            if duration > self.click_confirmation_time and now - self.last_click_time > self.click_delay:
                clicked = True

                # DOUBLE-CLICK LOGIC
                if self.awaiting_second_click and (now - self.first_click_time) < self.double_click_threshold:
                    # This is a double click
                    double_clicked = True
                    self.awaiting_second_click = False

                    # Execute two quick clicks
                    autopy.mouse.click()
                    time.sleep(self.double_click_delay)
                    autopy.mouse.click()
                else:
                    # This is a single click
                    self.first_click_time = now
                    self.awaiting_second_click = True
                    autopy.mouse.click()

                self.last_click_time = now
                self.click_mode = False
                self.pinky_kept_up_after_click = True  # Track that pinky was kept up after click

                # Start drag mode if pinky stays up after click
                if pinky_up and not double_clicked:
                    self.drag_mode = True
                    self.drag_start_time = now
                    autopy.mouse.toggle(down=True)  # Press mouse down for dragging

        # Reset if pinky goes down
        elif not pinky_up:
            # End drag mode if it was active
            if self.drag_mode:
                self.drag_mode = False
                autopy.mouse.toggle(down=False)  # Release mouse button

            self.pinky_was_down = True
            self.click_mode = False
            self.pinky_kept_up_after_click = False

            # Reset double-click tracking if too much time has passed
            if self.awaiting_second_click and (now - self.first_click_time) > self.double_click_threshold:
                self.awaiting_second_click = False

        return (clicked, double_clicked)

    def run(self):
        # Initialize camera
        cap = cv2.VideoCapture(0)

        # Get frame dimensions
        success, img = cap.read()
        if success:
            self.frame_height, self.frame_width = img.shape[:2]

        # Drawing parameters
        active_color = (0, 255, 0)  # Green
        inactive_color = (0, 0, 255)  # Red
        cursor_color = (255, 255, 0)  # Cyan
        locked_color = (255, 165, 0)  # Orange
        click_color = (0, 0, 255)  # Red
        double_click_color = (255, 0, 255)  # Purple
        drag_color = (128, 0, 255)  # Purple-ish

        # Initialize cursor position variables
        self.locked_screen_x, self.locked_screen_y = 0, 0
        self.locked_frame_x, self.locked_frame_y = 0, 0

        try:
            while True:
                # Read frame
                success, img = cap.read()
                if not success:
                    print("Failed to get frame")
                    break

                # Flip horizontally for mirror effect
                img = cv2.flip(img, 1)

                # Find hands
                img = self.detector.find_hands(img, flip_type=False)  # Already flipped
                hands = self.detector.find_positions(img)

                # Default status message
                status = "Mouse: INACTIVE (raise ONLY index finger)"
                status_color = inactive_color

                if hands:
                    # Use the first hand detected
                    hand = hands[0]

                    # Check finger states
                    only_index_up = self.check_index_finger_only(hand)
                    pinky_up = self.is_pinky_up(hand)
                    pinky_raising = self.is_pinky_starting_to_raise(hand)
                    pinky_level = self.pinky_raise_level

                    # Mouse is active if at least the index finger is up
                    index_tip = hand[8]
                    index_pip = hand[6]
                    index_up = index_tip[2] < index_pip[2]

                    if index_up:
                        self.mouse_active = True

                        # Status messages - prioritize drag mode
                        now = time.time()
                        if self.drag_mode:
                            status = "DRAG MODE: Move to drag objects"
                            status_color = drag_color
                        elif self.awaiting_second_click and (now - self.first_click_time) < self.double_click_threshold:
                            time_left = self.double_click_threshold - (now - self.first_click_time)
                            status = f"Double-click ready! ({time_left:.1f}s)"
                            status_color = double_click_color
                        elif self.cursor_locked:
                            status = "Cursor: LOCKED (for precise clicking)"
                            status_color = locked_color
                        elif pinky_up:
                            status = "Click Mode: ACTIVE (pinky raised)"
                            status_color = click_color
                        elif pinky_level > 0.3:
                            status = f"Slowing Cursor: {int(pinky_level * 100)}%"
                            status_color = (255, 165, 0)  # Orange
                        else:
                            status = "Mouse: ACTIVE (controlling cursor)"
                            status_color = active_color
                    else:
                        self.mouse_active = False

                    # Extract landmark positions when active
                    if self.mouse_active and len(hand) > 20:
                        # Get index finger tip position
                        index_x, index_y = index_tip[1], index_tip[2]

                        # Calculate slow factor based on pinky raise level
                        # 1.0 = normal speed, higher = slower
                        if pinky_level > 0.3 and pinky_level < 0.8:
                            # Progressive slowdown as pinky raises
                            # At 30% = 1.5x slower, at 50% = 3x slower, at 70% = 5x slower
                            slow_factor = 1.0 + (pinky_level * 6)
                        else:
                            slow_factor = 1.0

                        # Apply smoothing with slow factor
                        if self.cursor_locked:
                            # Use locked position
                            smooth_x, smooth_y = self.locked_frame_position
                            screen_x, screen_y = self.locked_position
                        else:
                            # Use current position with appropriate slowdown
                            smooth_x, smooth_y = self.get_smooth_position(index_x, index_y, slow_factor)
                            screen_x, screen_y = self.map_to_screen(smooth_x, smooth_y)
                            # Store for locking
                            self.locked_frame_x, self.locked_frame_y = smooth_x, smooth_y
                            self.locked_screen_x, self.locked_screen_y = screen_x, screen_y

                        # Draw cursor with appropriate color/size
                        if self.drag_mode:
                            # Drag mode cursor
                            cv2.circle(img, (int(smooth_x), int(smooth_y)),
                                       15, drag_color, cv2.FILLED)
                            cv2.circle(img, (int(smooth_x), int(smooth_y)),
                                       20, drag_color, 2)
                            # Show "DRAGGING" text
                            cv2.putText(img, "DRAGGING",
                                        (int(smooth_x) + 20, int(smooth_y)),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, drag_color, 2)
                        elif self.cursor_locked:
                            # Locked cursor (orange)
                            cv2.circle(img, (int(self.locked_frame_x), int(self.locked_frame_y)),
                                       15, locked_color, cv2.FILLED)
                            cv2.circle(img, (int(self.locked_frame_x), int(self.locked_frame_y)),
                                       20, locked_color, 2)
                            # Show "LOCKED" text
                            cv2.putText(img, "LOCKED",
                                        (int(self.locked_frame_x) + 20, int(self.locked_frame_y)),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, locked_color, 2)
                        elif pinky_level > 0.3:
                            # Slowing down cursor (orange outline)
                            cv2.circle(img, (int(smooth_x), int(smooth_y)),
                                       12, (255, 165, 0), 2)
                            cv2.circle(img, (int(smooth_x), int(smooth_y)),
                                       10, cursor_color, cv2.FILLED)

                            # Show slow factor
                            cv2.putText(img, f"Slow: {slow_factor:.1f}x",
                                        (int(smooth_x) + 20, int(smooth_y)),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 165, 0), 1)
                        else:
                            # Normal cursor
                            cv2.circle(img, (int(smooth_x), int(smooth_y)), 10, cursor_color, cv2.FILLED)

                        # Move mouse cursor (even in drag mode)
                        try:
                            autopy.mouse.move(screen_x, screen_y)
                        except:
                            print("Mouse movement out of bounds")

                        # Check for click (pinky up)
                        if self.click_mode and pinky_up:
                            duration = time.time() - self.pinky_up_time
                            # Show click-pending status
                            if duration < self.click_confirmation_time:
                                click_progress = int(duration * 100 / self.click_confirmation_time)  # 0-100%
                                cv2.putText(img, f"Click pending... {click_progress}%",
                                            (index_tip[1] - 50, index_tip[2] - 20),
                                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, click_color, 2)
                            else:
                                # Show double-click potential if applicable
                                if self.awaiting_second_click and (
                                        time.time() - self.first_click_time) < self.double_click_threshold:
                                    cv2.putText(img, "DOUBLE CLICK READY!",
                                                (index_tip[1] - 80, index_tip[2] - 20),
                                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, double_click_color, 2)
                                else:
                                    cv2.putText(img, "CLICK!",
                                                (index_tip[1] - 30, index_tip[2] - 20),
                                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, click_color, 2)

                        # Display drag mode indicators
                        if self.drag_mode:
                            drag_duration = time.time() - self.drag_start_time
                            cv2.putText(img, f"Dragging: {drag_duration:.1f}s",
                                        (20, 120),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, drag_color, 2)
                            # Draw trail effect
                            if len(self.prev_positions) > 2:
                                for i in range(1, len(self.prev_positions)):
                                    cv2.line(img,
                                             (int(self.prev_positions[i - 1][0]), int(self.prev_positions[i - 1][1])),
                                             (int(self.prev_positions[i][0]), int(self.prev_positions[i][1])),
                                             drag_color, 2)

                        # Display double-click window info
                        if self.awaiting_second_click:
                            time_remaining = self.double_click_threshold - (time.time() - self.first_click_time)
                            if time_remaining > 0:
                                cv2.putText(img, f"Double-click window: {time_remaining:.1f}s",
                                            (20, 90),
                                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, double_click_color, 2)

                                # Draw countdown bar
                                bar_width = 200
                                filled_width = int(bar_width * (1 - time_remaining / self.double_click_threshold))
                                cv2.rectangle(img, (20, 105), (20 + bar_width, 115), (100, 100, 100), cv2.FILLED)
                                cv2.rectangle(img, (20, 105), (20 + filled_width, 115), double_click_color, cv2.FILLED)

                        # Show pinky finger status with different colors
                        pinky_tip = hand[20]
                        if self.drag_mode:
                            pinky_color = drag_color  # Purple for dragging
                        elif pinky_up:
                            pinky_color = click_color  # Red for clicking
                        elif pinky_level > 0.3:
                            # Gradient from gray to orange based on level
                            pinky_color = (
                                128,  # B
                                max(0, int(165 * (1 - pinky_level))),  # G
                                min(255, int(165 + pinky_level * 90))  # R
                            )
                        else:
                            pinky_color = (128, 128, 128)  # Gray for inactive

                        cv2.circle(img, (pinky_tip[1], pinky_tip[2]), 8, pinky_color, cv2.FILLED)

                        # Show pinky level as percentage
                        cv2.putText(img, f"Pinky: {int(pinky_level * 100)}%",
                                    (20, 150),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, pinky_color, 1)

                        # Detect and perform click action
                        clicked, double_clicked = self.detect_click(hand)
                        if clicked:
                            # Visual feedback for click
                            click_pos = (
                                int(self.locked_frame_x), int(self.locked_frame_y)) if self.cursor_locked else (
                                int(smooth_x), int(smooth_y))

                            if double_clicked:
                                # Double-click feedback
                                cv2.circle(img, click_pos, 30, double_click_color, cv2.FILLED)
                                cv2.putText(img, "DOUBLE CLICK!",
                                            (click_pos[0] - 60, click_pos[1] - 40),
                                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, double_click_color, 2)
                            else:
                                # Single-click feedback
                                cv2.circle(img, click_pos, 20, click_color, cv2.FILLED)

                # Draw ROI (region of interest)
                roi_x1 = self.roi_margin_x
                roi_y1 = self.roi_margin_y
                roi_x2 = self.frame_width - self.roi_margin_x
                roi_y2 = self.frame_height - self.roi_margin_y
                cv2.rectangle(img, (roi_x1, roi_y1), (roi_x2, roi_y2), (255, 0, 0), 2)

                # Show control instructions
                cv2.putText(img, status, (20, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, status_color, 2)
                cv2.putText(img, "Pinky: SLOW->LOCK->CLICK->DRAG", (20, 60),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

                # Calculate and display FPS
                self.detector.update_fps()
                self.detector.display_fps(img, position=(20, 180))

                # Show frame
                cv2.imshow("Advanced Mouse Control", img)

                # Exit on 'q' press
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

        except KeyboardInterrupt:
            print("Program interrupted by user")
        except Exception as e:
            print(f"Error occurred: {e}")
            import traceback
            traceback.print_exc()
        finally:
            # Release resources
            if self.drag_mode:
                # Ensure mouse button is released if program exits during drag
                autopy.mouse.toggle(down=False)

            cap.release()
            cv2.destroyAllWindows()
            print("Resources released")


if __name__ == "__main__":
    controller = MouseController(smoothing=6, speed=1.2)
    controller.run()