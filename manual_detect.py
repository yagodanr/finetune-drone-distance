import cv2
import numpy as np
from collections import deque
import sys
import csv
import pandas as pd

class DronePositionAnalyzer:
    def __init__(self, video_path):
        self.video_path = video_path
        self.cap = cv2.VideoCapture(video_path)

        if not self.cap.isOpened():
            raise ValueError(f"Cannot open video file: {video_path}")

        # Video properties
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        # Tracking variables
        self.reference_point = None
        self.drone_center = None
        self.trajectory = deque(maxlen=500)
        self.distances = deque(maxlen=500)
        self.previous_frame = 0
        self.current_frame = 0

        # UI state
        self.paused = True
        self.tracking_box = None
        self.selecting_roi = False

        # Statistics
        self.max_distance = 0
        self.avg_distance = 0

        # Calibration
        self.initial_pixels_per_meter = None
        self.initial_drone_height_meters = None
        self.drone_size_meters = 0.3  # Default, user can change
        self.initial_size_px = None
        self.current_height = None

        # Improved tracking
        self.tracker = None
        self.kalman = None
        self.threshold = 100  # For contour detection, adjustable if needed
        self.last_known_center = None
        self.last_known_box = None

        # Coordinate storage
        self.coords = []
        self.frame_data = []

        # Load flight log if exists
        try:
            df = pd.read_csv('flight_log.csv')
            self.log_times = df['timestamp'].values
            self.log_z = df['z'].values
            print("Flight log loaded successfully.")
        except Exception as e:
            self.log_times = None
            self.log_z = None
            print(f"Failed to load flight_log.csv: {e}")

        print("Drone Position Hold Analyzer (Improved with Dynamic Height and CSV Export)")
        print("==========================================================================")
        print("Controls:")
        print("  SPACE - Play/Pause")
        print("  R - Select new ROI (drone)")
        print("  C - Clear reference point")
        print("  Left Click - Set reference point")
        print("  H - Set initial drone height and size for calibration")
        print("  Q/ESC - Quit")
        print("\nInstructions:")
        print("1. Press 'H' to enter initial drone height in meters and size")
        print("2. Press 'R' and drag to select the drone")
        print("3. Click to set reference point")
        print("4. Press SPACE to start tracking")
        print("CSV with (x,y,z) coordinates will be exported at the end.")
        if self.log_times is not None:
            print("Flight log detected; will refine coordinates using actual heights.")

    def setup_kalman(self):
        """Setup Kalman filter for smoothing position"""
        self.kalman = cv2.KalmanFilter(4, 2)  # State: [x, y, vx, vy], Measurement: [x, y]
        self.kalman.measurementMatrix = np.array([[1, 0, 0, 0], [0, 1, 0, 0]], np.float32)
        self.kalman.transitionMatrix = np.array([[1, 0, 1, 0], [0, 1, 0, 1], [0, 0, 1, 0], [0, 0, 0, 1]], np.float32)
        self.kalman.processNoiseCov = np.eye(4, dtype=np.float32) * 0.03
        self.kalman.measurementNoiseCov = np.eye(2, dtype=np.float32) * 1.0

    def select_roi(self, frame):
        """Let user select drone ROI"""
        print("\nSelect drone area and press ENTER or SPACE")
        roi = cv2.selectROI("Select Drone", frame, False, False)
        cv2.destroyWindow("Select Drone")

        if roi[2] > 0 and roi[3] > 0:
            self.tracking_box = roi
            # Initialize tracker - Use CSRT for better accuracy on sky background and scale changes
            self.tracker = cv2.TrackerCSRT_create()
            self.tracker.init(frame, self.tracking_box)
            # Setup Kalman
            self.setup_kalman()
            center_x = roi[0] + roi[2] / 2
            center_y = roi[1] + roi[3] / 2
            self.kalman.statePre = np.array([[center_x], [center_y], [0], [0]], dtype=np.float32)
            self.last_known_center = (int(center_x), int(center_y))
            self.last_known_box = roi
            # Set initial size if calibrated
            if self.initial_drone_height_meters is not None:
                self.set_initial_size()
            return True
        return False

    def set_initial_size(self):
        if self.tracking_box is not None:
            x, y, w, h = self.tracking_box
            self.initial_size_px = np.sqrt(w**2 + h**2)
            print(f"Initial drone size set: {self.initial_size_px:.2f} px")

    def mouse_callback(self, event, x, y, flags, param):
        """Handle mouse clicks for setting reference point"""
        if event == cv2.EVENT_LBUTTONDOWN:
            self.reference_point = (x, y)
            print(f"Reference point set at: ({x}, {y})")

    def calculate_distance(self, p1, p2):
        """Calculate Euclidean distance between two points"""
        return np.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)

    def calibrate_from_height(self, height_meters, drone_size_meters):
        """
        Calibrate pixel-to-meter conversion using initial drone height and drone size.
        """
        if self.tracking_box is not None:
            self.set_initial_size()
            self.initial_pixels_per_meter = self.initial_size_px / drone_size_meters
            self.initial_drone_height_meters = height_meters
            self.drone_size_meters = drone_size_meters

            print(f"\nCalibration set:")
            print(f"  Initial drone height: {height_meters}m")
            print(f"  Drone size: {drone_size_meters}m (diagonal)")
            print(f"  Initial pixels per meter: {self.initial_pixels_per_meter:.2f}")
        else:
            print("Please select the drone first!")

    def update_current_height_and_scale(self):
        """Update current height and pixels per meter based on size change"""
        if self.initial_size_px is not None and self.tracking_box is not None:
            x, y, w, h = self.tracking_box
            current_size_px = np.sqrt(w**2 + h**2)
            if current_size_px > 0:
                self.current_height = self.initial_drone_height_meters * (self.initial_size_px / current_size_px)
                current_pixels_per_meter = self.initial_pixels_per_meter * (current_size_px / self.initial_size_px)
                return current_pixels_per_meter
        return self.initial_pixels_per_meter

    def pixels_to_meters(self, pixels, pixels_per_meter):
        """Convert pixel distance to meters using current scale"""
        if pixels_per_meter is None:
            return None
        return pixels / pixels_per_meter

    def format_distance(self, pixels, pixels_per_meter):
        """Format distance string with both pixels and meters"""
        meters = self.pixels_to_meters(pixels, pixels_per_meter)
        if meters is not None:
            if meters < 1.0:
                return f"{pixels:.1f}px ({meters*100:.1f}cm)"
            else:
                return f"{pixels:.1f}px ({meters:.2f}m)"
        return f"{pixels:.1f}px"

    def format_height(self):
        """Format current height"""
        if self.current_height is not None:
            return f"{self.current_height:.2f}m"
        return "N/A"

    def find_drone_contour(self, frame, roi=None):
        """Detect drone using contours on sky background (assuming drone is darker)"""
        if roi is not None:
            # Crop to ROI
            x, y, w, h = roi
            crop = frame[y:y+h, x:x+w]
            if crop.size == 0:
                return None, None
        else:
            crop = frame

        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        # Gaussian blur to reduce noise
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        # Threshold - invert since drone is dark
        _, thresh = cv2.threshold(blurred, self.threshold, 255, cv2.THRESH_BINARY_INV)
        # Morphology to clean up
        kernel = np.ones((5,5), np.uint8)
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
        # Find contours
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            # Largest contour
            largest = max(contours, key=cv2.contourArea)
            if cv2.contourArea(largest) > 100:  # Min area
                # Bounding box relative to crop
                bx, by, bw, bh = cv2.boundingRect(largest)
                # Absolute if ROI
                if roi is not None:
                    bx += x
                    by += y
                # Center
                cx = bx + bw // 2
                cy = by + bh // 2
                return (cx, cy), (bx, by, bw, bh)
        return None, None

    def get_search_roi(self):
        """Get expanded ROI around last known position for redetection"""
        if self.last_known_center and self.last_known_box:
            cx, cy = self.last_known_center
            _, _, lw, lh = self.last_known_box
            expand_factor = 2.0  # Expand by 2x
            ew = int(lw * expand_factor)
            eh = int(lh * expand_factor)
            ex = max(0, cx - ew // 2)
            ey = max(0, cy - eh // 2)
            ex = min(ex, self.width - ew)
            ey = min(ey, self.height - eh)
            return (ex, ey, ew, eh)
        return None

    def draw_visualization(self, frame):
        """Draw all visualization elements on frame"""
        vis_frame = frame.copy()

        # Draw tracking box
        if self.tracking_box is not None:
            x, y, w, h = [int(v) for v in self.tracking_box]
            cv2.rectangle(vis_frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

            # Draw drone center
            if self.drone_center is not None:
                cv2.circle(vis_frame, self.drone_center, 5, (0, 255, 0), -1)

        # Draw reference point
        if self.reference_point is not None:
            cv2.circle(vis_frame, self.reference_point, 8, (0, 0, 255), -1)
            cv2.circle(vis_frame, self.reference_point, 15, (0, 0, 255), 2)

            # Draw line and distance if drone is tracked
            if self.drone_center is not None:
                cv2.line(vis_frame, self.reference_point, self.drone_center, (255, 0, 255), 2)

                # Calculate distance
                distance_px = self.calculate_distance(self.reference_point, self.drone_center)
                current_ppm = self.update_current_height_and_scale()
                self.distances.append(distance_px)
                self.trajectory.append(self.drone_center)

                # Update statistics
                self.max_distance = max(self.max_distance, distance_px)
                if len(self.distances) > 0:
                    self.avg_distance = np.mean(self.distances)

                # Draw distance text
                mid_point = ((self.reference_point[0] + self.drone_center[0]) // 2,
                            (self.reference_point[1] + self.drone_center[1]) // 2)
                distance_text = self.format_distance(distance_px, current_ppm)
                cv2.putText(vis_frame, distance_text,
                           (mid_point[0] + 10, mid_point[1] - 10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        # Draw trajectory
        if len(self.trajectory) > 1:
            points = np.array(self.trajectory, dtype=np.int32)
            for i in range(1, len(points)):
                alpha = i / len(points)
                color = (int(255 * alpha), int(100 * (1 - alpha)), int(255 * (1 - alpha)))
                cv2.line(vis_frame, tuple(points[i-1]), tuple(points[i]), color, 2)

        # Draw statistics panel
        panel_height = 200
        panel = np.zeros((panel_height, self.width, 3), dtype=np.uint8)

        y_offset = 25

        # Calibration status
        if self.initial_pixels_per_meter:
            cv2.putText(panel, f"Calibration: {self.initial_drone_height_meters}m init height, {self.drone_size_meters}m size, {self.initial_pixels_per_meter:.1f}px/m init",
                       (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (100, 255, 100), 1)
        else:
            cv2.putText(panel, "Not calibrated - Press 'H' to set height and size",
                       (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (100, 100, 255), 1)

        y_offset += 25
        cv2.putText(panel, f"Frame: {self.current_frame}/{self.total_frames}",
                   (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        y_offset += 25
        cv2.putText(panel, f"Current Height: {self.format_height()}",
                   (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 255), 1)

        if self.drone_center and self.reference_point:
            y_offset += 30
            current_text = self.format_distance(self.distances[-1], current_ppm)
            cv2.putText(panel, f"Current Distance: {current_text}",
                       (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
            y_offset += 30
            max_text = self.format_distance(self.max_distance, current_ppm)  # Note: max is in px, but for display use current ppm approx
            cv2.putText(panel, f"Max Distance: {max_text}",
                       (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
            y_offset += 30
            avg_text = self.format_distance(self.avg_distance, current_ppm)  # Approx
            cv2.putText(panel, f"Avg Distance: {avg_text}",
                       (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)

        y_offset += 30
        status = "PAUSED" if self.paused else "PLAYING"
        cv2.putText(panel, f"Status: {status}",
                   (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 1)

        # Combine frame and panel
        combined = np.vstack([vis_frame, panel])
        return combined

    def refine_coordinates(self):
        if self.log_times is None or not self.frame_data or self.initial_pixels_per_meter is None:
            return

        times_est = np.array([d['time'] for d in self.frame_data])
        z_est = np.array([d['z_est'] for d in self.frame_data])
        if np.any(z_est is None):
            print("Incomplete estimated heights; skipping refinement.")
            return

        min_error = float('inf')
        best_offset = 0
        for offset in np.arange(-10, 10.1, 0.033):  # Step size matching approximate log interval
            times_shift = times_est + offset
            z_log_interp = np.interp(times_shift, self.log_times, self.log_z, left=self.log_z[0], right=self.log_z[-1])
            error = np.mean((z_est - z_log_interp)**2)
            if error < min_error:
                min_error = error
                best_offset = offset

        print(f"Found best time offset: {best_offset:.3f}s with MSE {min_error:.4f}")

        # Refine coords
        f = self.initial_pixels_per_meter * self.initial_drone_height_meters
        for i, d in enumerate(self.frame_data):
            time = d['time'] + best_offset
            z_actual = np.interp(time, self.log_times, self.log_z, left=self.log_z[0], right=self.log_z[-1])
            ppm = f / z_actual if z_actual != 0 else 1
            x_m = d['dx_px'] / ppm
            y_m = d['dy_px'] / ppm
            self.coords[i]['x'] = x_m
            self.coords[i]['y'] = y_m
            self.coords[i]['z'] = z_actual

    def run(self):
        """Main processing loop"""
        cv2.namedWindow("Drone Position Hold Analysis")
        cv2.setMouseCallback("Drone Position Hold Analysis", self.mouse_callback)

        # Read first frame
        ret, frame = self.cap.read()
        if not ret:
            print("Cannot read video")
            return

        # Initial ROI selection
        self.select_roi(frame)
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

        while True:
            if not self.paused:
                ret, frame = self.cap.read()
                if not ret:
                    print("\nEnd of video reached")
                    self.paused = True
                    self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    self.current_frame = 0
                    continue
                self.previous_frame = self.current_frame
                self.current_frame += 1

                # Update tracker
                if self.tracker is not None:
                    success, box = self.tracker.update(frame)
                    if success:
                        self.tracking_box = box
                        # Calculate raw center
                        x, y, w, h = box
                        raw_center = np.array([[x + w / 2], [y + h / 2]], dtype=np.float32)
                        # Kalman correct
                        self.kalman.correct(raw_center)
                        # Kalman predict
                        predicted = self.kalman.predict()
                        self.drone_center = (int(predicted[0, 0]), int(predicted[1, 0]))
                        self.last_known_center = self.drone_center
                        self.last_known_box = box
                    else:
                        print("Tracking lost! Attempting redetection...")
                        search_roi = self.get_search_roi()
                        center, box = self.find_drone_contour(frame, roi=search_roi)
                        if box is not None:
                            self.tracking_box = box
                            self.tracker.init(frame, self.tracking_box)
                            self.drone_center = center
                            self.last_known_center = center
                            self.last_known_box = box
                            print("Drone redetected and tracker reinitialized.")
                        else:
                            # Try whole frame if ROI fails
                            center, box = self.find_drone_contour(frame)
                            if box is not None:
                                self.tracking_box = box
                                self.tracker.init(frame, self.tracking_box)
                                self.drone_center = center
                                self.last_known_center = center
                                self.last_known_box = box
                                print("Drone redetected in full frame.")
                            else:
                                self.drone_center = None
                                self.paused = True
                                print("Redetection failed. Please reselect ROI with 'R'.")
            else:
                # If paused, stay on current frame
                current_pos = self.cap.get(cv2.CAP_PROP_POS_FRAMES)
                if current_pos > 0:
                    self.cap.set(cv2.CAP_PROP_POS_FRAMES, current_pos - 1)
                ret, frame = self.cap.read()
                if not ret:
                    break

            # Update height and scale, record coordinates if possible
            if self.drone_center is not None and self.reference_point is not None and self.current_frame != 0 and self.current_frame != self.previous_frame:
                dx_px = self.drone_center[0] - self.reference_point[0]
                dy_px = self.drone_center[1] - self.reference_point[1]
                time = self.current_frame / self.fps if self.fps > 0 else self.current_frame * 0.033  # Fallback assume 30fps
                current_ppm = self.update_current_height_and_scale()
                z_est = self.current_height
                self.frame_data.append({'frame': self.current_frame, 'dx_px': dx_px, 'dy_px': dy_px, 'time': time, 'z_est': z_est})

                x_m = self.pixels_to_meters(dx_px, current_ppm) if current_ppm else dx_px
                y_m = self.pixels_to_meters(dy_px, current_ppm) if current_ppm else dy_px
                z_m = z_est if z_est is not None else 0
                self.coords.append({'frame': self.current_frame, 'x': x_m, 'y': y_m, 'z': z_m})

            # Draw visualization
            vis_frame = self.draw_visualization(frame)
            cv2.imshow("Drone Position Hold Analysis", vis_frame)

            # Handle keyboard input
            key = cv2.waitKey(30) & 0xFF

            if key == ord('q') or key == 27:  # Q or ESC
                break
            elif key == ord(' '):  # SPACE
                self.paused = not self.paused
                print("Paused" if self.paused else "Playing")
            elif key == ord('r') or key == ord('R'):  # R - reselect ROI
                self.paused = True
                self.select_roi(frame)
            elif key == ord('c') or key == ord('C'):  # C - clear reference
                self.reference_point = None
                self.trajectory.clear()
                self.distances.clear()
                self.max_distance = 0
                self.avg_distance = 0
                print("Reference point cleared")
            elif key == ord('h') or key == ord('H'):  # H - set height and size
                self.paused = True
                print("\n=== Calibration ===")
                try:
                    height = float(input("Enter initial drone height above camera (meters): "))
                    if height <= 0:
                        print("Height must be positive!")
                        continue
                    size = float(input("Enter drone diagonal size (meters, default 0.3): ") or "0.3")
                    if size <= 0:
                        print("Size must be positive!")
                        continue
                    self.calibrate_from_height(height, size)
                except ValueError:
                    print("Invalid input! Please enter numbers.")

        # Refine if possible
        self.refine_coordinates()

        self.cap.release()
        cv2.destroyAllWindows()

        # Export coordinates to CSV
        if self.coords:
            csv_filename = 'drone_coords.csv'
            with open(csv_filename, 'w', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=['frame', 'x', 'y', 'z'])
                writer.writeheader()
                writer.writerows(self.coords)
            print(f"\nCoordinates exported to {csv_filename}")
        else:
            print("\nNo coordinates recorded. Ensure calibration and tracking are set up.")

        # Print final statistics
        print("\n=== Final Statistics ===")
        if len(self.distances) > 0:
            # For final stats, use approximate ppm since distances are in px
            approx_ppm = self.initial_pixels_per_meter if self.initial_pixels_per_meter else None
            print(f"Maximum Distance: {self.format_distance(self.max_distance, approx_ppm)}")
            print(f"Average Distance: {self.format_distance(self.avg_distance, approx_ppm)}")
            print(f"Minimum Distance: {self.format_distance(min(self.distances), approx_ppm)}")
            print(f"Frames Analyzed: {len(self.distances)}")

            if self.initial_pixels_per_meter:
                print(f"\nCalibration used: {self.initial_drone_height_meters}m initial height, {self.drone_size_meters}m size")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python drone_tracker.py <video_path>")
        print("Example: python drone_tracker.py drone_test.mp4")
        sys.exit(1)

    video_path = sys.argv[1]

    try:
        analyzer = DronePositionAnalyzer(video_path)
        analyzer.run()
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)