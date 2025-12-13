"""
3D Object Localization using Ultralytics YOLO + Apple Depth Pro
===============================================================

This system estimates 3D coordinates (X, Y, Z in meters) of detected objects
using a static camera by combining:
1. YOLO for 2D object detection
2. Depth Pro for metric depth estimation
3. Camera intrinsics for 3D projection

Requirements:
    pip install ultralytics torch torchvision pillow numpy opencv-python
    pip install git+https://github.com/apple/ml-depth-pro.git
"""

import cv2
import numpy as np
import torch
from collections import deque
from PIL import Image
from pathlib import Path
from typing import Tuple, List, Dict
from filterpy.kalman import KalmanFilter

import time
import argparse

# Import YOLO from ultralytics
from ultralytics import YOLO

# Import Depth Pro
try:
    import depth_pro
except ImportError:
    print("Please install depth_pro: pip install git+https://github.com/apple/ml-depth-pro.git")
    raise
# Background subtraction approach

class DroneMotionDetector:
    def __init__(self):
        # Use advanced background subtractor
        self.bg_subtractor = cv2.createBackgroundSubtractorMOG2(
            history=500,
            varThreshold=16,
            detectShadows=False
        )
        self.kalman = cv2.KalmanFilter(4, 2)  # Track position

    def detect(self, frame):
        # Apply background subtraction
        fg_mask = self.bg_subtractor.apply(frame)

        # Threshold and clean
        _, fg_mask = cv2.threshold(fg_mask, 250, 255, cv2.THRESH_BINARY)
        fg_mask = cv2.medianBlur(fg_mask, 5)

        # Find moving objects
        contours, _ = cv2.findContours(fg_mask, cv2.RETR_EXTERNAL,
                                       cv2.CHAIN_APPROX_SIMPLE)

        # Select largest moving blob
        if contours:
            largest = max(contours, key=cv2.contourArea)
            if cv2.contourArea(largest) > 50:
                return cv2.boundingRect(largest)

        return None


class DroneTracker:
    def __init__(self):
        self.kf = KalmanFilter(dim_x=6, dim_z=2)

        # State: [x, y, vx, vy, ax, ay]
        self.kf.F = np.array([
            [1, 0, 1, 0, 0.5, 0],
            [0, 1, 0, 1, 0, 0.5],
            [0, 0, 1, 0, 1, 0],
            [0, 0, 0, 1, 0, 1],
            [0, 0, 0, 0, 1, 0],
            [0, 0, 0, 0, 0, 1]
        ])

        # Measurement: [x, y]
        self.kf.H = np.array([
            [1, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0]
        ])

        self.lost_frames = 0
        self.max_lost_frames = 10

    def update(self, detection):
        if detection is not None:
            x, y = detection[:2]
            self.kf.update([x, y])
            self.lost_frames = 0
        else:
            self.lost_frames += 1

        # Always predict next position
        self.kf.predict()
        predicted = self.kf.x[:2]

        if self.lost_frames > self.max_lost_frames:
            return None  # Track lost

        return predicted



class Object3DLocalizer:
    """
    Estimates 3D coordinates of objects detected in images using YOLO and Depth Pro.
    """

    def __init__(
        self,
        yolo_model: str = "./trained/best.pt",
        camera_params: Dict = None,
        device: str = None
    ):
        """
        Initialize the 3D object localizer.

        Args:
            yolo_model: YOLO model variant (yolo11n.pt, yolo11s.pt, yolo11m.pt, etc.)
            camera_params: Camera intrinsics dict with 'fx', 'fy', 'cx', 'cy' (optional)
            device: 'cuda' or 'cpu' (auto-detected if None)
        """
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")

        # Initialize YOLO
        print(f"Loading YOLO model: {yolo_model}")
        self.yolo_model = YOLO(yolo_model)
        self.motion_detector = DroneMotionDetector()
        self.tracker = DroneTracker()
        self.prev_frames = deque(maxlen=5)

        # Initialize Depth Pro
        print("Loading Depth Pro model...")
        self.depth_model, self.depth_transform = depth_pro.create_model_and_transforms(
            device=self.device
        )
        self.depth_model.eval()

        # Camera parameters (will be estimated by Depth Pro if not provided)
        self.camera_params = camera_params

        print("Initialization complete!")

    def estimate_depth(self, image_path: str) -> Tuple[np.ndarray, float]:
        """
        Estimate depth map using Depth Pro.

        Args:
            image_path: Path to input image

        Returns:
            depth_map: Depth in meters (H, W)
            focal_length_px: Estimated focal length in pixels
        """
        # Load and preprocess image
        image, _, f_px = depth_pro.load_rgb(image_path)
        image = self.depth_transform(image).to(self.device)

        # Run inference
        with torch.no_grad():
            prediction = self.depth_model.infer(image, f_px=f_px)

        depth = prediction["depth"].cpu().numpy()  # Depth in meters
        focal_length_px = prediction["focallength_px"]

        return depth, focal_length_px

    def pixel_to_3d(
        self,
        u: float,
        v: float,
        depth: float,
        focal_length_px: float,
        image_shape: Tuple[int, int]
    ) -> Tuple[float, float, float]:
        """
        Convert 2D pixel coordinates to 3D world coordinates.

        Args:
            u: Pixel x-coordinate
            v: Pixel y-coordinate
            depth: Depth at (u, v) in meters
            focal_length_px: Focal length in pixels
            image_shape: (height, width) of image

        Returns:
            (X, Y, Z) in meters relative to camera
        """
        h, w = image_shape

        # Use provided camera params or estimate from focal length
        if self.camera_params:
            fx = self.camera_params['fx']
            fy = self.camera_params['fy']
            cx = self.camera_params['cx']
            cy = self.camera_params['cy']
        else:
            # Assume square pixels and principal point at image center
            fx = fy = focal_length_px
            cx = w / 2.0
            cy = h / 2.0

        # Convert to 3D coordinates
        # Standard pinhole camera model
        X = (u - cx) * depth / fx
        Y = (v - cy) * depth / fy
        Z = depth

        return X, Y, Z

    def multi_scale_detect(self, frame, scales=[1.0, 0.75, 0.5]):
        best_detection = None
        best_confidence = 0

        for scale in scales:
            h, w = frame.shape[:2]
            resized = cv2.resize(frame, (int(w*scale), int(h*scale)))

            # Run detection on scaled image
            detections = self.yolo_model(
                resized,
                verbose=False
            )[0]

            if detections.confidence > best_confidence:
                # Scale bbox back to original coordinates
                bbox = detections.bbox / scale
                best_detection = bbox
                best_confidence = detections.confidence

        return best_detection

    # Hybrid detection pipeline
    def detect_drone(self, frame, confidence_threshold=0.5):
        # Primary: YOLO detection
        detections = self.multi_scale_detect(frame)

        if detections.confidence > confidence_threshold:
            method = "yolo"
        else:

            # Fallback: Classical CV methods
            # Method 1: Color-based detection (black quadcopter)
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            lower_black = np.array([0, 0, 0])
            upper_black = np.array([180, 255, 50])
            mask = cv2.inRange(hsv, lower_black, upper_black)

            # Method 2: Morphological operations
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5,5))
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

            # Find contours
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL,
                                            cv2.CHAIN_APPROX_SIMPLE)

            # Filter by area and aspect ratio
            valid_contours = []
            for cnt in contours:
                area = cv2.contourArea(cnt)
                if 100 < area < 10000:  # Adjust based on expected size
                    x, y, w, h = cv2.boundingRect(cnt)
                    aspect_ratio = w / float(h)
                    if 0.5 < aspect_ratio < 2.0:  # Roughly square
                        valid_contours.append((x, y, w, h))


            if valid_contours:
                detections = max(valid_contours, key=lambda x: x[2]*x[3])
                method = "classical"

            # 2. Fallback to motion detection
            elif len(self.prev_frames) >= 3:
                detections = self.motion_detector.detect(frame)
                method = "motion"

            # 3. Try template matching if we have previous detections
            else:
                detections = self.template_match(frame)
                method = "template"

        # 4. Update tracker (handles missing detections)
        position = self.tracker.update(detections)

        return position, method



    def detect_and_localize(
        self,
        image_path: str,
        conf_threshold: float = 0.25,
        iou_threshold: float = 0.7
    ) -> Tuple[np.ndarray, List[Dict]]:
        """
        Detect objects and estimate their 3D coordinates.

        Args:
            image_path: Path to input image
            conf_threshold: Confidence threshold for detections
            iou_threshold: IoU threshold for NMS

        Returns:
            annotated_image: Image with bounding boxes and 3D coordinates
            detections_3d: List of detection dicts with 3D info
        """
        start_time = time.time()

        # Load original image
        image = cv2.imread(image_path)
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        h, w = image.shape[:2]

        # Step 1: YOLO detection
        print("\n1. Running YOLO object detection...")
        yolo_start = time.time()
        results = self.detect_drone(
            image,
            confidence_threshold=conf_threshold,
        )[0]
        print(f"   YOLO inference: {time.time() - yolo_start:.3f}s")

        cv2.imshow("YOLO Detections", results.plot())
        cv2.waitKey(1)


        # # Step 2: Depth estimation
        # print("2. Running Depth Pro depth estimation...")
        # depth_start = time.time()
        # depth_map, focal_length_px = self.estimate_depth(image_path)
        # print(f"   Depth Pro inference: {time.time() - depth_start:.3f}s")

        # # Resize depth map to match image dimensions
        # if depth_map.shape != (h, w):
        #     depth_map = cv2.resize(depth_map, (w, h), interpolation=cv2.INTER_LINEAR)

        # # Step 3: Calculate 3D coordinates for each detection
        # print("3. Computing 3D coordinates...")
        # detections_3d = []

        # for box in results.boxes:
        #     # Get 2D bounding box
        #     x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
        #     confidence = float(box.conf[0])
        #     class_id = int(box.cls[0])
        #     class_name = results.names[class_id]

        #     # Get center point of bounding box
        #     center_u = (x1 + x2) / 2
        #     center_v = (y1 + y2) / 2

        #     # Ensure coordinates are within image bounds
        #     center_u = np.clip(int(center_u), 0, w - 1)
        #     center_v = np.clip(int(center_v), 0, h - 1)

        #     # Get depth at center point
        #     depth = depth_map[center_v, center_u]

        #     # Convert to 3D coordinates
        #     X, Y, Z = self.pixel_to_3d(
        #         center_u, center_v, depth, focal_length_px, (h, w)
        #     )

        #     # Store detection info
        #     detection = {
        #         'class_name': class_name,
        #         'class_id': class_id,
        #         'confidence': confidence,
        #         'bbox_2d': (x1, y1, x2, y2),
        #         'center_2d': (center_u, center_v),
        #         'coordinates_3d': (X, Y, Z),
        #         'depth': depth
        #     }
        #     detections_3d.append(detection)

        #     # Draw on image
        #     # Bounding box
        #     cv2.rectangle(image, (int(x1), int(y1)), (int(x2), int(y2)),
        #                  (0, 255, 0), 2)

        #     # Label with 3D coordinates
        #     label = f"{class_name} {confidence:.2f}"
        #     coord_text = f"3D: ({X:.2f}, {Y:.2f}, {Z:.2f})m"

        #     cv2.putText(image, label, (int(x1), int(y1) - 25),
        #                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        #     cv2.putText(image, coord_text, (int(x1), int(y1) - 5),
        #                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

        #     # Center point
        #     cv2.circle(image, (center_u, center_v), 5, (0, 0, 255), -1)

        #     # cv2.imshow("Depth Map", depth_map / np.max(depth_map))
        #     # cv2.imshow("3D Localization", image)
        #     # cv2.waitKey(1)

        total_time = time.time() - start_time
        print(f"\nTotal processing time: {total_time:.3f}s")
        print(f"Detected {len(detections_3d)} objects")

        return image, detections_3d

    def create_visualization(
        self,
        image: np.ndarray,
        depth_map: np.ndarray,
        detections_3d: List[Dict]
    ) -> np.ndarray:
        """
        Create side-by-side visualization of RGB, depth, and detections.

        Args:
            image: Original image with annotations
            depth_map: Depth map from Depth Pro
            detections_3d: List of 3D detections

        Returns:
            Combined visualization image
        """
        # Normalize depth map for visualization
        depth_vis = depth_map.copy()
        depth_vis = (depth_vis - depth_vis.min()) / (depth_vis.max() - depth_vis.min())
        depth_vis = 1.0 - depth_vis  # Invert for better visibility
        depth_vis = (depth_vis * 255).astype(np.uint8)
        depth_vis = cv2.applyColorMap(depth_vis, cv2.COLORMAP_TURBO)

        # Resize to match if needed
        h, w = image.shape[:2]
        if depth_vis.shape[:2] != (h, w):
            depth_vis = cv2.resize(depth_vis, (w, h))

        # Combine horizontally
        combined = np.hstack([image, depth_vis])

        return combined


def main():
    """
    Example usage of the 3D object localizer.
    """

    parser = argparse.ArgumentParser(description='Process image with object detection')
    parser.add_argument('--input', '-i', type=str, required=True, help='Path to input image file')
    parser.add_argument('--output', '-o', type=str, default='output.jpg', help='Path to output image file (default: output.jpg)')
    args = parser.parse_args()


    # Initialize localizer
    localizer = Object3DLocalizer(
        yolo_model="./trained/best.pt",  # Fastest YOLO11 model
        camera_params=None  # Will be estimated by Depth Pro
    )

    # Example: Process an image
    # image_path = "./train/DroneSmaller.png"  # Replace with your image path
    image_path = args.input  # Replace with your image path

    if not Path(image_path).exists():
        print(f"\nError: Image not found at {image_path}")
        print("Please provide a valid image path.")
        return

    # Detect and localize objects
    annotated_image, detections = localizer.detect_and_localize(
        image_path,
        conf_threshold=0.25,
        iou_threshold=0.7
    )

    # Print results
    print("\n" + "="*70)
    print("3D LOCALIZATION RESULTS")
    print("="*70)

    for i, det in enumerate(detections, 1):
        X, Y, Z = det['coordinates_3d']
        print(f"\n{i}. {det['class_name']} (confidence: {det['confidence']:.3f})")
        print(f"   2D Center: ({det['center_2d'][0]:.1f}, {det['center_2d'][1]:.1f}) px")
        print(f"   3D Position: X={X:.3f}m, Y={Y:.3f}m, Z={Z:.3f}m")
        print(f"   Distance from camera: {Z:.3f}m")

    # Save result
    output_path = args.output
    cv2.imwrite(output_path, annotated_image)
    print(f"\n✓ Annotated image saved to: {output_path}")

    output_path_depth = "output_depth_map.jpg"
    combined = localizer.create_visualization(
        annotated_image,
        localizer.estimate_depth(image_path)[0],
        detections
    )
    cv2.imwrite(output_path_depth, combined)
    print(f"✓ Combined visualization saved to: {output_path_depth}")


    # Optional: Display
    try:
        cv2.imshow("3D Object Localization", annotated_image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    except:
        print("(Display not available in this environment)")


if __name__ == "__main__":
    main()
