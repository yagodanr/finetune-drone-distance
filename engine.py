from ultralytics import YOLO
from sahi import AutoDetectionModel
from sahi.predict import get_sliced_prediction


# Load a model
# model = YOLO("yolo11n-pose.pt")  # load an official model
# model = YOLO("train/yoloe-11l-seg.onnx")  # load an official model
# model = YOLO("./trained/best.pt")
model = AutoDetectionModel.from_pretrained(
    model_type="ultralytics",
    model_path="./trained/best.pt",  # Your custom model path
    confidence_threshold=0.3,
    device="cpu"  # Or "cpu"
)
# # Define your specific prompts that you want to bake into the model
# names = ["flying drone", "flyging quadcopter", "quadcopter", "drone", "drone from below"]

# # Set the classes and get text embeddings BEFORE export
# model.set_classes(names, model.get_text_pe(names))


# Predict with the model
# results = model.track("video_2025-12-08_18-03-22.mp4", show=True)  # predict on an image
# results = model.track("video_2025-12-08_15-46-28.mp4", show=True)  # predict on an image

# results = predict(
#     model=model,
#     image_path="video_2025-12-08_18-03-22.mp4",
#     device="cpu",
#     return_results=True,
#     conf_thres=0.3,
#     iou_thres=0.5,
#     max_det=1000,
#     show=True
# )
# Access the results
# for result in results:
#     # xy = result.keypoints.xy  # x and y coordinates
#     # xyn = result.keypoints.xyn  # normalized
#     # kpts = result.keypoints.data  # x, y, visibility (if available)

#     boxes = result.boxes.xyxy.cpu().tolist()  # x1, y1, x2, y2
#     cls = result.boxes.cls.cpu().tolist()  # class
#     for box, c in zip(boxes, cls):
#         print(f"Box: {box}, Class: {model.names[c]}")


# Use OpenCV or similar to read video frames
import cv2
cap = cv2.VideoCapture("video_2025-12-08_15-46-28.mp4")
while cap.isOpened():
    ret, frame = cap.read()
    if not ret: break
    result = get_sliced_prediction(frame, model, slice_height=640, slice_width=640)
    # Process result.object_prediction_list per frame
    # Correct bbox access - use direct attributes
    for obj in result.object_prediction_list:
        bbox = obj.bbox  # BoundingBox object
        x1, y1, x2, y2 = int(bbox.minx), int(bbox.miny), int(bbox.maxx), int(bbox.maxy)
        score = obj.score
        category_name = obj.category.name

        print(f"Box: [{x1},{y1},{x2},{y2}], Score: {score}, Category: {category_name}")

        # Draw bounding box
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        label = f"{category_name}: {score}"
        cv2.putText(frame, label, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    cv2.imshow("Frame", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
