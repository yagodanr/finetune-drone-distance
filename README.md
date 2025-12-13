# Estimate position of the drone in space


## Prerequisites
- requirement.txt
- [Depth model from Apple](https://github.com/apple/ml-depth-pro.git)



## Manual detection
```
python manual_detection.py <video_path>
```
Opens up window with the first frame. Select the drone by dragging bounding box. Enter

Press H and input initial height and diagonal size of the drone in meters in the terminal. Enter


## Yolo
processes video
```
python engine.py [-h] --input INPUT [--output OUTPUT]
```


## Yolo + Depth
processes image
```
python engine_depth_model.py [-h] --input INPUT [--output OUTPUT]
```