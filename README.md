# CameraCalibration
Measures camera distortion values and computes the translation vector by which the image should be shifted.
It uses opencv function  [`cv.calibrateCamera()`](https://docs.opencv.org/4.x/dc/dbb/tutorial_py_calibration.html)
***
## How to use
1. Print a [chessboard](checkerboard_radon.png). 
2. Make sure to install numpy and opencv.
3. Run command:
``` 
python get_distortion_from_camera.py [CAMERA_NAME]
```
4. Move the chessboard in front of the camera until the program finishes.
5. The distortion parameters are stored in the file `camera_distortion.yaml` (by default in the code directory).

### Parameters 
* `-path [PATH]` optional parameter to set, where to store the camera distortion values. By default, is the directory, where the code is located.
* `-num_pic [NUMBER]` optional parameter to set the amount of pictures to make for calibration. More pictures, more accurate calibration. By default, is set to 20.
* `-demo` optional flag to use a demo mode. During the demo mode the camera image is showed along with current calibration applied.