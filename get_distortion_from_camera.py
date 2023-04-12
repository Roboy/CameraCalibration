import os
import numpy as np
import cv2 as cv
import yaml
from yaml import CLoader as Loader
import argparse


def calculate_shift_vector(mtx, image_size):
    """
    :param mtx: camera matrix from cv.calibrateCamera().
    :param image_size: image size of the original image.
    :return: translation vector by which the original image should be shifted.
    """

    img_size = np.array(image_size)
    img_size = np.flip(img_size)
    center = np.array([mtx[0][2], mtx[1][2]])
    shift_vector = img_size / 2 - center
    shift_vector = np.round(shift_vector)
    return shift_vector


def add_cross(img):
    """
    Add red cross in the center of the image.
    :param img:
    :return:
    """
    num_rows, num_cols = img.shape[:2]
    cv.line(img, (int(num_cols / 2), 0), (int(num_cols / 2), num_rows), (0, 0, 255), 5)
    cv.line(img, (0, int(num_rows / 2)), (num_cols, int(num_rows / 2)), (0, 0, 255), 5)


def get_camera(camera_name):
    """
    :param camera_name: parameter for cv.VideoCapture(camera_name).
    :return: opnecv camera instance.
    """

    webcam = cv.VideoCapture(camera_name)

    codec = 0x47504A4D  # MJPG
    webcam.set(cv.CAP_PROP_FPS, 30.0)
    webcam.set(cv.CAP_PROP_FOURCC, codec)
    webcam.set(cv.CAP_PROP_FRAME_WIDTH, 1920)
    webcam.set(cv.CAP_PROP_FRAME_HEIGHT, 1080)
    return webcam


def param_extract(objpoints, imgpoints, shape):
    calibration_flags = cv.fisheye.CALIB_RECOMPUTE_EXTRINSIC + cv.fisheye.CALIB_FIX_SKEW
    N_OK = len(objpoints)
    K = np.zeros((3, 3))
    D = np.zeros((4, 1))
    rvecs = [np.zeros((1, 1, 3), dtype=np.float64) for i in range(N_OK)]
    tvecs = [np.zeros((1, 1, 3), dtype=np.float64) for i in range(N_OK)]
    objpoints_np = np.asarray(objpoints, dtype=np.float64)
    imgpoints_np = np.asarray(imgpoints, dtype=np.float64)

    objpoints_np = np.reshape(objpoints_np, (N_OK, 1, 14 * 9, 3))
    imgpoints_np = np.reshape(imgpoints_np, (N_OK, 1, 14 * 9, 2))
    try:
        rms, _, _, _, _ = \
            cv.fisheye.calibrate(
                objpoints_np,
                imgpoints_np,
                shape,
                K,
                D,
                rvecs,
                tvecs,
                calibration_flags,
                (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 1e-6)
            )
    except:
        objpoints = objpoints[:-1].copy()
        imgpoints = imgpoints[:-1].copy()
        rms = False

    return rms, K, D, rvecs, tvecs


def calibrate_camera(camera=1, number_of_pictures=20, demo=False):
    """
    Uses opencv function to record and extract camera distortion values.
    :param camera: VideoCapture or parameter for cv.VideoCapture(camera_name).
    :param number_of_pictures: amount of pictures for calibration.
    :param demo: flag to show camera output.
    :return: camera_distortion, shift_vector
    """

    if hasattr(camera, "read"):
        webcam = camera
    else:
        webcam = get_camera(camera)

    # termination criteria
    criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
    objp = np.zeros((14 * 9, 3), np.float32)
    objp[:, :2] = np.mgrid[0:14, 0:9].T.reshape(-1, 2)
    # Arrays to store object points and image points from all the images.
    objpoints = []  # 3d point in real world space
    imgpoints = []  # 2d points in image plane.

    while True:
        print("Move the chessboard to the different position in front of the camera. Pictures left: ", number_of_pictures)
        cv.waitKey(1000)
        print("Hold the chessboard still...")
        cv.waitKey(500)
        print("3...")
        cv.waitKey(500)
        print("2...")
        cv.waitKey(500)
        print("1")
        print(chr(7))
        (_, img) = webcam.read()
        gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        # Find the chess board corners
        ret, corners = cv.findChessboardCorners(gray, (14, 9), None)
        # If found, add object points, image points (after refining them)
        if ret == True:

            # Add 3d points of the chessboard
            objpoints.append(objp)

            # Add positions of the chessboard w.r.t. image
            corners2 = cv.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
            imgpoints.append(corners2)

            if demo:
                # Draw and display image from camera after translation

                cv.drawChessboardCorners(img, (14, 9), corners2, ret)

                # retrieve distortion values from recorded images
                # ret, mtx, dist, rvecs, tvecs = cv.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)
                ret, mtx, dist, rvecs, tvecs = param_extract(objpoints, imgpoints, gray.shape[::-1])

                shift_vector = calculate_shift_vector(mtx, img.shape[:2])

                shift_matrix = np.float32([[1, 0, shift_vector[0]], [0, 1, shift_vector[1]]])
                num_rows, num_cols = img.shape[:2]
                img_translation = cv.warpAffine(img, shift_matrix, (num_cols, num_rows))

                imgS = cv.resize(img_translation, (960, 540))
                add_cross(imgS)
                cv.imshow('img', imgS)
                # cv.waitKey(500)

            # reduce amount of images left
            number_of_pictures -= 1
            if number_of_pictures <= 0:
                break

        else:
            if demo:
                # display pure image from camera
                imgS = cv.resize(img, (960, 540))
                add_cross(imgS)
                cv.imshow('img', imgS)
                # cv.waitKey(100)

    cv.destroyAllWindows()

    # retrieve distortion values from recorded images
    # ret, mtx, dist, rvecs, tvecs = cv.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)
    ret, mtx, dist, rvecs, tvecs = param_extract(objpoints, imgpoints, gray.shape[::-1])
    shift_vector = calculate_shift_vector(mtx, img.shape[:2])

    camera_distortion = {
        "ret": ret,
        "mtx": mtx,
        "dist": dist
    }
    return camera_distortion, shift_vector


def update_calibration(camera=1, number_of_pictures=20, path=None):
    """
    Do the calibration and save new parameters to the file
    :param camera: VideoCapture or parameter for cv.VideoCapture(camera_name).
    :param number_of_pictures: amount of pictures for calibration.
    :param path: directory, where to store camera distortion.
    :return:
    """
    if path is None:
        path = os.path.dirname(os.path.realpath(__file__))
    camera_distortion, image_shift = calibrate_camera(camera, number_of_pictures, demo=False)
    save_parameters(camera_distortion, image_shift, path)


def save_parameters(camera_distortion, image_shift, path):
    """
    Write a YAML representation of camera_distortion to 'camera_distortion.yaml'
    :param camera_distortion: camera distortion from opencv function.
    :param image_shift: calculated translation vector.
    :param path: directory, where to store camera distortion.
    :return:
    """

    if not os.path.exists(path):
        os.makedirs(path)

    if os.path.isdir(path):
        filename = 'camera_distortion.yaml'
        full_path = os.path.join(path, filename)
    else:
        full_path = path
    stream = open(full_path, 'w')
    data = camera_distortion.copy()
    if hasattr(image_shift, 'tolist'):
        image_shift = image_shift.tolist()
    data['image_shift'] = image_shift
    yaml.dump(data, stream)

    print("Calibration values are stored in: ", full_path)


def read_image_shift(path):
    """
    Read image shift vector from file
    :param path: directory, where camera distortion file is located.
    :return: image shift (translation) vector.
    """
    all_distortion = read_camera_distortion(path)
    return all_distortion['image_shift']


def read_camera_distortion(path):
    """
     Read camera distortion values from file.
    :param path: directory, where camera distortion file is located.
    :return: camera distortion (as a dictionary: ret, mtx, dist, rvecs, tvecs)
    """
    if os.path.isdir(path):
        filename = 'camera_distortion.yaml'
        full_path = os.path.join(path, filename)
    else:
        full_path = path
    stream = open(full_path, 'r')
    data = yaml.load(stream, Loader)
    return data


if __name__ == "__main__":

    # Read arguments
    parser = argparse.ArgumentParser("Camera calibration")
    parser.add_argument("camera_name", help="Name of the camera, which calibrate", type=str)
    parser.add_argument("-num_pic", help="Amount of photos to take during the calibration. The default "
                                         "value is 20. More photos, better precision", nargs='?',
                        const=20, default=20, type=int)
    parser.add_argument("-path", help="Path where to store camera parameters. Default is code location.", nargs='?',
                        const=None, default=None, type=str)
    parser.add_argument('-demo', help="Demo mode will show images with opencv.", action='store_const', default=False,
                        const=True)
    args = parser.parse_args()

    camera_name = int(args.camera_name)
    path = args.path
    number_of_pictures = args.num_pic
    demo = args.demo

    # if directory does not exist, create it
    if path is None:
        path = os.path.dirname(os.path.realpath(__file__))

    print("Camera name: ", camera_name)
    print("Path: ", path)
    print("Number of pictures", number_of_pictures)
    print("Demo: ", demo)

    # measure calibration values
    camera_distortion, image_shift = calibrate_camera(camera_name, number_of_pictures, demo)

    # save values to the file
    save_parameters(camera_distortion, image_shift, path)
