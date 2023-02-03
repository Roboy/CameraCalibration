import json
import os

import numpy as np
import cv2 as cv
import glob
import yaml
from yaml import CLoader as Loader
import argparse


def get_shift_vector(mtx, image_size):
    img_size = np.array(image_size)
    img_size = np.flip(img_size)
    center = np.array([mtx[0][2], mtx[1][2]])
    shift_vector = img_size / 2 - center
    shift_vector = np.round(shift_vector)
    return shift_vector


def add_cross(img):
    num_rows, num_cols = img.shape[:2]
    cv.line(img, (int(num_cols / 2), 0), (int(num_cols / 2), num_rows), (0, 0, 255), 5)
    cv.line(img, (0, int(num_rows / 2)), (num_cols, int(num_rows / 2)), (0, 0, 255), 5)

def get_camera(camera_name):
    webcam = cv.VideoCapture(camera_name)

    codec = 0x47504A4D  # MJPG
    webcam.set(cv.CAP_PROP_FPS, 30.0)
    webcam.set(cv.CAP_PROP_FOURCC, codec)
    webcam.set(cv.CAP_PROP_FRAME_WIDTH, 1920)
    webcam.set(cv.CAP_PROP_FRAME_HEIGHT, 1080)
    return webcam


def calibrate_camera(camera_name=1, number_of_pictures=20, demo=False):

    webcam = get_camera(camera_name)

    # termination criteria
    criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
    objp = np.zeros((14 * 9, 3), np.float32)
    objp[:, :2] = np.mgrid[0:14, 0:9].T.reshape(-1, 2)
    # Arrays to store object points and image points from all the images.
    objpoints = []  # 3d point in real world space
    imgpoints = []  # 2d points in image plane.

    while True:
        print("Move chessboard in front of the camera. Pictures left: ", number_of_pictures)
        (_, img) = webcam.read()
        gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        # Find the chess board corners
        ret, corners = cv.findChessboardCorners(gray, (14, 9), None)
        # If found, add object points, image points (after refining them)
        if ret == True:
            objpoints.append(objp)
            corners2 = cv.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
            imgpoints.append(corners2)
            # Draw and display the corners

            cv.drawChessboardCorners(img, (14, 9), corners2, ret)

            ret, mtx, dist, rvecs, tvecs = cv.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)
            shift_vector = get_shift_vector(mtx, img.shape[:2])

            shift_matrix = np.float32([[1, 0, shift_vector[0]], [0, 1, shift_vector[1]]])
            num_rows, num_cols = img.shape[:2]
            img_translation = cv.warpAffine(img, shift_matrix, (num_cols, num_rows))

            if demo:
                imgS = cv.resize(img_translation, (960, 540))
                add_cross(imgS)
                cv.imshow('img', imgS)
                cv.waitKey(500)

            number_of_pictures -= 1
            if number_of_pictures <= 0:
                break

        else:
            if demo:
                imgS = cv.resize(img, (960, 540))
                add_cross(imgS)
                cv.imshow('img', imgS)
                cv.waitKey(100)

    cv.destroyAllWindows()

    ret, mtx, dist, rvecs, tvecs = cv.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)
    camera_distortion = {
        "ret": ret,
        "mtx": mtx,
        "dist": dist
    }
    return camera_distortion, shift_vector


def update_calibration(camera_name=1, number_of_pictures=20, path=None):
    if path is None:
        path = os.path.dirname(os.path.realpath(__file__))
    camera_distortion, image_shift = calibrate_camera(camera_name, number_of_pictures, demo=False)
    save_parameters(camera_distortion, image_shift, path)


def save_parameters(camera_distortion, image_shift, path):
    # Write a YAML representation of camera_distortion to 'camera_distortion.yaml'.
    filename = 'camera_distortion.yaml'
    full_path = os.path.join(path, filename)
    if not os.path.exists(path):
        os.makedirs(path)
    stream = open(full_path, 'w')
    data = camera_distortion.copy()
    data['image_shift'] = image_shift
    yaml.dump(data, stream)

    print("Calibration values are stored in: ", full_path)



def get_image_shift(path):
    # returns image_shift values
    all_distortion = get_camera_distortion(path)
    return all_distortion['image_shift']


def get_camera_distortion(path):
    # returns camera_distortion values:
    # ret, mtx, dist, rvecs, tvecs
    filename = 'camera_distortion.yaml'
    full_path = os.path.join(path, filename)
    stream = open(full_path, 'r')
    data = yaml.load(stream, Loader)
    return data


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Camera calibration")
    parser.add_argument("camera_name", help="Name of the camera, which calibrate", type=str)
    parser.add_argument("-num_pic", help="Amount of photos to take during the calibration. The default "
                                                    "value is 20. More photos, better precision", nargs='?',
                        const=20 ,default=20, type=int)
    parser.add_argument("-path", help="Path where to store camera parameters. Default is code location.", nargs='?',
                        const=None, default=None, type=str)
    parser.add_argument('-demo', help="Demo mode will show images with opencv.", action='store_const', default=False,
                        const=True)
    args = parser.parse_args()

    camera_name = int(args.camera_name)
    path = args.path
    number_of_pictures = args.num_pic
    demo = args.demo


    if path is None:
        path = os.path.dirname(os.path.realpath(__file__))

    print("Camera name: ", camera_name)
    print("Path: ",  path)
    print("Number of pictures", number_of_pictures)
    print("Demo: ", demo)

    camera_distortion, image_shift = calibrate_camera(camera_name, number_of_pictures, demo)
    save_parameters(camera_distortion, image_shift, path)
