import cv2
import numpy as np
from cv2 import aruco

# Load the calibration parameters
calibration_data = np.load("Matrix.py")
camera_matrix = calibration_data["camera_matrix"]
dist_coeffs = calibration_data["dist_coeffs"]

# Capture an image
image = cv2.imread("marker_image.jpg")
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Define the ArUco dictionary
aruco_dict = aruco.Dictionary_get(aruco.DICT_4X4_100)

# Detect markers
corners, ids, _ = aruco.detectMarkers(gray, aruco_dict)

if ids is not None:
    # Estimate pose
    rvecs, tvecs, _ = aruco.estimatePoseSingleMarkers(corners, 0.05, camera_matrix, dist_coeffs)

    # Get the distance from the camera to the marker
    distance = tvecs[0][0][2]

    print("Distance to marker:", distance)