#!/usr/bin/env python
import numpy as np
import cv2
import glob
import sys

# termination criteria
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 50, 0.05)
# Grid pattern dimensions. Only inner intersection points count
grid_size = (9, 7)

# Size in units (mm) of each square
square_size = 24.5

# Number of captures to use..
num_captures = 26

features_per_chessboard = grid_size[0]*grid_size[1]
# prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
objp = np.zeros((features_per_chessboard, 3), np.float32)
objp[:,:2] = square_size * (np.mgrid[0:grid_size[0],0:grid_size[1]].T.reshape(-1,2))

# Arrays to store object points and image points from all the images.
objpoints = [] # 3d point in real world space
imgpoints = [] # 2d points in image plane.

def draw(img, corners, imgpts):
    imgpts = np.int32(imgpts).reshape(-1,2)

    # draw ground floor in green
    cv2.drawContours(img, [imgpts[:4]],-1,(0,255,0),-3)

    # draw pillars in blue color
    for i,j in zip(range(4),range(4,8)):
        cv2.line(img, tuple(imgpts[i]), tuple(imgpts[j]),(255),3)

    # draw top layer in red color
    cv2.drawContours(img, [imgpts[4:]],-1,(0,0,255),3)

    return img

device = 0
if len(sys.argv) < 2:
    print ("Usage: {} <device_no> or /path/to/image_%4d.png or video.mp4".format(sys.argv[0]))
    
if len(sys.argv) >= 2:
    device = sys.argv[1]
    try:
        device = int(device)
    except:
        pass

cap = cv2.VideoCapture(device)
#cap.set(cv2.CAP_PROP_FRAME_WIDTH, 800)
#cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 600)
frame_shape = None
last_pos = None
countDown = 0

while len(imgpoints) < num_captures:
    # Capture frame-by-frame
    ret, frame = cap.read()

    if not ret:
        print("Empty frame at ", len(imgpoints) )
        break

    # Our operations on the frame come here
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    frame_shape = gray.shape[::-1]

    # Find the chess board corners
    ret, corners = cv2.findChessboardCorners(gray, grid_size, None)
    countDown -= 1
    
    # If found, add object points, image points (after refining them)
    if ret == True:
        should_accept = not last_pos is None and np.linalg.norm(corners[0] - last_pos) < 1.0 and countDown <= 0
        # should_accept = True
        if should_accept:
            # Don't accept image for 50 frames
            countDown = 50
            cv2.cornerSubPix(gray,corners,(5,5),(-1,-1),criteria)

            objpoints.append(objp)
            imgpoints.append(corners)
            print('frame added')
        
        last_pos = corners[0]

        # Draw and display the corners
        cv2.drawChessboardCorners(frame, grid_size, corners,ret)

    # Display the resulting frame
    cv2.imshow('img', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

dist = np.zeros((5, 1), np.float64)
mtx = np.eye(3, dtype=np.float64)
flags = 0
calib_criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 60, sys.float_info.epsilon)
ret, cameraMatrix, distCoeffs, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, frame_shape, mtx, dist, flags = flags, criteria=calib_criteria)

c_order = lambda a: np.nditer(a.copy(order='C'))
print ("Reprojection error:", ret)
print ("Camera matrix:", ", ".join("{:0.06f}".format(float(i)) for i in c_order(mtx) ) )
print ("Distortion:", ", ".join("{:0.06f}".format(float(i)) for i in c_order(dist) ) )
del rvecs
del tvecs

axis = square_size * np.float32([[0,0,0], [0,3,0], [3,3,0], [3,0,0],
                   [0,0,-3],[0,3,-3],[3,3,-3],[3,0,-3] ])

# Now loop forever just detecting the pose of the chessboard
while True:
    # Capture frame-by-frame
    ret, frame = cap.read()

    if not ret:
        print("Empty frame..")
        break

    # Our operations on the frame come here
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    ret, corners = cv2.findChessboardCorners(gray, grid_size, None)

    if ret == True:
        cv2.cornerSubPix(gray,corners,(5,5),(-1,-1),criteria)

        # Find the rotation and translation vectors.
        rvecs, tvecs, inliers = cv2.solvePnPRansac(objp, corners, mtx, dist)

        if len(inliers) > 4:
            # project 3D points to image plane
            imgpts, jac = cv2.projectPoints(axis, rvecs, tvecs, mtx, dist)

            frame = draw(frame, corners, imgpts)

    # Display the resulting frame
    cv2.imshow('img', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()
