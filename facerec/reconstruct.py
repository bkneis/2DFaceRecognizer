import cv2
import sys
import numpy as np

from FaceDetector import FaceDetector
from LBP import LBP


def main(leftpath, rightpath):

    imgL = cv2.imread(leftpath)
    imgR = cv2.imread(rightpath)

    imgL = cv2.pyrDown(imgL)
    imgR = cv2.pyrDown(imgR)

    detector = FaceDetector()
    lbp = LBP()

    face_coords = detector.detect(imgL)
    x, y, w, h = face_coords
    x -= 400
    y -= 400
    w += 800
    h += 800

    # imgL = detector.crop_face(imgL, (x, y, w, h))
    # imgR = detector.crop_face(imgR, (x, y, w, h))

    # cv2.imwrite('left.png', imgL)
    # cv2.imwrite('right.png', imgL)

    window_size = 3
    min_disp = 16
    num_disp = 112 - min_disp

    stereo = cv2.StereoSGBM_create(minDisparity=min_disp,
                                   numDisparities=num_disp,
                                   blockSize=5,
                                   P1=8 * 3 * window_size ** 2,
                                   P2=32 * 3 * window_size ** 2,
                                   disp12MaxDiff=1,
                                   uniquenessRatio=10,
                                   speckleWindowSize=100,
                                   speckleRange=32
                                   )

    disp = stereo.compute(imgL, imgR).astype(np.float32) / 16.0

    h, w = imgL.shape[:2]
    f = 0.8 * w  # guess for focal length
    Q = np.float32([[1, 0, 0, -0.5 * w],
                    [0, -1, 0, 0.5 * h],  # turn points 180 deg around x-axis,
                    [0, 0, 0, -f],  # so that y-axis looks up
                    [0, 0, 1, 0]])

    points = cv2.reprojectImageTo3D(disp, Q)

    out_points = points  # [mask]
    out_points = out_points[:, :, 2]
    print(out_points)
    hist, bins = lbp.run(out_points, False)
    print('hist', hist)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
