import cv2
import sys
import numpy as np
import math
import subprocess
from FaceDetector import FaceDetector


ply_header = '''ply
format ascii 1.0
element vertex %(vert_num)d
property float x
property float y
property float z
property uchar red
property uchar green
property uchar blue
end_header
'''


def write_pcd(fn, verts, colors, disp, face_coords):

    x, y, w, h = face_coords

    rec_colors = np.delete(colors, [0, 1], 2)
    rec_verts = np.delete(verts, [0, 1], 2)
    rec_data = np.hstack([rec_colors, rec_verts])
    print('rec', rec_data.shape)
    cv2.imwrite('img.png', rec_colors)

    mask = disp > disp.min() + 15
    mask = mask[y:(y + h), x: (x + w)]
    verts = verts[mask]
    colors = colors[mask]

    verts = verts.reshape(-1, 3)
    colors = colors.reshape(-1, 3)
    verts = np.hstack([verts, colors])

    depth = np.hstack([verts])

    std = np.std(depth)

    with open('/tmp/%s.ply' % fn, 'wb') as f:
        f.write((ply_header % dict(vert_num=len(verts))).encode('utf-8'))
        np.savetxt(f, verts, fmt='%f %f %f %d %d %d ')

    return std, rec_data


def generate_encoding(leftpath, rightpath, detector, stereo, filename):

    imgL_ = cv2.imread(leftpath)
    imgR_ = cv2.imread(rightpath)

    face_coords_ = detector.detect(imgR_)
    x, y, w, h = face_coords_
    x -= 300
    y -= 300
    w += 600
    h += 600

    imgL = imgL_[y:(y + h), x: (x + w)]
    imgR = imgR_[y:(y + h), x: (x + w)]

    imgL = cv2.pyrDown(imgL)
    imgR = cv2.pyrDown(imgR)

    disp = stereo.compute(imgL, imgR).astype(np.float32) / 16.0

    h, w = imgL.shape[:2]
    f = 1600  # 0.8 * w  # guess for focal length
    Q = np.float32([[1, 0, 0, -0.5 * w],
                    [0, -1, 0, 0.5 * h],  # turn points 180 deg around x-axis,
                    [0, 0, 0, -f],  # so that y-axis looks up
                    [0, 0, 1, 0]])

    points = cv2.reprojectImageTo3D(disp, Q)
    colors = cv2.cvtColor(imgL, cv2.COLOR_BGR2RGB)

    gray = cv2.cvtColor(colors, cv2.COLOR_RGB2GRAY)

    face_coords = detector.detect(gray)
    x, y, w, h = face_coords

    out_points = points[y:(y + h), x: (x + w), :]
    out_colors = colors[y:(y + h), x: (x + w), :]

    return write_pcd(filename, out_points, out_colors, disp, face_coords)


def main(leftpath, rightpath, filename):

    detector = FaceDetector()

    imgL_ = cv2.imread(leftpath)
    imgR_ = cv2.imread(rightpath)

    face_coords_ = detector.detect(imgR_)
    x, y, w, h = face_coords_
    x -= 300
    y -= 300
    w += 600
    h += 600

    imgL = imgL_[y:(y + h), x: (x + w)]
    imgR = imgR_[y:(y + h), x: (x + w)]

    imgL = cv2.pyrDown(imgL)
    imgR = cv2.pyrDown(imgR)

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
    f = 1600  # 0.8 * w  # guess for focal length
    Q = np.float32([[1, 0, 0, -0.5 * w],
                    [0, -1, 0, 0.5 * h],  # turn points 180 deg around x-axis,
                    [0, 0, 0, -f],  # so that y-axis looks up
                    [0, 0, 1, 0]])

    points = cv2.reprojectImageTo3D(disp, Q)
    colors = cv2.cvtColor(imgL, cv2.COLOR_BGR2RGB)

    gray = cv2.cvtColor(colors, cv2.COLOR_RGB2GRAY)

    cv2.imwrite('test.png', imgL)
    cv2.imwrite('test2.png', imgR)

    face_coords = detector.detect(gray)
    x, y, w, h = face_coords

    out_points = points[y:(y + h), x: (x + w), :]
    out_colors = colors[y:(y + h), x: (x + w), :]

    std = write_pcd(filename, out_points, out_colors, disp, face_coords)

    if std < 50:
        color = (0, 0, 255)
        label = "Photo"
    else:
        color = (255, 0, 0)
        label = "Detected"

    x, y, w, h = face_coords_
    cv2.rectangle(imgR_, (x, y + h), (x + w, y), color, 6)
    cv2.putText(imgR_, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 0), 1, cv2.LINE_AA)

    cv2.imwrite('output.png', imgR_)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], sys.argv[3])