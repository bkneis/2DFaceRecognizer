import cv2
import sys
import numpy as np
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


def write_pcd(fn, verts, colors):
    verts = verts.reshape(-1, 3)
    colors = colors.reshape(-1, 3)
    verts = np.hstack([verts, colors])

    with open('/tmp/%s.ply' % fn, 'wb') as f:
        f.write((ply_header % dict(vert_num=len(verts))).encode('utf-8'))
        np.savetxt(f, verts, fmt='%f %f %f %d %d %d ')


def main(leftpath, rightpath, filename):

    detector = FaceDetector()

    imgL = cv2.imread(leftpath)
    imgR = cv2.imread(rightpath)

    face_coords = detector.detect(imgR)
    x, y, w, h = face_coords
    x -= 300
    y -= 300
    w += 600
    h += 600

    imgL = imgL[y:(y + h), x: (x + w)]
    imgR = imgR[y:(y + h), x: (x + w)]

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

    cv2.imwrite('test.png', gray)

    face_coords = detector.detect(gray)
    x, y, w, h = face_coords

    out_points = points[y:(y + h), x: (x + w), :]
    out_colors = colors[y:(y + h), x: (x + w), :]

    print(out_points.shape)

    mask = disp > disp.min() + 15
    print(mask.shape)
    mask = mask[y:(y + h), x: (x + w)]
    out_points = out_points[mask]
    out_colors = out_colors[mask]
    # out_points = points
    # out_colors = colors

    write_pcd(filename, out_points, out_colors)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], sys.argv[3])