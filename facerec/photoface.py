import sys
import os
import numpy as np
import math
import cv2
from plyfile import PlyData

from LBP import LBP, LocalBinaryPatterns
from classifiers import SVM, KNearest

# todo test the classifier
# todo validate on a 80/20 split

lbp = LBP()

svm = SVM()
knn = KNearest()

hists = []
labels = []


def main(photoface_dir):
    traverse(photoface_dir, describe_face)
    samples = np.array(hists, dtype=np.float32)
    ids = np.array(labels, dtype=np.int)

    # Train classifiers
    svm.train(samples, ids)
    knn.train(samples, ids)

    # test on a subject
    depth, gray = get_face('/media/arthur/124A-FB70/faceSample/2008-02-29_08-33-54/face_cropped.ply')
    hist, bins = lbp.run(depth)
    g_hist, g_bins = lbp.run(gray)
    test_sample = np.array([np.concatenate((hist, g_hist))], dtype=np.float32)
    dist, class_id = knn.predict(test_sample)
    svm_class_id = svm.predict(test_sample)
    print('Predicted class is : ', class_id)
    print('Predicted svm class is ', svm_class_id)
    print('Actual class is 0')
    print('================')

    # test on a subject
    depth, gray = get_face('/media/arthur/124A-FB70/faceSample/2008-02-21_08-19-45/face_cropped.ply')
    hist, bins = lbp.run(depth)
    g_hist, g_bins = lbp.run(gray)
    test_sample = np.array([np.concatenate((hist, g_hist))], dtype=np.float32)
    dist, class_id = knn.predict(test_sample)
    svm_class_id = svm.predict(test_sample)
    print('Predicted class is : ', class_id)
    print('Predicted svm class is ', svm_class_id)
    print('Actual class is 1')
    print('================')

    # test on a subject
    depth, gray = get_face('/media/arthur/124A-FB70/faceSample/2008-02-21_08-11-44/face_cropped.ply')
    hist, bins = lbp.run(depth)
    g_hist, g_bins = lbp.run(gray)
    test_sample = np.array([np.concatenate((hist, g_hist))], dtype=np.float32)
    dist, class_id = knn.predict(test_sample)
    svm_class_id = svm.predict(test_sample)
    print('Predicted class is : ', class_id)
    print('Predicted svm class is ', svm_class_id)
    print('Actual class is 2')
    print('================')


def traverse(root_dir, func):
    for iDir, subjectDirs, files in os.walk(root_dir):
        # Each subject
        for idx, subjectDir in enumerate(subjectDirs):
            subjectPath = os.path.join(root_dir, subjectDir)
            # Each session
            for jDir, sessionDirs, subjectFiles in os.walk(subjectPath):
                for sessionDir in sessionDirs:
                    sessionPath = os.path.join(subjectPath, sessionDir)
                    file_path = sessionPath + '/face_cropped.ply'
                    if os.path.isfile(file_path):
                        print('ID : ', idx)
                        print('Subject : ', subjectDir)
                        func(file_path, idx)


def describe_face(ply_path, idx):
    print('Describing face for class ', idx)
    print('Using file ', ply_path)
    depth, gray = get_face(ply_path)
    d_hist, d_bins = lbp.run(depth)
    g_hist, g_bins = lbp.run(gray)
    hists.append(np.concatenate((d_hist, g_hist)))
    labels.append(idx)


def get_face(ply_path):
    # Read the PLY file
    ply = PlyData.read(ply_path)
    # Extract the depth and RGB info
    vertex = ply['vertex']
    (z, r, g, b) = (vertex[t] for t in ('z', 'red', 'green', 'blue'))

    # Resize the 1D arrays into 2D
    size = int(math.sqrt(r.shape[0]))
    r = r.reshape(size, size)
    g = g.reshape(size, size)
    b = b.reshape(size, size)
    size = int(math.sqrt(z.shape[0]))
    # Stack the RGB values to create an image
    rgb = np.dstack((r, g, b))
    # Convert to gray scale
    gray = cv2.cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)

    # Return xyz (x,x) and gray scale info (x, x)
    return z.reshape(size, size), gray


if __name__ == '__main__':
    main(sys.argv[1])
