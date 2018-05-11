import cv2
import sys
import numpy as np
import math
import subprocess
import pickle
from FaceDetector import FaceDetector
from sklearn.externals import joblib
from sklearn.model_selection import train_test_split
from sklearn.multiclass import OneVsRestClassifier
from sklearn.preprocessing import label_binarize
from sklearn.svm import SVC

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


# todo generate decent 3D models
# workout enrol when increasing distance
# get 2D working

def write_pcd(fn, verts, colors, disp, face_coords, write=False, mutate=False):

    x, y, w, h = face_coords

    rec_colors = np.delete(colors, [0, 1], 2)
    rec_verts = np.delete(verts, [0, 1], 2)
    rec_data = np.hstack([rec_colors, rec_verts])

    if mutate:
        # verts = np.random.normal(verts, 0.01)
        celeb = cv2.imread(mutate)
        colors = cv2.resize(celeb, (colors.shape[0], colors.shape[1]), interpolation=cv2.INTER_CUBIC)

    mask = disp > disp.min() + 40
    mask = mask[y:(y + h), x: (x + w)]
    verts = verts[mask]
    colors = colors[mask]

    verts = verts.reshape(-1, 3)
    colors = colors.reshape(-1, 3)
    verts = np.hstack([verts, colors])

    depth = np.hstack([verts])

    std = np.std(depth)

    if write:
        with open('/tmp/%s.ply' % fn, 'wb') as f:
            f.write((ply_header % dict(vert_num=len(verts))).encode('utf-8'))
            np.savetxt(f, verts, fmt='%f %f %f %d %d %d ')

    return std, rec_data


def generate_encoding(leftpath, rightpath, detector, stereo, filename, mutate):

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

    std, rec_data = write_pcd(filename, out_points, out_colors, disp, face_coords, write=True, mutate=mutate)
    return std, rec_data, face_coords_


def flatten(l):
    return [item for sublist in l for item in sublist]


def generate_predictions(faces, labels, face):

    face_recognizer = cv2.face.createLBPHFaceRecognizer()
    face_recognizer.train([face], np.array([88], dtype=int))

    samples = flatten(face_recognizer.getHistograms())
    samples = np.array(samples, dtype=np.float32)

    try:
        clf = joblib.load('model.pkl')
    except Exception:
        print('Failed to load model, training new one...')
        face_recognizer = cv2.face.createLBPHFaceRecognizer()
        face_recognizer.train(faces, np.array(labels, dtype=int))

        clf = SVC(gamma=0.5, probability=True)

        samples = flatten(face_recognizer.getHistograms())
        responses = flatten(face_recognizer.getLabels())

        samples = np.array(samples, dtype=np.float32)
        responses = np.sort(np.array(responses, dtype=np.int))

        X_train, X_test, y_train, y_test = train_test_split(samples, responses, test_size=0,
                                                            random_state=0)
        clf.fit(X_train, y_train)
        joblib.dump(clf, 'model.pkl')

    y_score = clf.decision_function(samples)
    y_score = y_score[0]
    # min = np.amin(y_score)
    # normalized_score = (y_score - min / (7 - min))
    # print('norms', normalized_score)

    return y_score


def enrol(faces, labels, mutate_subjects):
    face_recognizer = cv2.face.createLBPHFaceRecognizer()
    face_recognizer.train(faces, np.array(labels, dtype=int))

    clf = SVC(gamma=0.5, probability=True)

    samples = flatten(face_recognizer.getHistograms())
    responses = flatten(face_recognizer.getLabels())

    samples = np.array(samples, dtype=np.float32)
    responses = np.sort(np.array(responses, dtype=np.int))

    X_train, X_test, y_train, y_test = train_test_split(samples, responses, test_size=0,
                                                        random_state=0)
    clf.fit(X_train, y_train)
    joblib.dump(clf, 'model.pkl')

    with open('subjects', 'wb') as fp:
        pickle.dump(mutate_subjects, fp)


def main(leftpath_, rightpath_, filename):

    detector = FaceDetector()

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

    if not leftpath_ == 'enrol':
        photo = False
        std, t_rec_data, face_coords_ = generate_encoding(leftpath_, rightpath_, detector, stereo, filename, mutate=False)
        if std < 50:
            photo = True
    else:
        t_rec_data = None

    subjects = ['Bryan_Kneis']
    faces = []
    labels = []

    for idx, subject in enumerate(subjects):
        leftpath = '/home/arthur/stereo_demo/%s/left.png' % subject
        rightpath = '/home/arthur/stereo_demo/%s/right.png' % subject
        std, rec_data, face_coords = generate_encoding(leftpath, rightpath, detector, stereo, 'ignore', mutate=False)
        faces.append(rec_data)
        labels.append(idx)

    mutate_subjects = ['Alfonso_Soriano', 'Amanda_Beard', 'Adam_Kennedy', 'Alanna_Ubach']

    # with open('subjects', 'rb') as fp:
    #     mutate_subjects = pickle.load(fp)
    #     if leftpath_ == 'enrol':
    #         mutate_subjects.append(rightpath_)

    for idx, subject in enumerate(mutate_subjects):
        leftpath = '/home/arthur/stereo_demo/mannequin/right.png'
        rightpath = '/home/arthur/stereo_demo/mannequin/left.png'
        subject_path = '/home/arthur/stereo_demo/celebs/%s.png' % subject
        std, rec_data, face_coords = generate_encoding(leftpath, rightpath, detector, stereo, subject, mutate=subject_path)
        faces.append(rec_data)
        labels.append(len(subjects) + idx)

    if leftpath_ == 'enrol':
        enrol(faces, labels, mutate_subjects)
        print('%s has now been enrolled' % rightpath_)
        exit()

    y_score = generate_predictions(faces, labels, t_rec_data)

    detected = False
    for idx, score in enumerate(y_score):
        if idx == 0:
            print('Distance from the hyperplane for Bryan Kneis is ', score)
        else:
            print('Distance from the hyperplane for %s is ' % mutate_subjects[idx - 1], score)
        if score > 5.2:
            detected = idx

    if photo:
        color = (0, 0, 255)
        label = "Photo"
    else:
        color = (255, 0, 0)
        label = "Detected"

    if detected is not False:
        color = (0, 255, 0)
        label = subjects[detected]

    x, y, w, h = face_coords_
    imgR = cv2.imread(rightpath_)
    cv2.rectangle(imgR, (x, y + h), (x + w, y), color, 6)
    cv2.putText(imgR, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 1, cv2.LINE_AA)

    cv2.imwrite('output.png', imgR)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], sys.argv[3])
