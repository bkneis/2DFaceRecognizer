import cv2
import numpy as np
import time

from LBP import LBP
from FaceDetector import FaceDetector
from classifiers import SVM, KNearest


# def timeit(method):
#
#     def timed(*args, **kw):
#         ts = time.time()
#         result = method(*args, **kw)
#         te = time.time()
#
#         print '%r (%r, %r) %2.2f sec' % \
#               (method.__name__, args, kw, te-ts)
#         return result
#
#     return timed


def main():

    # Get subjects to train the svm on
    imgs = ['/home/arthur/Downloads/lfw_funneled/Gian_Marco/Gian_Marco_0001.jpg',
            '/home/arthur/Downloads/lfw_funneled/Micky_Ward/Micky_Ward_0001.jpg',
            '/home/arthur/Downloads/lfw_funneled/Ziwang_Xu/Ziwang_Xu_0001.jpg',
            '/home/arthur/Downloads/lfw_funneled/Zhu_Rongji/Zhu_Rongji_0001.jpg']

    # Get photos of subject to train to ensure the same class label is assigned to
    me = ['/home/arthur/me.jpg']  # , '/home/arthur/face2.png', '/home/arthur/face3.png'

    # Create algorithm objects
    lbp = LBP()
    detector = FaceDetector()
    svm = SVM()
    knn = KNearest()

    # Array to store resulting LBP histograms
    hists = []
    labels = []

    # Loop over each subject and perform LBP operator and add to histogram and labels
    for idx, img in enumerate(imgs):
        image = cv2.imread(img, 0)
        face = detector.detect(image)
        if face is not None:
            face = detector.crop_face(image, face)
            hist, bins = lbp.run(face, False)
            hists.append(hist)
            labels.append(idx)
            print('Id %s person %s' % (idx, img))
        else:
            print('Warn no face detector')

    # Do the same as above for target with fixed class label (69)
    for idx, img in enumerate(me):
        image = cv2.imread(img, 0)
        face = detector.detect(image)
        if face is not None:
            print('Adding myself to models')
            face = detector.crop_face(image, face)
            hist, bins = lbp.run(face, False)
            hists.append(hist)
            labels.append(69)
        else:
            print('Warn no face detector')

    # Transform to np arrays
    samples = np.array(hists, dtype=np.float32)
    labels = np.array(labels, dtype=np.int)

    print(samples)
    print(labels)

    # Train classifiers
    svm.train(samples, labels)
    knn.train(samples, labels)

    # Test the svm
    test = cv2.imread('/home/arthur/image.png', 0)
    ts = time.time()
    face = detector.detect(test)
    face = detector.crop_face(test, face)
    hist, bins = lbp.run(face, False)
    test_sample = np.array([hist], dtype=np.float32)

    # Predict with svm
    class_id = svm.predict(test_sample)
    te = time.time()
    # print('prediction took %2.2f sec', te - ts)
    print('SVM predicted class label ', class_id)

    # Predict with knn
    knn.predict(test_sample)

    cap = cv2.VideoCapture(0)
    while (True):
        # Capture frame-by-frame
        ret, frame = cap.read()
        # Our operations on the frame come here
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        face_coords = detector.detect(gray)
        face = detector.crop_face(gray, face_coords)
        if face is not None:
            x, y, w, h = face_coords
            # cv2.imwrite('debug/face.png', face)
            hist, bins = lbp.run(face, False)
            test_sample = np.array([hist], dtype=np.float32)
            # class_id = svm.predict(test_sample)
            # print('SVM predicted class label ', class_id)
            dist, class_id = knn.predict(test_sample)
            color = (0, 0, 255)
            if class_id == 69 and dist < 3000000:
                color = (0, 255, 0)

            cv2.rectangle(frame, (x, y + h), (x + w, y), color, 3)

        # Display the resulting frame
        cv2.imshow('frame', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # When everything done, release the capture
    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
