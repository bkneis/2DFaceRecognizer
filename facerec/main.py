import cv2
import numpy as np

from LBP import LBP
from FaceDetector import FaceDetector
from classifiers import SVM, KNearest


def main():

    # Get subjects to train the svm on
    imgs = ['/home/arthur/Downloads/lfw/Gian_Marco/Gian_Marco_0001.jpg',
            '/home/arthur/Downloads/lfw/Micky_Ward/Micky_Ward_0001.jpg',
            '/home/arthur/Downloads/lfw/Ziwang_Xu/Ziwang_Xu_0001.jpg',
            '/home/arthur/Downloads/lfw/Zhu_Rongji/Zhu_Rongji_0001.jpg']

    # Get photos of subject to train to ensure the same class label is assigned to
    me = ['/home/arthur/face1.png'] # , '/home/arthur/face2.png', '/home/arthur/face3.png'

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
            face = detector.crop_face(image, face)
            hist, bins = lbp.run(face, False)
            hists.append(hist)
            labels.append(69)
        else:
            print('Warn no face detector')

    # Transform to np arrays
    samples = np.array(hists, dtype=np.float32)
    labels = np.array(labels, dtype=np.int)

    # Train svm
    svm.train(samples, labels)
    knn.train(samples, labels)

    # Test the svm
    test = cv2.imread('/home/arthur/pic2.jpg', 0)
    #test = cv2.imread('/home/arthur/Downloads/lfw/Georgina_Papin/Georgina_Papin_0001.jpg', 0)
    face = detector.detect(test)
    face = detector.crop_face(test, face)
    hist, bins = lbp.run(face, False)
    test_sample = np.array([hist], dtype=np.float32)

    # Predict with svm
    class_id = svm.predict(test_sample)
    print('SVM predicted class label ', class_id)

    # Predict with knn
    knn.predict(test_sample)


if __name__ == '__main__':
    main()
