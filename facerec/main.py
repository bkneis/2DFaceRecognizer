import cv2
import numpy as np

from LBP import LBP
from FaceDetector import FaceDetector
from SVM import SVM


def main():

    imgs = ['/home/arthur/face.png',
            '/home/arthur/Downloads/lfw/Gian_Marco/Gian_Marco_0001.jpg',
            '/home/arthur/Downloads/lfw/Micky_Ward/Micky_Ward_0001.jpg',
            '/home/arthur/Downloads/lfw/Ziwang_Xu/Ziwang_Xu_0001.jpg',
            '/home/arthur/Downloads/lfw/Zhu_Rongji/Zhu_Rongji_0001.jpg']

    lbp = LBP()
    detector = FaceDetector()
    svm = SVM()

    hists = []
    labels = []

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

    samples = np.array(hists, dtype=np.float32)
    labels = np.array(labels, dtype=np.int)

    svm.train(samples, labels)

    # Test the svm
    test = cv2.imread('/home/arthur/face2.png', 0)
    face = detector.detect(test)
    face = detector.crop_face(test, face)
    hist, bins = lbp.run(face, False)
    test_sample = np.array([hist], dtype=np.float32)
    retval, results = svm.predict(test_sample)
    print(retval, results)


if __name__ == '__main__':
    main()