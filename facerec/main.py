import cv2

from LBP import LBP
from FaceDetector import FaceDetector


def main():
    img = cv2.imread('/home/arthur/face.png')

    lbp = LBP()
    detector = FaceDetector()

    face = detector.detect(img)
    if face is not None:
        face = detector.crop_face(img, face)
        cv2.imwrite('/home/arthur/test.png', face)
        # hist, bins = lbp.run(face, True)
        # print(hist, bins)


if __name__ == '__main__':
    main()
