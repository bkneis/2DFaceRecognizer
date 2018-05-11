import os
import cv2

from exceptions import ImageNotProvidedException # todo remove .


class FaceDetector:
    """Implementation of the FaceDetector using opencv's cascade classifier """

    def __init__(self):
        super().__init__()
        data_path = os.path.join(os.path.dirname(__file__), 'assets/haarcascade_frontalface_default.xml')
        self.face_cascade = cv2.CascadeClassifier(data_path)

    def crop_face(self, img, face):
        if face is None:
            return None
        x, y, w, h = face
        return img[y:(y+h), x:(x+w)]

    def detect(self, img):
        if img is None:
            raise ImageNotProvidedException

        try:
            faces = self.face_cascade.detectMultiScale(img, 1.05, 5)
        except cv2.error as err:
            print('Detect multi scale failed', err)
            faces = []

        biggest = faces[0] if len(faces) else None
        width = 0

        for face in faces:
            x, y, w, h = face
            if w > width:
                width = w
                biggest = face

        return biggest
