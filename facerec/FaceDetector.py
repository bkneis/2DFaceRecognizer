import os
import cv2

from exceptions import ImageNotProvidedException # todo remove .


class FaceDetector:
    """Implementation of the FaceDetector using opencv's cascade classifier """

    def __init__(self):
        super().__init__()
        data_path = os.path.join(os.path.dirname(__file__), 'assets/haarcascade_frontalface_default.xml')
        # todo need to throw cv2.error and allow detectionTask to catch and fail if so
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
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        except cv2.error:
            # Cannot convert the image to grayscale, the image probably doesnt have 3 channels or is already gray
            gray = img

        try:
            faces = self.face_cascade.detectMultiScale(gray, 1.3, 5)  # todo add min and max size as filter
        except cv2.error as err:
            print('Detect multi scale failed', err)
            faces = []

        return faces[0] if len(faces) else None
