from FaceDetector import FaceDetector
import cv2

detector = FaceDetector()

img = cv2.imread('/home/arthur/stereo_demo/celebs/Alanna_Ubach.png', cv2.IMREAD_GRAYSCALE)

face_coords = detector.detect(img)
face = detector.crop_face(img, face_coords)

cv2.imwrite('/home/arthur/stereo_demo/celebs/Alanna_Ubach_.png', face)

img = cv2.imread('/home/arthur/stereo_demo/celebs/Alfonso_Soriano.png', cv2.IMREAD_GRAYSCALE)

face_coords = detector.detect(img)
face = detector.crop_face(img, face_coords)

cv2.imwrite('/home/arthur/stereo_demo/celebs/Alfonso_Soriano_.png', face)

img = cv2.imread('/home/arthur/stereo_demo/celebs/Alexandra_Jackson.png', cv2.IMREAD_GRAYSCALE)

face_coords = detector.detect(img)
face = detector.crop_face(img, face_coords)

cv2.imwrite('/home/arthur/stereo_demo/celebs/Alexandra_Jackson_.png', face)
