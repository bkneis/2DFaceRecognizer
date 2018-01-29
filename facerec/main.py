import cv2
import time
import argparse
import numpy as np

import util
from LBP import LBP
from FaceDetector import FaceDetector
from classifiers import SVM, KNearest


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
            face = cv2.resize(face, (120, 120), interpolation=cv2.INTER_CUBIC)
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
            face = cv2.resize(face, (120, 120), interpolation=cv2.INTER_CUBIC)
            hist, bins = lbp.run(face, False)
            hists.append(hist)
            labels.append(69)
        else:
            print('Warn no face detector')

    # Transform to np arrays
    samples = np.array(hists, dtype=np.float32)
    labels = np.array(labels, dtype=np.int)

    # Train classifiers
    svm.train(samples, labels)
    knn.train(samples, labels)

    # # Test the svm
    # test = cv2.imread('/home/arthur/image.png', 0)
    # face = detector.detect(test)
    # face = detector.crop_face(test, face)
    # hist, bins = lbp.run(face, False)
    # test_sample = np.array([hist], dtype=np.float32)
    #
    # # Predict with svm
    # class_id = svm.predict(test_sample)
    # te = time.time()
    # # print('prediction took %2.2f sec', te - ts)
    # print('SVM predicted class label ', class_id)
    #
    # # Predict with knn
    # knn.predict(test_sample)

    # Establish connection to camera
    cap = cv2.VideoCapture(0)

    # Continuously grab the next frame from the camera
    while cap.isOpened():
        # Capture frame-by-frame
        ret, frame = cap.read()

        # Start timer for performance logging
        start = time.time()

        # Convert frame to gray scale for face detector
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Detect a face in the frame and crop the image
        face_coords = detector.detect(gray)
        face = detector.crop_face(gray, face_coords)

        # Check we have detected a face
        if face is not None:
            # Apply LBP operator to get feature descriptor
            hist, bins = lbp.run(face, False)

            # Convert the LBP descriptor to numpy array for opencv classifiers
            test_sample = np.array([hist], dtype=np.float32)

            # Get the class of id of the closest neighbour and its distance
            dist, class_id = knn.predict(test_sample)

            # Draw the face if found
            util.draw_face(dist, class_id, frame, face_coords)
            # util.segment_face(frame)

        # Processing finished
        end = time.time()

        # Write the fps to the video
        util.write_fps(start, end, frame)

        # Display the resulting frame
        cv2.imshow('frame', frame)

        # Check if we should stop the application
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # When everything done, release the capture
    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
