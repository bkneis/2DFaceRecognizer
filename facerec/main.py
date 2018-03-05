import cv2
import time
import argparse
import numpy as np

import util
from LBP import LBP
from FaceDetector import FaceDetector
from classifiers import SVM, KNearest


def load_subjects(subjects, detector, lbp):
    # Loop over each subject and perform LBP operator and add to histogram and labels
    hists = []
    labels = []
    for idx, img in enumerate(subjects):
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

    image = cv2.imread('/home/arthur/me.jpg', 0)
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

    return hists, labels


def classify_snapshot(img, detector, lbp, classifier):
    face = detector.detect(img)
    face = detector.crop_face(img, face)
    hist, bins = lbp.run(face, False)
    test_sample = np.array([hist], dtype=np.float32)
    dist, class_id = classifier.predict(test_sample)
    print('Subject was ', class_id)


def main(args):

    # Create algorithm objects
    lbp = LBP()
    detector = FaceDetector()
    svm = SVM()
    knn = KNearest()

    # Get subjects to train the svm on
    imgs = ['/home/arthur/Downloads/lfw_funneled/Gian_Marco/Gian_Marco_0001.jpg',
            '/home/arthur/Downloads/lfw_funneled/Micky_Ward/Micky_Ward_0001.jpg',
            '/home/arthur/Downloads/lfw_funneled/Ziwang_Xu/Ziwang_Xu_0001.jpg',
            '/home/arthur/Downloads/lfw_funneled/Zhu_Rongji/Zhu_Rongji_0001.jpg']

    # Load the subjects and extract their features
    hists, labels = load_subjects(imgs, detector, lbp)

    # Transform to np arrays
    samples = np.array(hists, dtype=np.float32)
    labels = np.array(labels, dtype=np.int)

    # Train classifiers
    svm.train(samples, labels)
    knn.train(samples, labels)

    # Check which mode the app is running in (image vs. video)
    if args.image is not None:
        # Read the image from the file path provided
        img = cv2.imread(args.image, 0)
        # Check the image exists
        if img is not None:
            # Run face recognition algorithm
            classify_snapshot(img, detector, lbp, knn)
        else:
            print('The image could not be found...')
        return

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
    parser = argparse.ArgumentParser(description='2d Face Recognition with Local Binary Patterns')
    parser.add_argument('--image', nargs='?', const=1, type=str)
    _args = parser.parse_args()
    main(_args)
