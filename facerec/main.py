import cv2
import time
import argparse
import numpy as np
import os
import PyCapture2

import util
from stereoMatch import reconstruct
from LBP import LBP
from FaceDetector import FaceDetector
from classifiers import SVM, KNearest

from Camera import Camera, PG_CAMERA_TYPE, CV_CAMERA_TYPE


flatten = lambda l: [item for sublist in l for item in sublist]


def load_subjects(data_folder_path, detector):
    # ------STEP-1--------
    # get the directories (one directory for each subject) in data folder
    dirs = os.listdir(data_folder_path)

    # list to hold all subject faces
    faces = []
    # list to hold labels for all subjects
    labels = []

    # let's go through each directory and read images within it
    for dname in dirs:

        print('Subject ', dname)

        sdirs = os.listdir(os.path.join(data_folder_path, dname))

        for dir_name in sdirs:
            print('Session ', dir_name)
            # ------STEP-2--------
            # extract label number of subject from dir_name
            # format of dir name = slabel
            # , so removing letter 's' from dir_name will give us label
            label = int(dname)

            # build path of directory containing images for current subject subject
            # sample subject_dir_path = "training-data/s1"
            subject_dir_path = data_folder_path + "/" + dname + "/" + dir_name

            image_name = 'image.png'

            # build image path
            # sample image path = training-data/s1/1.pgm
            image_path = subject_dir_path + "/" + image_name

            # read image
            image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

            # detect face
            face_coords = detector.detect(image)
            face = detector.crop_face(image, face_coords)

            # ------STEP-4--------
            # for the purpose of this tutorial
            # we will ignore faces that are not detected
            if face is not None:
                # add face to list of faces
                faces.append(face)
                # add label for this face
                labels.append(label)

    return faces, labels


def classify_snapshot(img, detector, lbp, classifier):
    face = detector.detect(img)
    face = detector.crop_face(img, face)
    hist, bins = lbp.run(face, False)
    test_sample = np.array([hist], dtype=np.float32)
    dist, class_id = classifier.predict(test_sample)
    print('Subject was ', class_id)


def main(args):

    # Create algorithm objects
    # lbp = LBP()
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
    hists, labels = load_subjects('/home/arthur/test', detector)

    hists, labels = lbp.run(hists, labels)

    # Transform to np arrays
    samples = flatten(hists)
    responses = flatten(labels)
    samples = np.array(samples, dtype=np.float32)
    labels = np.array(responses, dtype=np.int)

    print('hists', samples[0].shape)
    print('labels', responses)

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
    l_cam = Camera(CV_CAMERA_TYPE, 0)

    # Establish connection to second camera
    # r_cam = Camera(PG_CAMERA_TYPE, 1)

    # Continuously grab the next frame from the camera
    while True:
        # Capture frame-by-frame
        start = time.time()
        frame = l_cam.get_image()
        end = time.time()
        print('Getting pic', end - start)

        # Start timer for performance logging
        start = time.time()

        # Convert frame to gray scale for face detector
        # try:
        #     gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # except cv2.error:
        #     gray = frame
        gray = frame

        start = time.time()
        # Detect a face in the frame and crop the image
        face_coords = detector.detect(gray)
        face = detector.crop_face(gray, face_coords)
        end = time.time()
        print('detect face', end - start)

        # Check we have detected a face
        if face is not None:
            # Take a photo with the right camera
            # rframe = r_cam.get_image()

            # Reconstruct 3D face from 2D images
            # reconstruct(frame, rframe)
            start = time.time()
            # Apply LBP operator to get feature descriptor
            hist, bins = lbp.run([face], [4], False)
            end = time.time()
            print('feature des', end - start)

            # Convert the LBP descriptor to numpy array for opencv classifiers
            test_sample = np.array([hist], dtype=np.float32)

            # Get the class of id of the closest neighbour and its distance
            # class_id = svm.predict(test_sample)
            # print('SVM class id', class_id)
            start = time.time()
            dist, class_id = knn.predict(hist) # test_sample
            end = time.time()
            print('Classifying', end - start)

            start = time.time()
            # Draw the face if found
            util.draw_face(dist, class_id, frame, face_coords)
            end = time.time()
            print('drawing', end - start)

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
    l_cam.cleanup()
    # r_cam.cleanup()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='2d Face Recognition with Local Binary Patterns')
    parser.add_argument('--image', nargs='?', const=1, type=str)
    parser.add_argument('--checkLive', nargs='?', const=1, type=bool)
    _args = parser.parse_args()
    if _args.image is not None and _args.checkLive:
        print('WARNING: Face liviness can only be checked given 3D information')
    main(_args)
