import cv2
import os
import sys
import numpy


def get_immediate_subdirectories(a_dir):
    return [name for name in os.listdir(a_dir)
            if os.path.isdir(os.path.join(a_dir, name))]


def traverse(root_dir):
    train_dir = '/home/arthur/ps_train_v2'
    test_dir = '/home/arthur/ps_test_v2'
    subject_dirs = get_immediate_subdirectories(root_dir)

    # Each subject
    for idx, subject_dir in enumerate(subject_dirs):

        # Create the test subjects directory
        test_subject_path = os.path.join(test_dir, subject_dir)
        if not os.path.exists(test_subject_path):
            os.makedirs(test_subject_path)

        # Create directory for training images
        train_subject_path = os.path.join(train_dir, subject_dir)
        if not os.path.exists(train_subject_path):
            os.makedirs(train_subject_path)

        # Get the folder of the subject containing the sessions
        subject_path = os.path.join(root_dir, subject_dir)

        # Get all sessions for a subject
        session_dirs = get_immediate_subdirectories(subject_path)

        # Ignore any subjects with only on session
        num_sessions = len(session_dirs)
        if num_sessions < 2:
            print('less than 2', subject_dir)
            continue

        # split the directories 80 / 20
        eval_split = int(num_sessions * 0.8)
        train_dirs = session_dirs[:eval_split]
        test_dirs = session_dirs[eval_split:]

        # Create the 2D image by averaging the four light sources
        for session_id, sessionDir in enumerate(train_dirs):
            sub_path = os.path.join(subject_path, sessionDir)
            print('Subject : ', sub_path)
            img0 = cv2.imread(sub_path + '/im0_cropped.bmp')
            if img0 is not None:
                img0 = img0.astype(numpy.int64)
                img1 = cv2.imread(sub_path + '/im1_cropped.bmp').astype(numpy.int64)
                img2 = cv2.imread(sub_path + '/im2_cropped.bmp').astype(numpy.int64)
                img3 = cv2.imread(sub_path + '/im3_cropped.bmp').astype(numpy.int64)
                average = (img0 + img1 + img2 + img3) / 4

                cv2.imwrite(train_subject_path + '/average-%s.png' % session_id, average)

        # Synthesize the images to create illumination variance
        for session_id, sessionDir in enumerate(test_dirs):
            sub_path = os.path.join(subject_path, sessionDir)
            print('Subject : ', sub_path)
            img0 = cv2.imread(sub_path + '/im0_cropped.bmp')
            if img0 is not None:
                img0 = img0.astype(numpy.int64)
                img1 = cv2.imread(sub_path + '/im1_cropped.bmp').astype(numpy.int64)
                img2 = cv2.imread(sub_path + '/im2_cropped.bmp').astype(numpy.int64)
                img3 = cv2.imread(sub_path + '/im3_cropped.bmp').astype(numpy.int64)

                average = (img0 + img1 + img2 + img3) / 4
                left = (img0 + img1) / 2

                cv2.imwrite(test_subject_path + '/average-%s.png' % session_id, average)
                cv2.imwrite(test_subject_path + '/left-%s.png' % session_id, left)
                # cv2.imwrite(test_subject_path + '/bi-%s.png' % session_id, leftright)
                cv2.imwrite(test_subject_path + '/intense-%s.png' % session_id, img0)
            else:
                print('Cant find images for ', subject_path)


if __name__ == '__main__':
    traverse(sys.argv[1])
