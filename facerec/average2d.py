import cv2
import os
import sys
import numpy


def traverse(root_dir):
    for iDir, subjectDirs, files in os.walk(root_dir):
        # Each subject
        for idx, subjectDir in enumerate(subjectDirs):
            subject_path = os.path.join(root_dir, subjectDir)
            # Each session
            for jDir, sessionDirs, subjectFiles in os.walk(subject_path):
                print('dirs', sessionDirs)
                for sessionDir in sessionDirs:
                    sub_path = os.path.join(subject_path, sessionDir)
                    print('Subject : ', sub_path)
                    img0 = cv2.imread(sub_path + '/im0.bmp')
                    if img0 is not None:
                        img0 = img0.astype(numpy.int64)
                        img1 = cv2.imread(sub_path + '/im1.bmp').astype(numpy.int64)
                        img2 = cv2.imread(sub_path + '/im2.bmp').astype(numpy.int64)
                        img3 = cv2.imread(sub_path + '/im3.bmp').astype(numpy.int64)
                        average = (img0 + img1 + img2 + img3) / 4
                        cv2.imwrite(sub_path + '/average.png', average)


if __name__ == '__main__':
    traverse(sys.argv[1])
