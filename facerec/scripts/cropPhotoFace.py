import cv2
import os
import sys
from FaceDetector import FaceDetector

from exceptions import ImageNotProvidedException


def main(rootDir):
    detector = FaceDetector()
    for iDir, subjectDirs, files in os.walk(rootDir):
        for subjectDir in subjectDirs:
            print('Walking subject folder at %s' % subjectDir)
            subjectPath = os.path.join(rootDir, subjectDir)
            # Each session
            for jDir, sessionDirs, subjectFiles in os.walk(subjectPath):
                for sessionDir in sessionDirs:
                    print('Walking sessions folder at %s' % sessionDir)
                    sessionPath = os.path.join(subjectPath, sessionDir)
                    check = cv2.imread(sessionPath + '/im0_cropped.bmp')
                    if check is not None:
                        continue
                    img = cv2.imread(sessionPath + '/average.png', 0)
                    try:
                        face_coords = detector.detect(img)
                    except ImageNotProvidedException:
                        print('Image not available ', sessionDir)
                        continue
                    for i in range(4):
                        imgFilePath = sessionPath + '/im%s.bmp' % i
                        imgCroppedFilePath = sessionPath + '/im%s_cropped.bmp' % i
                        if os.path.isfile(imgFilePath):
                            print('Cropping the image %s' % imgFilePath)
                            img = cv2.imread(imgFilePath)
                            if face_coords is None:
                                print('Did not find a face, ignoring imgage')
                                # cv2.imwrite(imgCroppedFilePath, img)
                            else:
                                face = detector.crop_face(img, face_coords)
                                cv2.imwrite(imgCroppedFilePath, face)


if __name__ == '__main__':
    print('Traversing the photoface db at %s' % sys.argv[1])
    main(sys.argv[1])
