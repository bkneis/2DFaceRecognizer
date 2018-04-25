import cv2
import numpy as np
import PyCapture2

PG_CAMERA_TYPE = 'pt'
CV_CAMERA_TYPE = 'cv'


class CameraNotFoundException(Exception):
    pass


class Camera:

    def __init__(self, strategy, idx):
        self.strategy = strategy
        if strategy == PG_CAMERA_TYPE:
            self.cam = self._create_pt_camera(idx)
        else:
            self.cam = self._create_cv_camera(idx)

    def get_image(self):
        if self.strategy == PG_CAMERA_TYPE:
            return self._get_image_pg()
        else:
            return self._get_image_cv()

    def cleanup(self):
        if self.strategy == PG_CAMERA_TYPE:
            return self._cleanup_gt()
        else:
            return self._cleanup_cv()

    def _cleanup_gt(self):
        self.cam.stopCapture()

    def _cleanup_cv(self):
        self.cam.release()

    def _get_image_pg(self):
        try:
            image = self.cam.retrieveBuffer()
        except PyCapture2.Fc2error as fc2err:
            print('Error retrieving buffer', fc2err)
        c = image.getCols()
        r = image.getRows()
        return cv2.pyrDown(np.asarray(image.getData(), dtype=np.uint8).reshape(c, r))

    def _get_image_cv(self):
        ret, frame = self.cam.read()
        return frame

    def _create_pt_camera(self, idx):
        bus = PyCapture2.BusManager()
        num_cams = bus.getNumOfCameras()

        if num_cams < 1:
            raise CameraNotFoundException

        cam = PyCapture2.Camera()
        cam.connect(bus.getCameraFromIndex(idx))
        cam.startCapture()
        return cam

    def _create_cv_camera(self, idx):
        cam = cv2.VideoCapture(idx)

        if not cam.isOpened():
            raise CameraNotFoundException

        return cam
