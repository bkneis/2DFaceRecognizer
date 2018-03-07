import numpy as np
import cv2
from skimage import feature


class LBP:

    def _encode(self, center, pixels):
        res = 0
        weights = [1, 2, 4, 8, 16, 32, 64, 128]
        for idx, a in enumerate(pixels):
            if a >= center:
                res += weights[idx]
        return res

    def _get_pixel(self, l, idx, idy, default=0):
        try:
            return l[idx, idy]
        except IndexError:
            return default

    def _get_neighbourhood(self, img, x, y):
        tl = self._get_pixel(img, x - 1, y - 1)
        tu = self._get_pixel(img, x, y - 1)
        tr = self._get_pixel(img, x + 1, y - 1)
        r = self._get_pixel(img, x + 1, y)
        l = self._get_pixel(img, x - 1, y)
        bl = self._get_pixel(img, x - 1, y + 1)
        br = self._get_pixel(img, x + 1, y + 1)
        bd = self._get_pixel(img, x, y + 1)
        return tl, tu, tr, r, l, bl, br, bd

    def run(self, img, debug=False):

        # Resize image to be same size as classified faces
        img = cv2.resize(img, (120, 120), interpolation=cv2.INTER_CUBIC)

        # Copy the image to transform
        transformed_img = img.copy()

        # Iterate over each pixel in image
        for x in range(0, len(img)):
            for y in range(0, len(img[0])):
                # Get the center pixel to encode
                center = img[x, y]
                # Get the neighbouring pixels
                neighbourhood = self._get_neighbourhood(img, x, y)
                # Apply the weighted LBP operator
                pattern = self._encode(center, neighbourhood)
                # Set the LBP value for the pixel
                transformed_img.itemset((x, y), pattern)

        # Investigate the image when debugging
        if debug:
            cv2.imwrite('debug/thresholded_image.png', transformed_img)

        # Return binned histogram of image
        return np.histogram(transformed_img.flatten(), 256, [0, 256])


class LocalBinaryPatterns:
    def __init__(self, numPoints, radius):
        # store the number of points and radius
        self.numPoints = numPoints
        self.radius = radius

    def run(self, image, eps=1e-7):
        # compute the Local Binary Pattern representation
        # of the image, and then use the LBP representation
        # to build the histogram of patterns
        lbp = feature.local_binary_pattern(image, self.numPoints,
                                           self.radius, method="default")
        (hist, _) = np.histogram(lbp.ravel(),
                                 bins=np.arange(0, self.numPoints + 3),
                                 range=(0, self.numPoints + 2))

        # normalize the histogram
        hist = hist.astype("float")
        hist /= (hist.sum() + eps)

        # return the histogram of Local Binary Patterns
        return hist, None