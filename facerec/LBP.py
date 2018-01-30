import numpy as np
import cv2


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
        return np.histogram(img.flatten(), 256, [0, 256])
