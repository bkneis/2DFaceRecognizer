import numpy as np
import cv2


class LBP:

    def _thresholded(self, center, pixels):
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

    def run(self, img, debug=False):

        # Resize image to be same size as classified faces
        img = cv2.resize(img, (120, 120), interpolation=cv2.INTER_CUBIC)

        # Copy the image to transform
        transformed_img = img.copy()

        # Iterate over each pixel in image
        for x in range(0, len(img)):
            for y in range(0, len(img[0])):
                center = img[x, y]
                top_left = self._get_pixel(img, x - 1, y - 1)
                top_up = self._get_pixel(img, x, y - 1)
                top_right = self._get_pixel(img, x + 1, y - 1)
                right = self._get_pixel(img, x + 1, y)
                left = self._get_pixel(img, x - 1, y)
                bottom_left = self._get_pixel(img, x - 1, y + 1)
                bottom_right = self._get_pixel(img, x + 1, y + 1)
                bottom_down = self._get_pixel(img, x, y + 1)

                res = self._thresholded(center, [top_left, top_up, top_right, right, bottom_right,
                                                 bottom_down, bottom_left, left])

                transformed_img.itemset((x, y), res)

        if debug:
            cv2.imwrite('debug/thresholded_image.png', transformed_img)

        return np.histogram(img.flatten(), 256, [0, 256])  # histogram and bins
