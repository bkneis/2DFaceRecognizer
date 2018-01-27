import numpy as np
import cv2


class LBP:

    def _thresholded(self, center, pixels):
        out = []
        for a in pixels:
            if a >= center:
                out.append(1)
            else:
                out.append(0)
        return out

    def _get_pixel(self, l, idx, idy, default=0):
        try:
            return l[idx, idy]
        except IndexError:
            return default

    def run(self, img, debug=False):

        # Copy the image to transform
        transformed_img = img.copy()

        # Down sample the image to make processing faster
        img = cv2.resize(img, (200, 200), interpolation=cv2.INTER_CUBIC)
        transformed_img = cv2.resize(transformed_img, (200, 200), interpolation=cv2.INTER_CUBIC)

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

                values = self._thresholded(center, [top_left, top_up, top_right, right, bottom_right,
                                                    bottom_down, bottom_left, left])

                weights = [1, 2, 4, 8, 16, 32, 64, 128]
                res = 0
                for a in range(0, len(values)):
                    res += weights[a] * values[a]

                transformed_img.itemset((x, y), res)

        if debug:
            cv2.imwrite('debug/thresholded_image.png', transformed_img)

        return np.histogram(img.flatten(), 256, [0, 256]) # histogram and bins
