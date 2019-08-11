"""
Author: Kfir Zvi
Date: 11/08/19
"""

from collections import Counter

import cv2
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle

# Constants
# This determines the maximal component width in the evolution map
MAX_COMPONENT_WIDTH = 200
# This determines the size of the gaussian kernel that is being performed on the evolution map representing the
# relative total area. It is set to 17 after performing several experiments with it, with 17 usually showing the best
# results
GAUSSIAN_KERNEL_SIZE = 17


class EvolutionMapBuilder:
    """
    This class holds all the methods that are relevant to building the evolution map from scratch.
    It has 2 interface methods which are: show_em, build_em.
    show_em is used to build a graph of the evolution map (a 2d-nd-array)
    build_em is used to build the evolution map given an image

    The structure of the class can be changed, I have built it this way for my own convenience
    """

    @staticmethod
    def __threshold_img(img, threshold=127):
        """
        Performs a threshold on an image, returning a THRESH_BINARY result.
        It stores each threshed result image in the "threshed" folder for
        debugging purposes.
        :param img: a 2d-nd-array
        :param threshold: the value of the minimum threshold
        :return: a 2d-nd-array containing the result
        """
        _, threshed = cv2.threshold(img, threshold, 255, cv2.THRESH_BINARY)
        cv2.imwrite(f'threshed/output-{threshold}.jpg', img)
        return threshed

    @staticmethod
    def __invert_img(img):
        """
        Inverts the image. It is used after the threshold is performed
        to revert the black and white colors in the image.

        :param img: 2d-nd-array of an image
        :return: 2d-nd-array of the inverted image
        """
        backup = img.copy()  # taking backup of the input image
        return 255 - backup  # colour inversion

    @staticmethod
    def __run_gaussian_kernel_filtering(em):
        """
        Performs gaussian filtering on the evolution map representing the relative total area.
        This step is required according to the article.
        The kernel size is determined by the constant GAUSSIAN_KERNEL_SIZE
        :param em: 2d-nd-array representing the evolution map of the relative total area
        :return: 2d-nd-array of the em after the gaussian filter
        """
        blur = cv2.GaussianBlur(em, (GAUSSIAN_KERNEL_SIZE, GAUSSIAN_KERNEL_SIZE), cv2.BORDER_DEFAULT)
        return blur

    @staticmethod
    def __find_contours(img):
        """
        Used to find all the contours in the image in the form of tuples.
        Each tuple contains all the relevant information about a contour:
        width, height, x-y coordinates, etc.
        :param img: 2d-nd-array representing an image after threshold
        :return: a list of contours tuples
        """
        _, contours, _ = cv2.findContours(img, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
        return contours

    @staticmethod
    def __mark_countours_rects(img, contours):
        """
        Used to mark all the contours in an image mainly for debugging purposes
        :param img: 2d-nd-array, representing an image
        :param contours: a list of contours to be marked on the image
        :return: 2d-nd-array, representing an image with the marked contours as rectangles on it
        """
        img = img.copy()
        for i, contour in enumerate(contours):
            x, y, w, h = cv2.boundingRect(contour)
            if w >= MAX_COMPONENT_WIDTH:
                continue
            cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 1)
        return img

    @staticmethod
    def show_em(em, blobs=None):
        """
        Used to build a graph for the evolution map
        :param em: 2d-nd-array representing an evolution map
        :param blobs: optional, a list of blobs objects to be marked on the evolution map
        :return: None
        """
        plt.imshow(em, interpolation=None, cmap='seismic')
        plt.gca().invert_yaxis()

        if blobs:
            for blob in blobs:
                plt.gca().add_patch(
                    Rectangle((blob.x, blob.y), blob.width, blob.height, linewidth=1, edgecolor='r', facecolor='none'))

        plt.show()

    @staticmethod
    def build_em(img):
        """
        :param img: 2d-array, representing an image in gray-scale
        :return: tuple,
                    index 0 - evolution map calculated according the relative total area (a 3-d array)
                    index 1 - evolution map calculated as number of components of width w at threshold g
        """
        original_img = img.copy()

        relative_total_area_em = np.zeros((256, MAX_COMPONENT_WIDTH))
        components_count_em = np.zeros((256, MAX_COMPONENT_WIDTH))

        for threshold in range(256):
            img = original_img.copy()
            img = EvolutionMapBuilder.__threshold_img(img, threshold)
            inverted_img = EvolutionMapBuilder.__invert_img(img)
            contours = EvolutionMapBuilder.__find_contours(inverted_img)

            # optional: Writing output results
            # marked_img = EvolutionMapBuilder.__mark_countours_rects(img, contours)
            #             # cv2.imwrite(f'res/output-{threshold}.jpg', marked_img)
            # --------------------------------

            contours_under_max_width = list(filter(lambda c: cv2.boundingRect(c)[2] < MAX_COMPONENT_WIDTH, contours))

            """ calculates number of components in threshold per each width"""

            width_freqs = Counter(map(lambda c: cv2.boundingRect(c)[2], contours_under_max_width))

            for width, freq in width_freqs.items():
                components_count_em[threshold][width] += freq

            """ calculates relative total area """

            for c in contours_under_max_width:
                _, _, w, h = cv2.boundingRect(c)
                relative_total_area_em[threshold][w] += w * h  # add area of each component

        relative_total_area_em /= (img.shape[0] * img.shape[
            1])  # normalize values of relative area by dividing in the image's total area

        relative_total_area_em = EvolutionMapBuilder.__run_gaussian_kernel_filtering(relative_total_area_em)

        return relative_total_area_em, components_count_em


if __name__ == '__main__':
    img = cv2.imread('txt5.png', cv2.IMREAD_GRAYSCALE)

    relative_total_area_em, components_count_emap = EvolutionMapBuilder.build_em(img)

    EvolutionMapBuilder.show_em(relative_total_area_em)

    np.save('emap-relative', relative_total_area_em)
    np.save('emap-components-count', components_count_emap)
