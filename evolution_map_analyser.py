"""
Author: Kfir Zvi
Date: 11/08/19
"""

from functools import reduce
from math import floor
import cv2
import numpy as np
import scipy.optimize as opt

from evolution_map_builder import EvolutionMapBuilder
from blob import Blob


class EvolutionMapAnalyser:
    # constants - for the score function
    A = 1
    C1 = 1
    C2 = 1

    @staticmethod
    def __find_peaks(emap, max_peaks=20):
        """
        Takes an evolution map, extracts all the unique values in it, and sorts them from high to low.
        Then it iterates the unique values, and tries to find the peaks of the components in
        the evolution map.
        Eventually it returns the first max_peaks peaks of the components it found, with default of 20 peaks

        :param emap: 2d-nd-array, representing an evolution map
        :param max_peaks: the maximum number of components' peaks to find
        :return: 2d-nd-array of shape (#peaks, 2), listing the x-y coordinates of the peaks found
        """
        steps = np.flipud(np.sort(np.unique(emap)))
        # does not iterate the first 10% of low values
        steps = steps[1:floor(steps.shape[0] * 0.9)]

        peaks = np.zeros((0, 2)).astype(np.int)

        for thresh in steps:
            bin_map = np.where(emap > thresh, 1, 0).astype(np.uint8)

            # note - nb_labels includes 0 which is a non-component or a background component
            nb_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(bin_map)

            if nb_labels > 1:
                # remove previously found peaks
                for i in range(peaks.shape[0]):
                    peak_id = labels[peaks[i, 0], peaks[i, 1]]
                    labels = np.where(labels == peak_id, 0, labels)

                # add new component

                """ These TODOs rarely happen, and are not implemented in the MATLAB code as well """
                # TODO: handle addition of more than one peak in iteration
                # TODO: take the center point of the new component instead of a random one

                new_peak_idx = np.where(labels != 0)
                if new_peak_idx[0].size and new_peak_idx[1].size:
                    (new_peak_x,), (new_peak_y,) = new_peak_idx
                    peaks = np.append(peaks, [[int(new_peak_x), int(new_peak_y)]], axis=0)

                if peaks.shape[0] >= max_peaks:
                    break
        return peaks

    @staticmethod
    def __expand_peak(emap, x, y, expansion_percentage_limit=0.5):
        """
        Used to expand a peak to an entire component

        :param emap: 2d-nd-array, representing an evolution map
        :param x: int, the x coordinate of the peak in the emap
        :param y: int, the y coordinate of the peak in the emap
        :param expansion_percentage_limit:
                    the minimum percentage that needed for the expansion of the
                    component in each expansion iteration
        :return: tuple made of:
                    1. stats of the component map
                    2. the component map
                note:
                    the component map is basically a binary array which contains "1"s
                    where the component found resides, and "0"s elsewhere
        """
        thresh = emap[x, y]
        other_components = np.zeros(emap.shape)
        component_map = np.zeros(emap.shape)
        component_map_stats = None

        if thresh <= 0:
            return [], []

        while thresh > 0:
            bin_map = np.where(emap >= thresh, 1, 0).astype(np.uint8)

            nb_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(bin_map)

            our_component = np.where(labels == labels[x, y], 1, 0)

            """ This is the problematic step which causes blobs to not be recognized properly.
                In order to return it, just uncomment the next 2 lines of code """
            # # checks for overlaps with other components
            # if np.any(np.logical_and(our_component, other_components)):
            #     break

            addition = np.logical_and(np.logical_not(component_map), our_component)
            addition_avg = np.mean(emap[addition])
            addition_percentage_of_orig = addition_avg / emap[x, y]

            if addition_percentage_of_orig < expansion_percentage_limit:
                break

            component_map = our_component.astype(np.uint8)
            component_map_stats = stats[labels[x, y], :]
            other_components = np.logical_and(labels, np.logical_not(our_component))

            dialated = cv2.dilate(component_map, np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], np.uint8))
            comp_surrounding = (dialated - component_map) * emap
            thresh = np.max(comp_surrounding)

        return component_map_stats, component_map


    @staticmethod
    def __expand_blobs(peaks, emap):
        """
        Used to expand the peaks found in the previous step to blobs with the help of
        the expand_peak method
        :param peaks: 2d-nd-array, listing the peaks in the emap
        :param emap: 2d-nd-array, representing an evolution map
        :return: list of Blob objects, listing all the blobs found in the evolution map
        """
        blobs = []
        mark_map = np.zeros(emap.shape)

        for peak_x, peak_y in peaks:
            peak_stats, peak_comp = EvolutionMapAnalyser.__expand_peak(emap, peak_x, peak_y)

            # check that a component was found
            if np.count_nonzero(peak_comp) == 0:
                continue

            # check for intersection between the current component and the components found so far
            if np.any(np.logical_and(mark_map, peak_comp)):
                continue

            mark_map = mark_map + peak_comp

            blob_x = peak_stats[cv2.CC_STAT_LEFT]
            blob_y = peak_stats[cv2.CC_STAT_TOP]
            blob_width = peak_stats[cv2.CC_STAT_WIDTH]
            blob_height = peak_stats[cv2.CC_STAT_HEIGHT]
            blob_component = peak_comp

            blob = Blob(blob_component, blob_x, blob_y, blob_width, blob_height)

            # blob.gaussian = model_by_gaussian(emap, blob)

            blobs.append(blob)

        return blobs

    @staticmethod
    def __calc_blobs_scores(relative_total_area_em, components_count_em, blobs):
        """
        Calculates the score for each blob, and sets the score field of each Blob object accordingly
        :param relative_total_area_em: 2d-nd-array, representing the relative-total-area evolution map
        :param components_count_em: 2d-nd-array, representing the components-count evolution map
        :param blobs: a list of Blob objects
        :return: None
        """
        for blob in blobs:
            nb_blob_components = np.sum(components_count_em[blob.y:blob.y + blob.height, blob.x: blob.x + blob.width])
            blob_area_percentage_of_image = np.sum(
                relative_total_area_em[blob.y:blob.y + blob.height, blob.x: blob.x + blob.width])

            p = blob_area_percentage_of_image
            n = nb_blob_components

            a = EvolutionMapAnalyser.A
            c1 = EvolutionMapAnalyser.C1
            c2 = EvolutionMapAnalyser.C2

            blob.score = (a * p) * (1 / (1 + np.exp(-c1 * (n - c2))))


    @staticmethod
    def analyze_em(relative_total_area_em, components_count_emap):
        """
        An interface method, which analyzes an evolution map and finds its blobs and their respective scores
        :param relative_total_area_em: 2d-nd-array, representing the relative-total-area evolution map
        :param components_count_emap: 2d-nd-array, representing the components-count evolution map
        :return: None
        """
        emap_peaks = EvolutionMapAnalyser.__find_peaks(relative_total_area_em)

        blobs = EvolutionMapAnalyser.__expand_blobs(emap_peaks, relative_total_area_em)

        EvolutionMapAnalyser.__calc_blobs_scores(relative_total_area_em, components_count_emap, blobs)

        # can be used to show only top-score blob
        # max_blob = reduce(lambda max_blob, b: max_blob if b.score < max_blob.score else b, blobs, blobs[0])

        EvolutionMapBuilder.show_em(relative_total_area_em, blobs)


if __name__ == '__main__':
    relative_total_area_em = np.load('emap-relative.npy')
    components_count_emap = np.load('emap-components-count.npy')

    EvolutionMapAnalyser.analyze_em(relative_total_area_em, components_count_emap)

    #
    # emap_peaks = find_peaks(relative_total_area_em)
    #
    # blobs = expand_blobs(emap_peaks, relative_total_area_em)
    #
    # calc_blobs_scores(relative_total_area_em, components_count_emap, blobs)
    #
    # max_blob = reduce(lambda max_blob, b: max_blob if b.score < max_blob.score else b, blobs, blobs[0])
    #
    # # show_gaussians(relative_total_area_em, blobs)
    #
    # EvolutionMapBuilder.show_em(relative_total_area_em, blobs)
