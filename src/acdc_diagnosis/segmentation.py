"""Reconstruct the left ventricle (LV) from a segmentation that only contains RV and myocardium.

The LV cavity is the region enclosed by the myocardium. For each 2D slice:
1. fill the outer contour of the myocardium and keep the pixels that are not myocardium;
2. keep the largest connected component as the LV;
3. if the myocardium ring is open (no enclosed region), close it with a one-pixel
   morphological dilation, fill it, dilate the result back by one pixel and add the
   isolated background pixels left between the LV and the myocardium.

Usage:
    from acdc_diagnosis.segmentation import fill_lv_volume, remove_lv
    seg_with_lv = fill_lv_volume(seg_without_lv)       # (H, W, slices) label volume
"""

import cv2
import numpy as np

from acdc_diagnosis import LV, MYO, RV

CROSS = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], dtype=np.uint8)


def remove_lv(seg):
    """Return a copy of a label volume with the LV label set to background.

    Args:
        seg (numpy.ndarray): label volume.

    Returns:
        numpy.ndarray: copy of seg without label LV.
    """
    return np.where(seg == LV, 0, seg)


def enclosed_region(mask):
    """Pixels enclosed by the outer contours of a binary mask but not part of it.

    Args:
        mask (numpy.ndarray): uint8 binary mask (H, W).

    Returns:
        numpy.ndarray: boolean mask of the enclosed pixels.
    """
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    filled = cv2.drawContours(np.zeros_like(mask), contours, -1, 1, thickness=cv2.FILLED)
    return (filled == 1) & (mask == 0)


def largest_component(region):
    """Keep the largest 8-connected component of a non-empty boolean mask.

    Args:
        region (numpy.ndarray): boolean mask with at least one True pixel.

    Returns:
        numpy.ndarray: boolean mask of the largest component.
    """
    _, labels, stats, _ = cv2.connectedComponentsWithStats(region.astype(np.uint8), connectivity=8)
    return labels == 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])


def fill_isolated_pixels(seg, box):
    """Label as LV the background pixels surrounded by LV and myocardium only.

    Pixels are visited in raster order inside the bounding box, so a pixel labelled
    LV can complete the neighbourhood of the next one.

    Args:
        seg (numpy.ndarray): 2D label slice, modified in place.
        box (tuple[int, int, int, int]): x, y, width, height as returned by cv2.boundingRect.

    Returns:
        numpy.ndarray: the updated slice.
    """
    x, y, w, h = box
    for i in range(y, y + h):
        for j in range(x, x + w):
            patch = seg[i - 1:i + 2, j - 1:j + 2]
            if seg[i, j] == 0 and np.sum(patch == 0) == 1 and LV in patch and MYO in patch:
                seg[i, j] = LV
    return seg


def fill_open_myocardium(seg, mask):
    """Segment the LV when the myocardium ring is open (fallback of fill_lv_slice).

    Args:
        seg (numpy.ndarray): 2D label slice without LV.
        mask (numpy.ndarray): uint8 binary mask of the myocardium.

    Returns:
        numpy.ndarray: slice with the LV label added.
    """
    closed = cv2.dilate(mask, CROSS)
    lv = cv2.dilate(enclosed_region(closed).astype(np.uint8), CROSS) == 1
    out = np.where(lv & (seg == 0), LV, seg)
    contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    return fill_isolated_pixels(out, cv2.boundingRect(contours[0]))


def fill_lv_slice(seg):
    """Add the LV label to one 2D slice containing RV and myocardium labels.

    Slices without myocardium are returned unchanged.

    Args:
        seg (numpy.ndarray): 2D label slice without LV.

    Returns:
        numpy.ndarray: slice with the LV label added.
    """
    if MYO not in seg:
        return seg
    mask = (seg > RV).astype(np.uint8)
    region = enclosed_region(mask)
    if not region.any():
        return fill_open_myocardium(seg, mask)
    return np.where(largest_component(region), LV, seg)


def fill_lv_volume(seg):
    """Add the LV label to every slice of a label volume.

    Args:
        seg (numpy.ndarray): label volume (H, W, slices) without LV.

    Returns:
        numpy.ndarray: new label volume with the LV label.
    """
    slices = [fill_lv_slice(np.ascontiguousarray(seg[:, :, k])) for k in range(seg.shape[2])]
    return np.stack(slices, axis=2)
