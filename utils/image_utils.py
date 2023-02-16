from typing import List

import numpy as np
import pydicom


def apply_2d_mask_to_image_stack(image_stack, mask):

    res = np.zeros(image_stack.shape)

    for i in range(image_stack.shape[0]):
        res[i, :, :,] = (
            mask * image_stack[i, :, :]
        )

    return res


def clip_image(image: np.array, indices: List[int], mode: str = "single") -> np.array:
    """Clip 2D image or a stack of 2D images.

    Parameters
    ----------
    image : np.array
        Image to be clipped. Either a single image (set mode='single')
        or a stack of images (set mode='stack').
        for a stack of type (nslices, nsettings, nx, ny)
    indices : List[int]
        List of indices [clip_top, clip_bottom, clip_left, clip_right]
        for the clip image[clip_top:-clip_bottom, clip_left:-clip_right]
    mode : str
        'single' or 'stack'. default is 'single'

    Returns
    -------
    np.array
        clipped image stack

    """
    x1 = indices[0]
    x2 = indices[1]
    y1 = indices[2]
    y2 = indices[3]

    if mode == "single":
        return image[x1:-x2, y1:-y2]

    elif mode == "stack":

        tmp1 = image[0, x1:-x2, y1:-y2]
        tmp2 = np.zeros((image.shape[0], tmp1.shape[0], tmp1.shape[1]))

        for i in range(tmp2.shape[0]):
            tmp2[i, :, :] = image[i, x1:-x2, y1:-y2]

        return tmp2

    else:
        assert False


def z_normalize_image(image: np.array, mode: str = "single") -> np.array:
    """z-normalize 2D magnitude images.

    Parameters
    ----------
    image : np.array()
        Magnitude image to be normalized. Either a single image
        (set mode='single') or a stack of images (set mode='stack').
    mode : str
        'single' or 'stack', default is 'single'

    Returns
    -------
    np.array()
        z-normalized image stack

    """
    if mode == "single":

        return (image - image.mean()) / image.std()

    elif mode == "stack":

        # assume first dim is nr of images
        nr_images = image.shape[0]

        for i in range(nr_images):
            tmp = image[i, :, :]
            tmp = (tmp - tmp.mean()) / tmp.std()
            image[i, :, :] = tmp

        return image

    else:
        assert False


def find_clip_indices(image: np.array, slack: int = 0):
    """Find the row- and column indices of the unused
    pixelspace around a 2D image.

    Parameters
    ----------
    image : np.array
        magnitude 2D image
    slack : int, optional
        set number of extra pixels, by default 0

    Returns
    -------
    x1: int
        index of first row containing positive values
    x2: int
        index of last row with magnitude positive values
    y1: int
        index of first image column with magnitude positive values
    y2: int
        index of last image column with magnitude positive values

    """
    nrows, ncols = image.shape

    for row in range(nrows):
        if np.count_nonzero(image[row, :]) > 0:
            x1 = row - slack
            break

    temp = np.flipud(image)

    for row in range(nrows):
        if np.count_nonzero(temp[row, :]) > 0:
            x2 = row - slack
            break

    for col in range(ncols):
        if np.count_nonzero(image[:, col]) > 0:
            y1 = col - slack
            break

    temp = np.fliplr(image)

    for col in range(ncols):
        if np.count_nonzero(image[:, col]) > 0:
            y2 = col - slack
            break

    return x1, x2, y1, y2


def load_dicom_magnitude_image(file_path: str, scale: float = 1.0) -> np.array:
    """Load magnitude 2D dicom image.

    Parameters
    ----------
    file_path : str
        file path to .dicom file
    scale : float
        scaling factor
    Returns
    -------
    np.array
        magnitude 2D dicom image

    """
    tmp = pydicom.read_file(file_path)
    img = scale * tmp.RescaleSlope * np.array(tmp.pixel_array)

    return img
