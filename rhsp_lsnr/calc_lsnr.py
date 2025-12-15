'''
Functions for LSNR calculation given the segmentation and image
'''

# %%
import numpy as np
import SimpleITK as sitk
import matplotlib.pyplot as plt
import pandas as pd
import scipy.ndimage


# %%
def find_max_slice_for_labels(label_img: np.array):
    '''
    Find the slice with the maximum area for each label in the label image

    Parameters
    ----------
    label_img : np.array
        3D label image in [z, y, x] format

    Returns
    -------
    label_slices : np.array
        1D array of slice indices with maximum area for each label
    label_areas : np.array
        1D array of maximum areas for each label
    '''

    label_areas = []
    for islice in range(label_img.shape[0]):
        hist = np.histogram(label_img[islice], bins=np.arange(0, label_img.max() + 2))[0]
        label_areas.append(hist[1:])
    label_areas = np.array(label_areas)

    # find the max area and slices for each label
    label_slices = np.argmax(label_areas, axis=0)
    label_areas = np.max(label_areas, axis=0)

    return label_slices, label_areas


def remove_extreme_labels(
    label_img: np.array,
    label_slices: np.array,
    label_areas: np.array,
    ref_area: float,
    area_tol: float
):
    '''
    Remove labels that have areas too small or too large compared to the reference area

    Parameters
    ----------
    label_img : np.array
        3D label image in [z, y, x] format
    label_slices : np.array
        1D array of slice indices with maximum area for each label
    label_areas : np.array
        1D array of maximum areas for each label
    ref_area : float
        Reference area to compare against
    area_tol : float
        Tolerance factor for area comparison

    Returns
    -------
    inds: list
        List of label indices that are preserved
    label_small : np.array
        3D label image of removed small labels in [z, y, x] format
    lable_large : np.array
        3D label image of removed large labels in [z, y, x] format
    label_preserve : np.array
        3D label image of preserved labels in [z, y, x] format
    '''
    # remove small and large labels
    inds_small = np.where(label_areas < ref_area * area_tol)[0]
    inds_large = np.where(label_areas > ref_area / area_tol)[0]
    # remove the labels whose max area slice is the first or last slice
    inds_first_last_slice = np.where((label_slices == 0) | (label_slices == label_img.shape[0] - 1))[0]
    print(
        'Labels in total {}, small {}, large {}, edge slices {}'.format(
            len(label_areas), len(inds_small), len(inds_large), len(inds_first_last_slice)
        ),
        flush=True
    )

    # display removed
    label_small = np.zeros_like(label_img)
    for i in inds_small:
        label_small[label_img == i + 1] = i + 1
    lable_large = np.zeros_like(label_img)
    for i in inds_large:
        lable_large[label_img == i + 1] = i + 1

    # For the rest, mark the center and area
    inds = [i for i in range(len(label_slices)) if i not in inds_small and i not in inds_large]
    inds = [i for i in inds if i not in inds_first_last_slice]
    print('Labels to be preserved:', len(inds), flush=True)
    label_preserve = np.zeros_like(label_img)
    for i in inds:
        label_preserve[label_img == i + 1] = i + 1

    return inds, label_small, lable_large, label_preserve


def find_label_centers(label_img: np.array, label_slices: np.array, label_areas: np.array, inds: list):
    '''
    Find the centers and areas of specified labels based on their maximum area slices

    Parameters
    ----------
    label_img : np.array
        3D label image in [z, y, x] format
    label_slices : np.array
        1D array of slice indices with maximum area for each label
    label_areas : np.array
        1D array of maximum areas for each label
    inds : list
        List of label indices to process

    Returns
    -------
    df_label : pd.DataFrame
        DataFrame containing label indices, centers (z, y, x), and areas
    label_img_max : np.array
        3D label image with only the maximum area slices for specified labels, in [z, y, x] format
    '''
    label_img_max = np.zeros_like(label_img)
    label_inds = []
    label_centers = []
    label_areas_max = []
    for i in inds:
        islice = label_slices[i]
        label = i + 1

        # mark on the image
        slice_img = label_img[islice] == label
        label_img_max[islice][slice_img] = label

        # find the center
        y, x = np.where(slice_img)
        label_centers.append([islice, y.mean(), x.mean()])

        # find the area
        label_areas_max.append(label_areas[i])

        label_inds.append(label)
    label_centers = np.array(label_centers)

    df_label = pd.DataFrame({
        'Label': label_inds,
        'CenterZ': label_centers[:, 0],
        'CenterY': label_centers[:, 1],
        'CenterX': label_centers[:, 2],
        'Area': label_areas_max
    })

    return df_label, label_img_max
