'''
Functions for LSNR calculation given the segmentation and image
'''

# %%
import numpy as np
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
    inds_small = np.where(label_areas < ref_area * area_tol[0])[0]
    inds_large = np.where(label_areas > ref_area * area_tol[1])[0]
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


# %%
def calc_crn_on_patch(
    df_label_patch: pd.DataFrame,
    bg_template: np.array,
    input_img: np.array,
    mask_img: np.array,
    label_img_max: np.array,
):
    '''
    Calculate CNR for each sphere in the given patch

    Parameters
    ----------
    df_label_patch : pd.DataFrame
        DataFrame containing label indices and centers for spheres within the patch
    bg_template : np.array
        2D binary array indicating the background area around a spheres, in [y, x] format
    input_img : np.array
        3D input image in [z, y, x] format
    mask_img : np.array
        3D mask image in [z, y, x] format
    label_img_max : np.array
        3D label image with only the maximum area slices for specified labels, in [z, y, x] format

    Returns
    -------
    cnrs_patch : list
        List of CNR values for each sphere in the patch
    contrast_patch : list
        List of contrast values for each sphere in the patch
    noise_patch : list
        List of noise values for each sphere in the patch
    '''

    ipad = bg_template.shape[0] // 2

    # first pad all the images by ipad to avoid the edge effect
    input_img = np.pad(input_img, ((0, 0), (ipad, ipad), (ipad, ipad)), mode='constant')
    # pad 1 for mask so that the padded area will be automatically excluded
    mask_img = np.pad(mask_img, ((0, 0), (ipad, ipad), (ipad, ipad)), mode='constant', constant_values=1)
    label_img_max = np.pad(label_img_max, ((0, 0), (ipad, ipad), (ipad, ipad)), mode='constant')

    cnrs_patch = []
    contrast_patch = []
    noise_patch = []
    for i, row in df_label_patch.iterrows():
        iz = int(np.round(row['CenterZ']))
        iy = int(np.round(row['CenterY'])) + ipad  # adjust for the padding
        ix = int(np.round(row['CenterX'])) + ipad

        # crop to the patch whose size is 2 * ipad + 1
        input_img_patch = input_img[:, iy - ipad:iy + ipad + 1, ix - ipad:ix + ipad + 1]
        mask_img_patch = mask_img[:, iy - ipad:iy + ipad + 1, ix - ipad:ix + ipad + 1]
        label_img_max_patch = label_img_max[:, iy - ipad:iy + ipad + 1, ix - ipad:ix + ipad + 1]

        ball_mask = label_img_max_patch[iz]
        ball_mask = np.where(ball_mask == row['Label'], 1, 0)
        ball_mask = scipy.ndimage.binary_erosion(ball_mask, iterations=3)

        if ball_mask.sum() == 0:
            ball_val = input_img_patch[iz, ipad, ipad]  # if the mask to too small, use center value
        else:
            ball_val = np.mean(input_img_patch[iz][ball_mask])  # use average within the mask

        # identify the bg area with bg_radius around the ball
        # expand long z
        bg_mask = np.repeat(bg_template[np.newaxis, :, :], mask_img_patch.shape[0], axis=0)

        # the bg area should exclude 1s on the mask
        bg_mask[mask_img_patch == 1] = 0

        bg_val = input_img_patch[bg_mask == 1]

        bg_mean = bg_val.mean()
        bg_noise = bg_val.std()

        cnrs_patch.append((bg_mean - ball_val) / bg_noise)
        contrast_patch.append(bg_mean - ball_val)
        noise_patch.append(bg_noise)

    return cnrs_patch, contrast_patch, noise_patch


def calc_cnr_curve(
    df_label: pd.DataFrame,
    bg_radius: float,
    input_img: np.array,
    mask_img: np.array,
    label_img_max: np.array,
    patch_size: int,
    step_size: int,
):
    '''
    Calculate CNR curve along the depth of the image

    Parameters
    ----------
    df_label : pd.DataFrame
        DataFrame containing label indices and centers for all spheres
    bg_radius : float
        Radius of the background area around a sphere in pixels
    input_img : np.array
        3D input image in [z, y, x] format
    mask_img : np.array
        3D mask image in [z, y, x] format
    label_img_max : np.array
        3D label image with only the maximum area slices for specified labels, in [z, y, x] format
    patch_size : int
        Size of the depth patch to process at a time in pixels
    step_size : int
        Step size to move the depth patch in pixels

    Returns
    -------
    cnrs : np.array
        1D array of mean CNR values for each depth patch
    cnr_stds : np.array
        1D array of standard deviation of CNR values for each depth patch
    contrasts : np.array
        1D array of mean contrast values for each depth patch
    noises : np.array
        1D array of mean noise values for each depth patch
    nballs : np.array
        1D array of number of spheres considered for each depth patch
    depths : np.array
        1D array of depth values (center of each patch) in pixels
    '''

    cnrs = []
    cnr_stds = []
    contrasts = []
    noises = []
    nballs = []
    depths = []

    mask_img = scipy.ndimage.binary_dilation(mask_img, iterations=1)

    # create a bg template
    bg_template = np.zeros([int(bg_radius) * 2 + 3] * 2, int)
    ic = int(bg_radius) + 1
    for i in range(bg_template.shape[0]):
        for j in range(bg_template.shape[1]):
            if (i - ic)**2 + (j - ic)**2 <= bg_radius**2:
                bg_template[i, j] = 1

    for i in range(0, input_img.shape[1], step_size):
        print('Processing depth:', i, flush=True)
        if i + patch_size > input_img.shape[1]:
            istart = input_img.shape[1] - patch_size
        else:
            istart = i

        # find the balls whose center are within the patch
        df = df_label[
            (df_label['CenterY'] >= istart)
            & (df_label['CenterY'] < istart + patch_size)
        ]

        cnrs_patch, contrast_patch, noise_patch = calc_crn_on_patch(df, bg_template, input_img, mask_img, label_img_max)

        cnrs.append(np.mean(cnrs_patch))
        cnr_stds.append(np.std(cnrs_patch))
        contrasts.append(np.mean(contrast_patch))
        noises.append(np.mean(noise_patch))
        nballs.append(len(cnrs_patch))
        depths.append(istart + patch_size / 2)

    cnrs = np.array(cnrs)
    cnr_stds = np.array(cnr_stds)
    contrasts = np.array(contrasts)
    noises = np.array(noises)
    nballs = np.array(nballs)
    depths = np.array(depths)

    return cnrs, cnr_stds, contrasts, noises, nballs, depths
