'''
Functions for auto segmentation
'''

# %%
import SimpleITK as sitk
import pydicom
import numpy as np
import scipy.ndimage

import matplotlib.pyplot as plt
import skimage.segmentation
import ct_projector.prior.numpy.recon_prior as recon_prior


# %%
def extract_volume(input_filename, z_interp=1, num_zeros=25):
    '''
    Extract volume from DICOM series

    Parameters
    ----------
    input_filename : str
        Path to one of the DICOM files in the series
    z_interp : int, optional
        Interpolation factor in z direction, by default 1
    num_zeros : int, optional
        Continuous number of zeros to determine if the current row/column is out of the FOV

    Returns
    -------
    volume : SimpleITK.Image
        Extracted volume
    '''
    dcm = pydicom.dcmread(input_filename)
    img = dcm.pixel_array

    if len(img.shape) == 4:
        img = img[..., 0]  # should work for both rgb and ybr

    assert len(img.shape) == 3, f'Expected 3D image, got {img.shape}'

    # starting from the center, detect the first row/column that has num_zeros zeros on each side
    cx = img.shape[2] // 2
    cy = img.shape[1] // 2
    y_vals = np.max(img[:, :, cx - num_zeros:cx + num_zeros], axis=(0, 2))
    x_vals = np.max(img[:, cy - num_zeros:cy + num_zeros, :], axis=(0, 1))

    # crop the FOV
    y_start = np.where(y_vals[:img.shape[1] // 2] == 0)[0][-1] + 1
    y_end = np.where(y_vals[img.shape[1] // 2:] == 0)[0][0] + img.shape[1] // 2
    x_start = np.where(x_vals[:img.shape[2] // 2] == 0)[0][-1] + 1
    x_end = np.where(x_vals[img.shape[2] // 2:] == 0)[0][0] + img.shape[2] // 2

    res = img[:, y_start:y_end, x_start:x_end]

    # interpolate in z direction if needed
    if z_interp > 1:
        res = scipy.ndimage.zoom(res, (z_interp, 1, 1), order=1)

    # find the spacing information and create SimpleITK image
    sitk_res = sitk.GetImageFromArray(res)
    try:
        dx = dcm.SequenceOfUltrasoundRegions[0].PhysicalDeltaX
        dy = dcm.SequenceOfUltrasoundRegions[0].PhysicalDeltaY
        sitk_res.SetSpacing([dx, dy, (dx + dy) / 2])
    except Exception:
        print('Could not find spacing information in the dicom file')
        sitk_res.SetSpacing([1, 1, 1])

    return sitk_res


# %%
def non_local_mean_denoising(img, sigma_scale=0.125, gpu_id=0):
    '''
    Apply non-local mean denoising to the input image

    Parameters
    ----------
    img : np.ndarray
        Input image in [z, y, x] format
    sigma_scale : float
        Scale factor for noise standard deviation
    gpu_id : int, optional
        GPU ID to use, by default 0

    Returns
    -------
    img_nlm : np.ndarray
        Denoised image in [z, y, x] format
    '''
    std = np.std(img)

    recon_prior.set_device(gpu_id)
    img_nlm = recon_prior.nlm(
        np.copy(img[np.newaxis], 'C'),
        np.copy(img[np.newaxis], 'C'),
        std * sigma_scale,
        [9, 9, 9],
        [3, 3, 3],
        1
    )[0]

    return img_nlm


def n4_correction(input_img, downsample_rate=4, n4_control_points=[4, 8, 4]):
    '''
    Apply N4 bias field correction to the input image
    Parameters
    ----------
    input_img : np.ndarray
        Input image in [z, y, x] format
    downsample_rate : int, optional
        Downsample rate for N4 correction, by default 4
    n4_control_points : list, optional
        Control points for N4 correction, by default [4, 8, 4]

    Returns
    -------
    img_n4 : np.ndarray
        N4 corrected image in [z, y, x] format
    '''

    sitk_input = sitk.GetImageFromArray(input_img)

    # calculate the bias field based on the first mr
    img = sitk.Cast(sitk_input, sitk.sitkFloat32)
    img = sitk.Shrink(img, [downsample_rate] * 3)

    # n4 correction
    corrector = sitk.N4BiasFieldCorrectionImageFilter()
    corrector.SetNumberOfControlPoints(n4_control_points)
    corrector.SetConvergenceThreshold(0.0001)
    corrector.SetMaximumNumberOfIterations([50, 40, 30])
    img_n4 = corrector.Execute(img)

    # hand calculate the bias field
    bias_field = sitk.GetArrayFromImage(img_n4) / (sitk.GetArrayFromImage(img) + 1e-4)
    bias_field = scipy.ndimage.zoom(bias_field, downsample_rate, order=1)
    bias_field = scipy.ndimage.median_filter(bias_field, size=3)
    pads = [[0, 0], [0, 0], [0, 0]]
    target_size = sitk_input.GetSize()[::-1]
    current_size = bias_field.shape
    for i in range(3):
        pads[i][1] = target_size[i] - current_size[i]
    bias_field = np.pad(bias_field, pads, mode='edge')
    bias_field = sitk.GetImageFromArray(bias_field)
    bias_field.CopyInformation(sitk_input)

    # correct the input image
    img_n4 = sitk.Cast(sitk.Cast(sitk_input, sitk.sitkFloat32) * bias_field, sitk.sitkInt16)

    return sitk.GetArrayFromImage(img_n4)


def adaptive_threshold(img, num_depth_groups=10, threshold_factor=2, show_plots=True):
    '''
    Apply adaptive thresholding to segment the image

    Parameters
    ----------
    img : np.ndarray
        Input image in [z, y, x] format
    num_depth_groups : int, optional
        Number of depth groups to divide the image into, by default 10
    threshold_factor : float, optional
        Factor to multiply the background std to determine the threshold, by default 2

    Returns
    -------
    masks : np.ndarray
        Binary masks in [z, y, x] format
    '''
    y_per_group = img.shape[1] / num_depth_groups

    bin_max = np.percentile(img, 99.9)
    bin_width = bin_max / 50
    bins = np.arange(0, bin_max + bin_width, bin_width)

    masks = []

    for i in range(num_depth_groups):
        istart = int(np.round(y_per_group * i))
        iend = int(np.round(y_per_group * (i + 1)))

        if i == num_depth_groups - 1:
            img_slice = img[:, istart:, :]
        else:
            img_slice = img[:, istart:iend, :]

        # estimate the background with histogram analysis
        hist_slice, bin_edges = np.histogram(img_slice.flatten(), bins=bins)
        # find the peak
        ipeak = np.argmax(hist_slice)
        # right side of the peak are all background, mirror them to the peaks left
        hist_background = np.zeros_like(hist_slice)
        hist_background[ipeak] = hist_slice[ipeak]  # peak
        hist_background[ipeak:] = hist_slice[ipeak:]  # right side
        nright = len(hist_slice) - ipeak - 1
        nleft = ipeak
        nmirror = min(nright, nleft)
        hist_background[ipeak - nmirror:ipeak] = hist_slice[ipeak + 1:ipeak + nmirror + 1][::-1]  # left side

        # estimate the mean and std of the background
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
        # construct an array following the histogram of the background
        bg = []
        for i in range(len(hist_background)):
            bg += [bin_centers[i]] * hist_background[i]
        mean = np.mean(bg)
        std = np.std(bg)
        th = mean - threshold_factor * std

        mask = np.where(img_slice < th, 1, 0)
        masks.append(mask)

        if show_plots:
            plt.figure(figsize=[6, 3])
            plt.subplot(121)
            plt.imshow(img_slice[img_slice.shape[0] // 2], 'gray', vmin=0, vmax=bin_max)
            plt.title('Depth {0} to {1}'.format(istart, iend))
            plt.subplot(122)
            plt.plot(bin_edges[:-1], hist_slice)
            plt.plot(bin_edges[:-1], hist_background)
            plt.plot([th, th], [0, np.max(hist_slice)], 'r')
            plt.tight_layout()
            plt.show()

    masks = np.concatenate(masks, axis=1)

    return masks


def denoise_mask(mask, num_openings=3):
    '''
    Apply morphological opening to denoise the binary mask

    Parameters
    ----------
    mask : np.ndarray
        Input binary mask in [z, y, x] format
    num_opening : int, optional
        Number of times to apply morphological opening, by default 3

    Returns
    -------
    mask_denoised : np.ndarray
        Denoised binary mask in [z, y, x] format
    '''
    processed_mask = scipy.ndimage.binary_erosion(mask, iterations=num_openings, border_value=1)
    processed_mask = scipy.ndimage.binary_dilation(processed_mask, iterations=num_openings, border_value=0)

    return processed_mask


def split_spheres(mask, peak_min_distance=5):
    '''
    Split connected spheres in the binary mask using watershed algorithm

    Parameters
    ----------
    mask : np.ndarray
        Input binary mask in [z, y, x] format
    peak_min_distance : int, optional
        Minimum distance between peaks for watershed seeds, by default 5

    Returns
    -------
    labels : np.ndarray
        Labeled mask in [z, y, x] format
    peak_morph_mask : np.ndarray
        Peaks used as seeds for watershed in [z, y, x] format
    '''

    # watershed algorithm to separate the split labels
    distance = scipy.ndimage.distance_transform_edt(mask)
    peaks_morph = skimage.feature.peak_local_max(distance, min_distance=peak_min_distance, exclude_border=False)
    peak_morph_mask = np.zeros_like(mask)
    peak_morph_mask[tuple(peaks_morph.T)] = 1
    peak_morph_mask[mask == 0] = 0

    seeds = skimage.measure.label(peak_morph_mask)
    labels = skimage.segmentation.watershed(-distance, seeds, mask=mask)

    return labels, peak_morph_mask
