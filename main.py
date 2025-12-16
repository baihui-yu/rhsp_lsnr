'''
The main function to process a single DICOM and save LSNR information
'''

# %%
import os
import argparse
import sys
import subprocess
import time

import SimpleITK as sitk
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import rhsp_lsnr.autoseg as autoseg
import rhsp_lsnr.calc_lsnr as calc_lsnr

from rhsp_lsnr.locations import working_dir


# %%
def get_args(default_args=[]):
    parser = argparse.ArgumentParser('Process a single DICOM and save LSNR information')

    parser.add_argument('--input_filename', required=True)
    parser.add_argument('--output_dir', required=True)

    parser.add_argument('--sphere_diameter_cm', type=float, default=0.2,
                        help='Diameter of the spheres in the phantom in cm')
    parser.add_argument('--pixel_size_cm', type=float, default=-1,
                        help='Pixel size in cm; if -1, will read from DICOM')
    parser.add_argument('--sphere_diameter_tol', type=float, nargs=2, default=[0.5, 1.25],
                        help='Tolerance range for measured sphere diameter as a fraction of the expected diameter')

    parser.add_argument('--save_intermediate', type=int, default=1,
                        help='Whether to save intermediate results')
    parser.add_argument('--show_plots', type=int, default=1,
                        help='Whether to show plots during processing')
    parser.add_argument('--gpu_id', type=int, default=0,
                        help='GPU ID to use for NLM denoising')

    parser.add_argument('--crop_z_interp', type=float, default=1.0,
                        help='Interpolation in z if scanner moves too fast')
    parser.add_argument('--crop_num_zeros', type=int, default=25,
                        help='Number of continuous zeros to determine FOV boundary')

    parser.add_argument('--seg_starting_depth', type=int, default=10,
                        help='Starting depth (in pixel) for LSNR segmentation')
    parser.add_argument('--seg_ending_depth', type=int, default=-10,
                        help='Ending depth (in pixel) for LSNR segmentation')
    parser.add_argument('--seg_nlm_sigma_scale', type=float, default=0.125,
                        help='Sigma scale for NLM denoising during segmentation')
    parser.add_argument('--seg_num_depth_groups', type=int, default=10,
                        help='Number of depth groups for adaptive thresholding')
    parser.add_argument('--seg_threshold_factor', type=float, default=2.0,
                        help='Threshold factor for adaptive thresholding')
    parser.add_argument('--seg_mask_num_openings', type=int, default=3,
                        help='Number of openings to denoise the segmentation mask')
    parser.add_argument('--seg_ws_min_distance', type=int, default=5,
                        help='Minimum distance between peaks for watershed segmentation')

    parser.add_argument('--lsnr_patch_size', type=float, default=25,
                        help='> 1: Total depth divided by each patch depth')
    parser.add_argument('--lsnr_step_size', type=float, default=1.0,
                        help='The step size relative to the patch size')

    if 'ipykernel' in sys.argv[0]:
        args = parser.parse_args(default_args)
    else:
        args = parser.parse_args()

    try:
        args.git_hash = subprocess.check_output(['git', 'rev-parse', 'HEAD']).strip().decode('utf-8')
    except Exception:
        args.git_hash = 'N/A'
    args.datetime = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())
    args.sys_argv = sys.argv
    args.script = os.path.abspath(__file__)

    for k in vars(args):
        print(f'{k}: {getattr(args, k)}', flush=True)

    return args


# %%
def main(args):
    input_filename = os.path.join(working_dir, args.input_filename)
    output_dir = os.path.join(working_dir, args.output_dir)
    os.makedirs(output_dir, exist_ok=True)

    if args.save_intermediate:
        intermediate_dir = os.path.join(output_dir, 'intermediate/')
        os.makedirs(intermediate_dir, exist_ok=True)

    print('Extracting volume...', flush=True)
    sitk_src = autoseg.extract_volume(
        input_filename,
        z_interp=args.crop_z_interp,
        num_zeros=args.crop_num_zeros,
    )
    if args.save_intermediate:
        sitk.WriteImage(sitk_src, os.path.join(intermediate_dir, os.path.basename(input_filename) + '.nii.gz'))

    print('Removing starting and ending depths...', flush=True)
    img_src = sitk.GetArrayFromImage(sitk_src)
    img_origin = img_src[:, args.seg_starting_depth:args.seg_ending_depth, :]
    if args.save_intermediate:
        sitk_origin = sitk.GetImageFromArray(img_origin)
        sitk_origin.SetSpacing(sitk_src.GetSpacing())
        sitk.WriteImage(sitk_origin, os.path.join(intermediate_dir, 'origin.nii.gz'))

    print('Denoising with NLM...', flush=True)
    img_nlm = autoseg.non_local_mean_denoising(img_origin, args.seg_nlm_sigma_scale, args.gpu_id)
    if args.save_intermediate:
        sitk_nlm = sitk.GetImageFromArray(img_nlm.astype(np.int16))
        sitk_nlm.SetSpacing(sitk_src.GetSpacing())
        sitk.WriteImage(sitk_nlm, os.path.join(intermediate_dir, 'nlm.nii.gz'))

    print('N4 correction...', flush=True)
    img_n4 = autoseg.n4_correction(img_nlm)
    if args.save_intermediate:
        sitk_n4 = sitk.GetImageFromArray(img_n4.astype(np.int16))
        sitk_n4.SetSpacing(sitk_src.GetSpacing())
        sitk.WriteImage(sitk_n4, os.path.join(intermediate_dir, 'n4.nii.gz'))

    print('Adaptive thresholding...', flush=True)
    mask = autoseg.adaptive_threshold(
        img_n4,
        num_depth_groups=args.seg_num_depth_groups,
        threshold_factor=args.seg_threshold_factor,
        show_plots=args.show_plots,
    )
    if args.save_intermediate:
        sitk_mask = sitk.GetImageFromArray(mask.astype(np.int16))
        sitk_mask.SetSpacing(sitk_src.GetSpacing())
        sitk.WriteImage(sitk_mask, os.path.join(intermediate_dir, 'mask.seg.nrrd'), useCompression=True)

    print('Denoising the mask...', flush=True)
    processed_mask = autoseg.denoise_mask(mask, num_openings=args.seg_mask_num_openings)
    if args.save_intermediate:
        sitk_processed_mask = sitk.GetImageFromArray(processed_mask.astype(np.int16))
        sitk_processed_mask.SetSpacing(sitk_src.GetSpacing())
        sitk.WriteImage(
            sitk_processed_mask, os.path.join(intermediate_dir, 'mask_denoised.seg.nrrd'), useCompression=True
        )

    print('Using WaterShed to separate connected spheres...', flush=True)
    labels, peak_morph_mask = autoseg.split_spheres(processed_mask, peak_min_distance=args.seg_ws_min_distance)
    if args.save_intermediate:
        sitk_labels = sitk.GetImageFromArray(labels.astype(np.int16))
        sitk_labels.SetSpacing(sitk_src.GetSpacing())
        sitk.WriteImage(sitk_labels, os.path.join(intermediate_dir, 'labels.seg.nrrd'), useCompression=True)

        sitk_peak = sitk.GetImageFromArray(peak_morph_mask.astype(np.uint8))
        sitk_peak.SetSpacing(sitk_src.GetSpacing())
        sitk.WriteImage(sitk_peak, os.path.join(intermediate_dir, 'peak_morph_mask.seg.nrrd'), useCompression=True)
    print('Found {} spheres'.format(len(np.unique(labels))), flush=True)

    print('Calculating expected sphere size...', flush=True)
    if args.pixel_size_cm < 0:
        pixel_size_cm = sitk_src.GetSpacing()[0]
    else:
        pixel_size_cm = args.pixel_size_cm
    ref_diameter = args.sphere_diameter_cm / pixel_size_cm
    ref_area = np.pi * (ref_diameter / 2) ** 2
    print('Pixel size: {:.4f} cm, Reference diameter: {:.2f} pixels, Reference area: {:.2f} pixels^2'.format(
        pixel_size_cm, ref_diameter, ref_area
    ), flush=True)

    print('Finding maximum area slices for each label...', flush=True)
    label_img = labels
    label_slices, label_areas = calc_lsnr.find_max_slice_for_labels(label_img)

    print('Removing extreme labels...', flush=True)
    inds, label_areas_small, label_areas_large, label_areas_preserve = calc_lsnr.remove_extreme_labels(
        label_img,
        label_slices,
        label_areas,
        ref_area,
        (args.sphere_diameter_tol[0]**2, args.sphere_diameter_tol[1]**2),
    )
    if args.save_intermediate:
        sitk_small = sitk.GetImageFromArray(label_areas_small.astype(np.int16))
        sitk_small.SetSpacing(sitk_src.GetSpacing())
        sitk.WriteImage(sitk_small, os.path.join(intermediate_dir, 'labels_small.seg.nrrd'), useCompression=True)

        sitk_large = sitk.GetImageFromArray(label_areas_large.astype(np.int16))
        sitk_large.SetSpacing(sitk_src.GetSpacing())
        sitk.WriteImage(sitk_large, os.path.join(intermediate_dir, 'labels_large.seg.nrrd'), useCompression=True)

        sitk_preserve = sitk.GetImageFromArray(label_areas_preserve.astype(np.int16))
        sitk_preserve.SetSpacing(sitk_src.GetSpacing())
        sitk.WriteImage(
            sitk_preserve, os.path.join(intermediate_dir, 'labels_preserve.seg.nrrd'), useCompression=True
        )

    print('Finding the centers and areas of preserved labels...', flush=True)
    df_label, label_img_max = calc_lsnr.find_label_centers(label_img, label_slices, label_areas, inds)
    if args.save_intermediate:
        sitk_label_max = sitk.GetImageFromArray(label_img_max.astype(np.int16))
        sitk_label_max.SetSpacing(sitk_src.GetSpacing())
        sitk.WriteImage(
            sitk_label_max, os.path.join(intermediate_dir, 'label_max.seg.nrrd'), useCompression=True
        )
        df_label.to_csv(os.path.join(intermediate_dir, 'label_info.csv'), index=False)

    print('Calculating LSNR...', flush=True)
    patch_size = int(label_img.shape[1] / args.lsnr_patch_size)
    step_size = int(patch_size * args.lsnr_step_size)
    cnrs, cnr_stds, contrasts, noises, nballs, depths = calc_lsnr.calc_cnr_curve(
        df_label, ref_diameter * 2, img_origin, mask, label_img_max, patch_size, step_size
    )
    # convert the depth to cm and append the offset
    depths = (depths + args.seg_starting_depth) * pixel_size_cm

    print('Saving LSNR results...', flush=True)
    df_res = pd.DataFrame({
        'Depth': depths,
        'CNR': cnrs,
        'CNR_std': cnr_stds,
        'Contrast': contrasts,
        'Noise': noises,
        'Nballs': nballs
    })
    df_res.to_csv(os.path.join(output_dir, 'cnr_curve.csv'), index=False)

    if args.show_plots:
        plt.figure(figsize=[12, 3])
        plt.subplot(131)
        plt.imshow(img_origin[img_origin.shape[0] // 2], cmap='gray')
        plt.title(os.path.basename(input_filename))
        plt.subplot(132)
        plt.errorbar(depths, -cnrs, yerr=cnr_stds / np.sqrt(nballs))
        plt.xlabel('Depth (cm)')
        plt.ylabel('LSNR')
        plt.subplot(133)
        plt.plot(depths, nballs, '.-')
        plt.xlabel('Depth (cm)')
        plt.ylabel('Sphere Count')
        plt.tight_layout()
        if args.save_intermediate:
            plt.savefig(os.path.join(intermediate_dir, 'cnr_curve.png'))
        plt.show()

        plt.figure(figsize=[12, 3])
        plt.subplot(132)
        plt.plot(depths, contrasts, '.-')
        plt.xlabel('Depth (cm)')
        plt.ylabel('Contrast')
        plt.subplot(133)
        plt.plot(depths, noises, '.-')
        plt.xlabel('Depth (cm)')
        plt.ylabel('Noise')
        plt.tight_layout()
        if args.save_intermediate:
            plt.savefig(os.path.join(intermediate_dir, 'contrast_noise_curve.png'))
        plt.show()

    print('Done!', flush=True)

    return df_res


# %%
if __name__ == '__main__':
    args = get_args([
        '--input_filename', 'example.dcm',
        '--output_dir', './output/',
    ])

    res = main(args)
