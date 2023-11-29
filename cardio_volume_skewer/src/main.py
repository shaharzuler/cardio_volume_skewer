import os

import numpy as np

from cardio_volume_skewer import VolumeSkewer, create_video_from_xys_seqs


def _create_frame_sequences_for_video(r1s:np.array, r2s:np.array, theta1s:np.array, theta2s:np.array, hs:np.array, npys_dir:str, image_or_mask:str, theta_distribution_method:str, zero_outside_mask:str, blur_around_mask_radious:str):
    x_seq = []
    y_seq = []
    z_seq = []
    for r1, r2, theta1, theta2, h in zip(r1s, r2s, theta1s, theta2s, hs):
        print( round(theta1, 2), round(theta2, 2), round(r1, 2), round(r2, 2), round(h, 2) )
        suffix = f"_thetas_{round(theta1,2)}_{round(theta2,2)}_rs_{round(r1,2)}_{round(r2,2)}_h_{round(h,2)}_{theta_distribution_method}_mask_{zero_outside_mask}_blur_radious_{blur_around_mask_radious}"
        arr = np.load(os.path.join(npys_dir, f"{image_or_mask}_skewed{suffix}.npy")).astype(float)
        x_seq.append(arr[arr.shape[0]//2,:,:])
        y_seq.append(arr[:,arr.shape[1]//2,:])
        z_seq.append(arr[:,:,arr.shape[2]//2])
    for r1, r2, theta1, theta2, h in zip(r1s[1:-1][::-1], r2s[1:-1][::-1], theta1s[1:-1][::-1], theta2s[1:-1][::-1], hs[1:-1][::-1]):
        print( round(theta1, 2), round(theta2, 2), round(r1, 2), round(r2, 2), round(h, 2) )
        suffix = f"_thetas_{round(theta1,2)}_{round(theta2,2)}_rs_{round(r1,2)}_{round(r2,2)}_h_{round(h,2)}_{theta_distribution_method}_mask_{zero_outside_mask}_blur_radious_{blur_around_mask_radious}"
        arr = np.load(os.path.join(npys_dir, f"{image_or_mask}_skewed{suffix}.npy")).astype(float)
        x_seq.append(arr[arr.shape[0]//2,:,:])
        y_seq.append(arr[:,arr.shape[1]//2,:])
        z_seq.append(arr[:,:,arr.shape[2]//2])
    return x_seq, y_seq, z_seq

def create_skewed_sequences(r1s_end:float, r2s_end:float, theta1s_end:float, theta2s_end:float, hs_end:float, \
    output_dir:str, template_3dimage_path:str, template_mask_path:str, extra_mask_path:str, num_frames:int, \
        zero_outside_mask:bool, blur_around_mask_radious:int, theta_distribution_method:str):

    r1s_start = 1.0
    r2s_start = 1.0
    theta1s_start = 0.
    theta2s_start = 0.
    hs_start = 1.0
    ratio_start = 0.
    
    r1s     = np.linspace( r1s_start,     r1s_end,     num_frames)
    r2s     = np.linspace( r2s_start,     r2s_end,     num_frames)
    theta1s = np.linspace( theta1s_start, theta1s_end, num_frames)
    theta2s = np.linspace( theta2s_start, theta2s_end, num_frames)
    hs      = np.linspace( hs_start,      hs_end,      num_frames)
    ratios  = np.linspace( ratio_start,   1.,          num_frames)

    output_subdir = f"thetas_{round(theta1s_end,2)}_{round(theta2s_end,2)}_rs_{round(r1s_end,2)}_{round(r2s_end,2)}_h_{round(hs_end,2)}_{theta_distribution_method}_mask_{zero_outside_mask}_blur_radious_{blur_around_mask_radious}"
    output_dir = os.path.join(output_dir, output_subdir)

    three_d_image = np.load(template_3dimage_path)
    binary_mask = np.load(template_mask_path)
    extra_binary_mask = np.load(extra_mask_path)
    volume_skewer = VolumeSkewer(
        save_nrrd=True, zero_outside_mask=zero_outside_mask, blur_around_mask_radious=blur_around_mask_radious, \
            warping_borders_pad='zeros', image_warping_interp_mode='bilinear', mask_warping_interp_mode='nearest', \
                theta_changing_method=theta_distribution_method)

    volume_skewer.skew_volume(
            theta1=theta1s_end, theta2=theta2s_end, 
            r1=r1s_end, r2=r2s_end, 
            h=hs_end, 
            three_d_image=three_d_image, 
            three_d_binary_mask=binary_mask,
            extra_three_d_binary_mask=extra_binary_mask,
            output_dir=output_dir
            )
    maximal_flow = volume_skewer.flow_for_mask

    for n, (ratio, r1, r2, theta1, theta2, h) in enumerate(zip(ratios, r1s, r2s, theta1s, theta2s, hs)):
        volume_skewer.scaled_flow_for_mask = maximal_flow * ratio
        if n  == 0:
            volume_skewer.skewed_three_d_image = volume_skewer.three_d_image
            volume_skewer.skewed_three_d_binary_mask = volume_skewer.three_d_binary_mask
            volume_skewer.skewed_extra_three_d_binary_mask = volume_skewer.extra_three_d_binary_mask
            volume_skewer.scaled_flow_for_image = volume_skewer.scaled_flow_for_mask

        else:
            volume_skewer.skewed_three_d_binary_mask = volume_skewer.flow_warp(image=volume_skewer.three_d_binary_mask.astype(float), flow=volume_skewer.scaled_flow_for_mask,  warping_borders_pad=volume_skewer.warping_borders_pad, warping_interp_mode=volume_skewer.mask_warping_interp_mode)
            volume_skewer.skewed_extra_three_d_binary_mask = volume_skewer.flow_warp(image=volume_skewer.extra_three_d_binary_mask.astype(float), flow=volume_skewer.scaled_flow_for_mask,  warping_borders_pad=volume_skewer.warping_borders_pad, warping_interp_mode=volume_skewer.mask_warping_interp_mode)
            if volume_skewer.zero_outside_mask: # only moves pixels inside the seg mask 
                volume_skewer.scaled_flow_for_image = volume_skewer.handle_outside_mask()
            else:
                volume_skewer.scaled_flow_for_image = volume_skewer.scaled_flow_for_mask
            volume_skewer.skewed_three_d_image = volume_skewer.flow_warp(image=volume_skewer.three_d_image, flow=volume_skewer.scaled_flow_for_image, warping_borders_pad=volume_skewer.warping_borders_pad, warping_interp_mode=volume_skewer.image_warping_interp_mode )
        
        suffix = f"_thetas_{round(theta1,2)}_{round(theta2,2)}_rs_{round(r1,2)}_{round(r2,2)}_h_{round(h,2)}_{theta_distribution_method}_mask_{zero_outside_mask}_blur_radious_{blur_around_mask_radious}"

        if volume_skewer.save_nrrd:
            volume_skewer.save_nrrds(suffix=suffix)
        if volume_skewer.save_npy:
            volume_skewer.save_npys(suffix=suffix)

    print("skewing completed")

    x_seq, y_seq, z_seq = _create_frame_sequences_for_video(r1s, r2s, theta1s, theta2s, hs, output_dir, "image", theta_distribution_method, zero_outside_mask, blur_around_mask_radious)
    create_video_from_xys_seqs(x_seq*10, y_seq*10, z_seq*10, os.path.join(output_dir,f"vid_{output_subdir}.avi"))

    x_seq_mask, y_seq_mask, z_seq_mask = _create_frame_sequences_for_video(r1s, r2s, theta1s, theta2s, hs, output_dir, "mask", theta_distribution_method, zero_outside_mask, blur_around_mask_radious)
    create_video_from_xys_seqs(x_seq_mask*10, y_seq_mask*10, z_seq_mask*10, os.path.join(output_dir,f"vid_mask_{output_subdir}.avi"))

    x_seq_extra_mask, y_seq_extra_mask, z_seq_extra_mask = _create_frame_sequences_for_video(r1s, r2s, theta1s, theta2s, hs, output_dir, "extra_mask", theta_distribution_method, zero_outside_mask, blur_around_mask_radious)
    create_video_from_xys_seqs(x_seq_extra_mask*10, y_seq_extra_mask*10, z_seq_extra_mask*10, os.path.join(output_dir,f"vid_extra_mask_{output_subdir}.avi"))

    template_synthetic_image_path = os.path.join(output_dir, f"image_orig_{output_subdir}.npy")
    unlabeled_synthetic_image_path = os.path.join(output_dir, f"image_skewed_{output_subdir}.npy")
    template_synthetic_mask_path = os.path.join(output_dir, f"mask_orig_{output_subdir}.npy")
    unlabeled_synthetic_mask_path = os.path.join(output_dir, f"mask_skewed_{output_subdir}.npy")
    template_synthetic_extra_mask_path = os.path.join(output_dir, f"extra_mask_orig_{output_subdir}.npy")
    unlabeled_synthetic_extra_mask_path = os.path.join(output_dir, f"extra_mask_skewed_{output_subdir}.npy")
    synthetic_flow_path = os.path.join(output_dir, f"flow_for_image_{output_subdir}.npy")

    return template_synthetic_image_path, unlabeled_synthetic_image_path, template_synthetic_mask_path, unlabeled_synthetic_mask_path, template_synthetic_extra_mask_path, unlabeled_synthetic_extra_mask_path, synthetic_flow_path
