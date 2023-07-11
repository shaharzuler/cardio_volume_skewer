# TODO fix opposite flow direction
# TODO automatically copy orig image to be the first image

import numpy as np
import nrrd
import os

from cardio_volume_skewer import VolumeSkewer, create_video_from_xys_seqs


def _read_ct_and_mask(timestep, save_nrrd=False):
    img3d_path = f"/home/shahar/data/cardiac_3d_data/{timestep}/orig/voxels/xyz_arr_raw.npy"
    mask_path = f"/home/shahar/data/cardiac_3d_data/{timestep}/orig/voxels/xyz_voxels_mask_smooth.npy"
    img3d = np.load(img3d_path)
    mask = np.load(mask_path)
    print(img3d.shape, mask.shape)

    if save_nrrd:
        np.save("ct_scan", img3d)
        np.save("binary_mask", mask)

    return img3d, mask

def _create_frame_sequences_for_video(r1s, r2s, theta1s, theta2s, hs, nrrds_dir):
    x_seq = []
    y_seq = []
    z_seq = []
    for r1, r2, theta1, theta2, h in zip(r1s, r2s, theta1s, theta2s, hs):
        print( round(theta1, 2), round(theta2, 2), round(r1, 2), round(r2, 2), round(h, 2) )
        suffix = f"_thetas_{round(theta1,2)}_{round(theta2,2)}_rs_{round(r1,2)}_{round(r2,2)}_h_{round(h,2)}"
        nrrd_arr = nrrd.read(os.path.join(nrrds_dir, f"img_skewed{suffix}.nrrd"))[0]
        x_seq.append(nrrd_arr[nrrd_arr.shape[0]//2,:,:])
        y_seq.append(nrrd_arr[:,nrrd_arr.shape[1]//2,:])
        z_seq.append(nrrd_arr[:,:,nrrd_arr.shape[2]//2])
    for r1, r2, theta1, theta2, h in zip(r1s[1:-1][::-1], r2s[1:-1][::-1], theta1s[1:-1][::-1], theta2s[1:-1][::-1], hs[1:-1][::-1]):
        print( round(theta1, 2), round(theta2, 2), round(r1, 2), round(r2, 2), round(h, 2) )
        suffix = f"_thetas_{round(theta1,2)}_{round(theta2,2)}_rs_{round(r1,2)}_{round(r2,2)}_h_{round(h,2)}"
        nrrd_arr = nrrd.read(os.path.join(nrrds_dir, f"img_skewed{suffix}.nrrd"))[0]
        x_seq.append(nrrd_arr[nrrd_arr.shape[0]//2,:,:])
        y_seq.append(nrrd_arr[:,nrrd_arr.shape[1]//2,:])
        z_seq.append(nrrd_arr[:,:,nrrd_arr.shape[2]//2])
    return x_seq, y_seq, z_seq

# def calc_start(orig_param_start, end, num_frames):
#     gap = (end - orig_param_start)/num_frames
#     return orig_param_start + gap

def create_skewed_sequences(r1s_end, r2s_end, theta1s_end, theta2s_end, hs_end, output_dir, template_timestep=None):
    if template_timestep is None:
        template_timestep = 18 if r1s_end<=1 else 28

    num_frames = 5

    r1s_start = 1.0
    r2s_start = 1.0
    theta1s_start = 0
    theta2s_start = 0
    hs_start = 1.0
    
    r1s     = np.linspace( r1s_start,     r1s_end,     num_frames + 1 )
    r2s     = np.linspace( r2s_start,     r2s_end,     num_frames + 1 )
    theta1s = np.linspace( theta1s_start, theta1s_end, num_frames + 1 )
    theta2s = np.linspace( theta2s_start, theta2s_end, num_frames + 1 )
    hs      = np.linspace( hs_start,      hs_end,      num_frames + 1 )

    
    output_subdir = f"_thetas_{round(theta1s_end,2)}_{round(theta2s_end,2)}_rs_{round(r1s_end,2)}_{round(r2s_end,2)}_h_{round(hs_end,2)}"
    output_dir = os.path.join(output_dir, output_subdir)


    three_d_image, binary_mask = _read_ct_and_mask(timestep=template_timestep)
    volume_skewer =  VolumeSkewer(warping_borders_pad='zeros', img_warping_interp_mode='bilinear', mask_warping_interp_mode='nearest')
    for r1, r2, theta1, theta2, h in zip(r1s, r2s, theta1s, theta2s, hs):
        print( round(theta1, 2), round(theta2, 2), round(r1, 2), round(r2, 2), round(h, 2) )
        volume_skewer.skew_volume(
            theta1=theta1, theta2=theta2, 
            r1=r1, r2=r2, 
            h=h, 
            three_d_image=three_d_image, 
            three_d_binary_mask=binary_mask,
            output_dir=output_dir
            )
    print("skewing completed")

    x_seq, y_seq, z_seq = _create_frame_sequences_for_video(r1s, r2s, theta1s, theta2s, hs, output_dir)

    create_video_from_xys_seqs(x_seq*10, y_seq*10, z_seq*10, os.path.join(output_dir,f"vid{output_subdir}.avi"))
