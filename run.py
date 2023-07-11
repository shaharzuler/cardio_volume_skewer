# TODO fix opposite flow direction
# TODO automatically copy orig image to be the first image
# TODO add pad type as a param

import numpy as np
import nrrd
import os

from cardio_volume_skewer import VolumeSkewer, create_video_from_xys_seqs


def create_toy_box_image_and_mask():
    binary_mask = np.zeros((100,100,100), dtype=bool)
    binary_mask[40:60, 40:60, 10:90] = True
    binary_mask[30:50, 30:50, 10:40] = True
    binary_mask[50:70, 50:70, 40:60] = True
    binary_mask[70:90, 70:90, 60:90] = True
    binary_mask[20:81,40:61, 40:61] = True


    three_d_image = np.arange(binary_mask.flatten().shape[0]).astype(float)
    three_d_image = three_d_image.reshape(binary_mask.shape)
    three_d_image = binary_mask.copy().astype(float)

def read_ct_and_mask(timestep, save_nrrd=False):
    img3d_path = f"/home/shahar/data/cardiac_3d_data/{timestep}/orig/voxels/xyz_arr_raw.npy"
    mask_path = f"/home/shahar/data/cardiac_3d_data/{timestep}/orig/voxels/xyz_voxels_mask_smooth.npy"
    img3d = np.load(img3d_path)
    mask = np.load(mask_path)
    print(img3d.shape, mask.shape)

    if save_nrrd:
        np.save("ct_scan", img3d)
        np.save("binary_mask", mask)

    return img3d, mask

def create_frame_sequences_for_video(r1s, r2s, theta1s, theta2s, hs, nrrds_dir):
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

def calc_start(orig_param_start, end, num_frames):
    gap = (end - orig_param_start)/num_frames
    return orig_param_start + gap


def create_skewed_sequences(r1s_end, r2s_end, theta1s_end, theta2s_end, hs_end):
    timestep = 18 if r1s_end<=1 else 28

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

    major_output_dir = "/home/shahar/cardio_corr/my_packages/cardio_volume_skewer_project/cardio_volume_skewer/outputs/self_validation_params_exp"
    minor_output_dir = f"_thetas_{round(theta1s_end,2)}_{round(theta2s_end,2)}_rs_{round(r1s_end,2)}_{round(r2s_end,2)}_h_{round(hs_end,2)}"
    output_dir = os.path.join(major_output_dir, minor_output_dir)

    SKEW = True


    if SKEW:
        three_d_image, binary_mask = read_ct_and_mask(timestep=timestep)
        volume_skewer =  VolumeSkewer(warping_borders_pad='zeros', warping_interp_mode='bilinear')
        for r1, r2, theta1, theta2, h in zip(r1s, r2s, theta1s, theta2s, hs):
            print( round(theta1, 2), round(theta2, 2), round(r1, 2), round(r2, 2), round(h, 2) )
            skewed_3d_image = volume_skewer.skew_volume(
                theta1=theta1, theta2=theta2, 
                r1=r1, r2=r2, 
                h=h, 
                three_d_image=three_d_image, 
                three_d_binary_mask=binary_mask,
                output_dir=output_dir
                )
        print("skewing completed")

    x_seq, y_seq, z_seq = create_frame_sequences_for_video(r1s, r2s, theta1s, theta2s, hs, output_dir)

    create_video_from_xys_seqs(x_seq*10, y_seq*10, z_seq*10, os.path.join(output_dir,f"vid{minor_output_dir}.avi"))


## WINNER!!! create_skewed_sequences(r1s_end=0.5,r2s_end=0.5, theta1s_end=40, theta2s_end=-40, hs_end=0.8 ) 

create_skewed_sequences(r1s_end=0.5,r2s_end=0.5, theta1s_end=40, theta2s_end=-40, hs_end=0.8 ) 

# create_skewed_sequences(r1s_end=0.8, r2s_end=1., theta1s_end=0., theta2s_end=-0., hs_end=1.) 
# create_skewed_sequences(r1s_end=0.6, r2s_end=1., theta1s_end=0., theta2s_end=-0., hs_end=1.) 
# create_skewed_sequences(r1s_end=1., r2s_end=0.8, theta1s_end=0., theta2s_end=-0., hs_end=1.) 
# create_skewed_sequences(r1s_end=1., r2s_end=0.6, theta1s_end=0., theta2s_end=-0., hs_end=1.) 
# create_skewed_sequences(r1s_end=1., r2s_end=1., theta1s_end=30., theta2s_end=-0., hs_end=1.) 
# create_skewed_sequences(r1s_end=1., r2s_end=1., theta1s_end=45., theta2s_end=-0., hs_end=1.) 
# create_skewed_sequences(r1s_end=1., r2s_end=1., theta1s_end=60., theta2s_end=-0., hs_end=1.) 
# create_skewed_sequences(r1s_end=1., r2s_end=1., theta1s_end=0., theta2s_end=-30., hs_end=1.) 
# create_skewed_sequences(r1s_end=1., r2s_end=1., theta1s_end=0., theta2s_end=-45., hs_end=1.) 
# create_skewed_sequences(r1s_end=1., r2s_end=1., theta1s_end=0., theta2s_end=-60., hs_end=1.) 
# create_skewed_sequences(r1s_end=1., r2s_end=1., theta1s_end=0., theta2s_end=-0., hs_end=0.8) 
# create_skewed_sequences(r1s_end=1., r2s_end=1., theta1s_end=0., theta2s_end=-0., hs_end=0.6) 



