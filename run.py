# TODO fix 1st frame glitch in video

from cardio_volume_skewer import create_skewed_sequences
import numpy as np
import os
output_dir="/home/shahar/cardio_corr/my_packages/cardio_volume_skewer_project/cardio_volume_skewer/outputs/magix_ts_30_test_scale_bug_5"
for r in [ 0.7]:
    for h in [0.85]:#0.8, 0.825]:
        for theta1 in [0.0]:#-7.5, -5.0, -2.5, 0.0, 2.5, 5.0, 7.5]:
            for theta2 in  [45.0]:#[-7.5, -5.0, -2.5, 0.0, 2.5, 5.0, 7.5]:
                if False: # theta1==theta2:
                    print(theta1,theta2,"skip")
                # elif (theta1 in (0.0,-5.0,5.0) and theta2 in (0.0,-5.0,5.0)):
                #     print("skip")
                else:#if f"thetas_{theta1}_{theta2}_rs_{r}_{r}_h_{h}_linear_mask_True_blur_radious_10" not in os.listdir(output_dir):#else:
                    print(r, r, theta1, theta2, h)
                    paths = create_skewed_sequences(
                        r1s_end=r, r2s_end=r, theta1s_end=theta1, theta2s_end=theta2, hs_end=h,
                        output_dir=output_dir, 
                        template_3dimage_path="/home/shahar/cardio_corr/outputs/magix/miccai_experiments/tot_torsion_50_torsion_version_3/dataset_tot_torsion_50_torsion_version_3/28/orig/voxels/xyz_arr_raw.npy",
                        template_mask_path="/home/shahar/cardio_corr/outputs/magix/miccai_experiments/tot_torsion_50_torsion_version_3/dataset_tot_torsion_50_torsion_version_3/28/orig/voxels/xyz_voxels_mask_smooth.npy",
                        template_extra_mask_path="/home/shahar/cardio_corr/outputs/magix/miccai_experiments/tot_torsion_50_torsion_version_3/dataset_tot_torsion_50_torsion_version_3/28/orig/voxels/xyz_voxels_extra_mask_smooth.npy",
                        num_frames=6,
                        zero_outside_mask=True,
                        blur_around_mask_radious=20,
                        theta_distribution_method="linear")
                # else:
                #     print("skip")

                    # img_skewed = np.load(f"/home/shahar/cardio_corr/my_packages/cardio_volume_skewer_project/cardio_volume_skewer/outputs/magix_ts_30/thetas_{theta1}_{theta2}_rs_{r}_{r}_h_{h}_linear_mask_True_blur_radious_10/image_skewed_thetas_{theta1}_{theta2}_rs_{r}_{r}_h_{h}_linear_mask_True_blur_radious_10.npy")
                    # mask_skewed = np.load(f"/home/shahar/cardio_corr/my_packages/cardio_volume_skewer_project/cardio_volume_skewer/outputs/magix_ts_30/thetas_{theta1}_{theta2}_rs_{r}_{r}_h_{h}_linear_mask_True_blur_radious_10/mask_skewed_thetas_{theta1}_{theta2}_rs_{r}_{r}_h_{h}_linear_mask_True_blur_radious_10.npy")
                    # img_orig = np.load(f"/home/shahar/cardio_corr/my_packages/cardio_volume_skewer_project/cardio_volume_skewer/outputs/magix_ts_30/thetas_{theta1}_{theta2}_rs_{r}_{r}_h_{h}_linear_mask_True_blur_radious_10/image_orig_thetas_thetas_{theta1}_{theta2}_rs_{r}_{r}_h_{h}_linear_mask_True_blur_radious_10.npy")
                    # mask_orig = np.load(f"/home/shahar/cardio_corr/my_packages/cardio_volume_skewer_project/cardio_volume_skewer/outputs/magix_ts_30/thetas_{theta1}_{theta2}_rs_{r}_{r}_h_{h}_linear_mask_True_blur_radious_10/mask_orig_thetas_thetas_{theta1}_{theta2}_rs_{r}_{r}_h_{h}_linear_mask_True_blur_radious_10.npy")
                    # img_gt = np.load(f"/home/shahar/cardio_corr/outputs/magix_ts_90/synthetic_dataset102/01/orig/voxels/xyz_arr_raw.npy")
                    # mask_gt = np.load(f"/home/shahar/cardio_corr/outputs/magix_ts_90/synthetic_dataset103/01/orig/voxels/xyz_voxels_mask_smooth.npy")
                    # print(paths)
                    # print("mask sum", (mask_gt^mask_skewed).sum())
                    # print("img sum", (abs(img_gt-img_skewed)).sum())




