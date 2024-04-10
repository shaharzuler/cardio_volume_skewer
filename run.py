# TODO fix 1st frame glitch in video

from cardio_volume_skewer import create_skewed_sequences


output_dir="/home/shahar/cardio_corr/my_packages/cardio_volume_skewer_project/cardio_volume_skewer/outputs/"

for r in [ 0.7]:
    for h in [0.85, 0.8, 0.825]:
        for theta1 in [0.0, -7.5, -5.0, -2.5, 0.0, 2.5, 5.0, 7.5]:
            for theta2 in  [45.0, -7.5, -5.0, -2.5, 0.0, 2.5, 5.0, 7.5]:                
                print(r, theta1, theta2, h)
                paths = create_skewed_sequences(
                    r1s_end=r, r2s_end=r, theta1s_end=theta1, theta2s_end=theta2, hs_end=h,
                    output_dir=output_dir, 
                    template_3dimage_path="/home/shahar/home/shahar/projects/CardioSpecrum_inference_project/CardioSpectrum_inference/sample_scan/output/magix_sample_dataset/28/orig/voxels/xyz_arr_raw.npy",
                    template_mask_path="/home/shahar/home/shahar/projects/CardioSpecrum_inference_project/CardioSpectrum_inference/sample_scan/output/magix_sample_dataset/28/orig/voxels/xyz_voxels_mask_smooth.npy",
                    template_extra_mask_path="/home/shahar/home/shahar/projects/CardioSpecrum_inference_project/CardioSpectrum_inference/sample_scan/output/magix_sample_dataset/28/orig/voxels/xyz_voxels_extra_mask_smooth.npy",
                    num_frames=6,
                    zero_outside_mask=True,
                    blur_around_mask_radious=20,
                    theta_distribution_method="linear",
                    scale_down_by=4)





