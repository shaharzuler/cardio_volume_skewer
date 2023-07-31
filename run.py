# TODO fix opposite flow direction
# TODO fix 1st frame glitch in video

from cardio_volume_skewer import create_skewed_sequences
                                                                                                                                                                                          

paths = create_skewed_sequences(
    r1s_end=0.95, r2s_end=0.95, theta1s_end=90.0, theta2s_end=0.0, hs_end=1.0,
    output_dir="/home/shahar/cardio_corr/my_packages/cardio_volume_skewer_project/cardio_volume_skewer/outputs/self_validation_params_exp_blur", 
    template_3dimage_path="/home/shahar/cardio_corr/outputs/synthetic_dataset20/18/orig/voxels/xyz_arr_raw.npy",
    template_mask_path="/home/shahar/cardio_corr/outputs/synthetic_dataset20/18/orig/voxels/xyz_voxels_mask_smooth.npy",
    num_frames=2,
    zero_outside_mask=True,
    blur_around_mask_radious=5)

print(paths)

## WINNER!!! create_skewed_sequences(r1s_end=0.5,r2s_end=0.5, theta1s_end=40, theta2s_end=-40, hs_end=0.8 ,output_dir="/home/shahar/cardio_corr/my_packages/cardio_volume_skewer_project/cardio_volume_skewer/outputs/self_validation_params_exp", template_timestep=18) ) 

# create_skewed_sequences(r1s_end=0.8, r2s_end=1., theta1s_end=0., theta2s_end=-0., hs_end=1. ,output_dir="/home/shahar/cardio_corr/my_packages/cardio_volume_skewer_project/cardio_volume_skewer/outputs/self_validation_params_exp", template_timestep=18) ) 
# create_skewed_sequences(r1s_end=0.6, r2s_end=1., theta1s_end=0., theta2s_end=-0., hs_end=1. ,output_dir="/home/shahar/cardio_corr/my_packages/cardio_volume_skewer_project/cardio_volume_skewer/outputs/self_validation_params_exp", template_timestep=18) ) 
# create_skewed_sequences(r1s_end=1., r2s_end=0.8, theta1s_end=0., theta2s_end=-0., hs_end=1. ,output_dir="/home/shahar/cardio_corr/my_packages/cardio_volume_skewer_project/cardio_volume_skewer/outputs/self_validation_params_exp", template_timestep=18) ) 
# create_skewed_sequences(r1s_end=1., r2s_end=0.6, theta1s_end=0., theta2s_end=-0., hs_end=1. ,output_dir="/home/shahar/cardio_corr/my_packages/cardio_volume_skewer_project/cardio_volume_skewer/outputs/self_validation_params_exp", template_timestep=18) ) 
# create_skewed_sequences(r1s_end=1., r2s_end=1., theta1s_end=30., theta2s_end=-0., hs_end=1. ,output_dir="/home/shahar/cardio_corr/my_packages/cardio_volume_skewer_project/cardio_volume_skewer/outputs/self_validation_params_exp", template_timestep=18) ) 
# create_skewed_sequences(r1s_end=1., r2s_end=1., theta1s_end=45., theta2s_end=-0., hs_end=1. ,output_dir="/home/shahar/cardio_corr/my_packages/cardio_volume_skewer_project/cardio_volume_skewer/outputs/self_validation_params_exp", template_timestep=18) ) 
# create_skewed_sequences(r1s_end=1., r2s_end=1., theta1s_end=60., theta2s_end=-0., hs_end=1. ,output_dir="/home/shahar/cardio_corr/my_packages/cardio_volume_skewer_project/cardio_volume_skewer/outputs/self_validation_params_exp", template_timestep=18) ) 
# create_skewed_sequences(r1s_end=1., r2s_end=1., theta1s_end=0., theta2s_end=-30., hs_end=1. ,output_dir="/home/shahar/cardio_corr/my_packages/cardio_volume_skewer_project/cardio_volume_skewer/outputs/self_validation_params_exp", template_timestep=18) ) 
# create_skewed_sequences(r1s_end=1., r2s_end=1., theta1s_end=0., theta2s_end=-45., hs_end=1. ,output_dir="/home/shahar/cardio_corr/my_packages/cardio_volume_skewer_project/cardio_volume_skewer/outputs/self_validation_params_exp", template_timestep=18) ) 
# create_skewed_sequences(r1s_end=1., r2s_end=1., theta1s_end=0., theta2s_end=-60., hs_end=1. ,output_dir="/home/shahar/cardio_corr/my_packages/cardio_volume_skewer_project/cardio_volume_skewer/outputs/self_validation_params_exp", template_timestep=18) ) 
# create_skewed_sequences(r1s_end=1., r2s_end=1., theta1s_end=0., theta2s_end=-0., hs_end=0.8 ,output_dir="/home/shahar/cardio_corr/my_packages/cardio_volume_skewer_project/cardio_volume_skewer/outputs/self_validation_params_exp", template_timestep=18) ) 
# create_skewed_sequences(r1s_end=1., r2s_end=1., theta1s_end=0., theta2s_end=-0., hs_end=0.6 ,output_dir="/home/shahar/cardio_corr/my_packages/cardio_volume_skewer_project/cardio_volume_skewer/outputs/self_validation_params_exp", template_timestep=18) ) 



