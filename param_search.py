import numpy as np
from matplotlib import pyplot as plt
import matplotlib as mpl

# img_skewed = np.load("/home/shahar/cardio_corr/my_packages/cardio_volume_skewer_project/cardio_volume_skewer/outputs/magix_ts_30/thetas_0.0_0.0_rs_0.6_0.6_h_0.85_linear_mask_True_blur_radious_10/image_skewed_thetas_0.0_0.0_rs_0.6_0.6_h_0.85_linear_mask_True_blur_radious_10.npy")
# mask_skewed = np.load("/home/shahar/cardio_corr/my_packages/cardio_volume_skewer_project/cardio_volume_skewer/outputs/magix_ts_30/thetas_0.0_0.0_rs_0.6_0.6_h_0.85_linear_mask_True_blur_radious_10/mask_skewed_thetas_0.0_0.0_rs_0.6_0.6_h_0.85_linear_mask_True_blur_radious_10.npy")
# img_orig = np.load("/home/shahar/cardio_corr/my_packages/cardio_volume_skewer_project/cardio_volume_skewer/outputs/magix_ts_30/thetas_0.0_0.0_rs_0.6_0.6_h_0.85_linear_mask_True_blur_radious_10/image_orig_thetas_0.0_0.0_rs_0.6_0.6_h_0.85_linear_mask_True_blur_radious_10.npy")
# mask_orig = np.load("/home/shahar/cardio_corr/my_packages/cardio_volume_skewer_project/cardio_volume_skewer/outputs/magix_ts_30/thetas_0.0_0.0_rs_0.6_0.6_h_0.85_linear_mask_True_blur_radious_10/mask_orig_thetas_0.0_0.0_rs_0.6_0.6_h_0.85_linear_mask_True_blur_radious_10.npy")
img_gt = np.load("/home/shahar/data/cardiac_3d_data_magix/90/orig/voxels/xyz_arr_raw.npy")
mask_gt = np.load("/home/shahar/data/cardiac_3d_data_magix/90/orig/voxels/xyz_voxels_mask_smooth.npy")
import os
top_d = "/home/shahar/cardio_corr/my_packages/cardio_volume_skewer_project/cardio_volume_skewer/outputs/magix_ts_30"
min_sum_mask = 1e10
min_sum_img = 1e10
min_sum_mask_path = ""
min_sum_img_path = ""
for d in os.listdir(top_d):
    img_skewed = np.load(f"{top_d}/{d}/image_skewed_{d}.npy")
    mask_skewed = np.load(f"{top_d}/{d}/mask_skewed_{d}.npy")
    img_orig = np.load(f"{top_d}/{d}/image_skewed_thetas_0.0_0.0_rs_1.0_1.0_h_1.0_linear_mask_True_blur_radious_10.npy")
    mask_orig = np.load(f"{top_d}/{d}/mask_skewed_thetas_0.0_0.0_rs_1.0_1.0_h_1.0_linear_mask_True_blur_radious_10.npy")

    print(d)
    if True:
        plt.imshow(img_orig[:,:,114],  cmap=mpl.colormaps["bone"])
        plt.savefig("org1.jpg")
        plt.imshow(img_skewed[:,:,114], cmap=mpl.colormaps["bone"])
        plt.savefig("skw1.jpg")
        plt.imshow(img_orig[:,103,:],  cmap=mpl.colormaps["bone"])
        plt.savefig("org2.jpg")
        plt.imshow(img_skewed[:,103,:], cmap=mpl.colormaps["bone"])
        plt.savefig("skw2.jpg")
        plt.imshow(img_orig[103,:,:],  cmap=mpl.colormaps["bone"])
        plt.savefig("org3.jpg")
        plt.imshow(img_skewed[103,:,:], cmap=mpl.colormaps["bone"])
        plt.savefig("skw3.jpg")
    sum_mask = (mask_gt^mask_skewed).sum()
    print("mask sum", sum_mask)
    if sum_mask<min_sum_mask:
        min_sum_mask = sum_mask
        min_sum_mask_path = d
    
    sum_img =  (abs(img_gt-img_skewed)).sum()
    print("img sum", sum_img)
    if sum_img<min_sum_img:
        min_sum_img = sum_img
        min_sum_img_path = d

    print("FINAL MIN VALS:")
    print(min_sum_mask, min_sum_mask_path)
    print(min_sum_img, min_sum_img_path)
 




# plt.imshow((mask_orig)[:,:,160])
# plt.savefig("mask_orig.jpg")
# plt.imshow((mask_skewed)[:,:,160])
# plt.savefig("mask_skewed.jpg")
# plt.imshow((mask_gt)[:,:,160])
# plt.savefig("mask_gt.jpg")
# plt.imshow((mask_gt^mask_skewed)[:,:,160])
# plt.savefig("xor_mask.jpg")

# plt.imshow((img_orig)[:,:,160])
# plt.savefig("img_orig.jpg")
# plt.imshow((img_skewed)[:,:,160])
# plt.savefig("img_skewed.jpg")
# plt.imshow((img_gt)[:,:,160])
# plt.savefig("img_gt.jpg")
# plt.imshow((abs(img_gt-img_skewed))[:,:,160])
# plt.savefig("diff_img.jpg")


print(1)



