
import numpy as np

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
