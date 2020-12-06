import utils_lib as pyutils
import numpy as np
import pykitti
import cv2
import time
import os

def torgb(img):
    # Load RGB
    rgb_cv = np.asarray(img).copy()
    rgb_cv = cv2.cvtColor(rgb_cv, cv2.COLOR_RGB2BGR)
    rgb_cv = rgb_cv.astype(np.float32)/255
    return rgb_cv

def plot_depth(dmap_raw_np, rgb_cv, indx, name="win"):
    image_name = f"{indx}.png"
    print(f"Saving {image_name}")
    filename_clr = os.path.join(os.getcwd(), "final_images", "colour_images", image_name)
    filename_dm = os.path.join(os.getcwd(), "final_images", "depth_dense_maps", image_name)
    img = rgb_cv
    rgb_cv = rgb_cv.copy()

    dmax = np.max(dmap_raw_np)
    dmin = np.min(dmap_raw_np)
    for r in range(0, dmap_raw_np.shape[0], 1):
        for c in range(0, dmap_raw_np.shape[1], 1):
            depth = dmap_raw_np[r, c]
            if depth > 0.1:
                dcol = depth/20
                rgb_cv[r, c, :] = [1-dcol, dcol, 0]
                #cv2.circle(rgb_cv, (c, r), 1, [1-dcol, dcol, 0], -1)
            else:
                rgb_cv[r, c, :] = [0, 0, 0]

    #cv2.namedWindow(name)
    #cv2.moveWindow(name, 2500, 50)
    
    # Print Upsampled Dense Depth Map
    # cv2.imshow(name, rgb_cv)

    img_normed = 255 * (img - img.min()) / (img.max() - img.min())
    img_normed = np.array(img_normed, np.int)

    rgb_cv_normed = 255 * rgb_cv
    rgb_cv_normed = np.array(rgb_cv_normed, np.int)

    cv2.imwrite(filename_clr, img_normed)
    cv2.imwrite(filename_dm, rgb_cv_normed)
    cv2.waitKey(15)

# Parameters
basedir = "kitti"
date = "2011_09_26"
drive = "0001"
# date = "2011_10_03"
# drive = "0047"

# KITTI Load
p_data = pykitti.raw(basedir, date, drive, frames=None)
for indx in range(len(p_data)):
    print(f"{indx} out of {len(p_data)}")
    M_imu2cam = p_data.calib.T_cam2_imu
    M_velo2cam = p_data.calib.T_cam2_velo
    intr_raw = p_data.calib.K_cam2
    raw_img_size = p_data.get_cam2(indx).size
    img = p_data.get_cam2(indx)

    # Load Velodyne Data
    velodata = p_data.get_velo(indx)  # [N x 4] [We could clean up the low intensity ones here!]
    velodata[:, 3] = 1.

    # Upsampled
    upsampled_img_size = (int(768/2),int(256/2))
    uchange = float(upsampled_img_size[0])/float(raw_img_size[0])
    vchange = float(upsampled_img_size[1])/float(raw_img_size[1])
    intr_upsampled = intr_raw.copy()
    intr_upsampled[0,:] *= uchange
    intr_upsampled[1,:] *= vchange
    intr_upsampled_append = np.append(intr_upsampled, np.array([[0, 0, 0]]).T, axis=1)
    upsampled_img = cv2.resize(torgb(img), upsampled_img_size, interpolation=cv2.INTER_LINEAR)
    upsampled_params = {"filtering": 1, "upsample": 4}
    dmap_upsampled = pyutils.generate_depth(velodata, intr_upsampled_append, M_velo2cam, upsampled_img_size[0], upsampled_img_size[1], upsampled_params)
    plot_depth(dmap_upsampled, upsampled_img, indx, "upsampled_img")

    # Print Upsampled Colour Image
    # cv2.imshow("win", upsampled_img)

# cv2.waitKey(0)
