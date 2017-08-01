from haonet import HaoNet
from utils import random_crop
import numpy as np
import tensorflow as tf

list_of_imgs_path = [
    "D:\\BreaKHis_v1\\histology_slides\\breast\\malignant\\SOB\\papillary_carcinoma\\SOB_M_PC_14-15704\\100X\\SOB_M_PC-14-15704-100-002.png",
    "D:\\BreaKHis_v1\\histology_slides\\breast\\malignant\\SOB\\papillary_carcinoma\\SOB_M_PC_14-15704\\100X\\SOB_M_PC-14-15704-100-014.png",
    "D:\\BreaKHis_v1\\histology_slides\\breast\\malignant\\SOB\\ductal_carcinoma\\SOB_M_DC_14-11951\\100X\\SOB_M_DC-14-11951-100-002.png",
    "D:\\BreaKHis_v1\\histology_slides\\breast\\malignant\\SOB\\ductal_carcinoma\\SOB_M_DC_14-11951\\100X\\SOB_M_DC-14-11951-100-010.png",
    "D:\\BreaKHis_v1\\histology_slides\\breast\\malignant\\SOB\\lobular_carcinoma\\SOB_M_LC_14-13412\\100X\\SOB_M_LC-14-13412-100-030.png",
    "D:\\BreaKHis_v1\\histology_slides\\breast\\malignant\\SOB\\ductal_carcinoma\\SOB_M_DC_14-17614\\100X\\SOB_M_DC-14-17614-100-023.png",
    "D:\\BreaKHis_v1\\histology_slides\\breast\\benign\\SOB\\adenosis\\SOB_B_A_14-29960CD\\100X\\SOB_B_A-14-29960CD-100-009.png",
    "D:\\BreaKHis_v1\\histology_slides\\breast\\malignant\\SOB\\lobular_carcinoma\\SOB_M_LC_14-13412\\100X\\SOB_M_LC-14-13412-100-032.png",
    "D:\\BreaKHis_v1\\histology_slides\\breast\\malignant\\SOB\\papillary_carcinoma\\SOB_M_PC_14-15704\\100X\\SOB_M_PC-14-15704-100-028.png",
    "D:\\BreaKHis_v1\\histology_slides\\breast\\benign\\SOB\\fibroadenoma\\SOB_B_F_14-9133\\100X\\SOB_B_F-14-9133-100-016.png",
    "D:\\BreaKHis_v1\\histology_slides\\breast\\malignant\\SOB\\lobular_carcinoma\\SOB_M_LC_14-15570\\100X\\SOB_M_LC-14-15570-100-019.png",
    "D:\\BreaKHis_v1\\histology_slides\\breast\\benign\\SOB\\tubular_adenoma\\SOB_B_TA_14-3411F\\100X\\SOB_B_TA-14-3411F-100-014.png",
    "D:\\BreaKHis_v1\\histology_slides\\breast\\malignant\\SOB\\lobular_carcinoma\\SOB_M_LC_14-15570\\100X\\SOB_M_LC-14-15570-100-010.png",
    "D:\\BreaKHis_v1\\histology_slides\\breast\\malignant\\SOB\\ductal_carcinoma\\SOB_M_DC_14-17915\\100X\\SOB_M_DC-14-17915-100-003.png",
    "D:\\BreaKHis_v1\\histology_slides\\breast\\malignant\\SOB\\ductal_carcinoma\\SOB_M_DC_14-20636\\100X\\SOB_M_DC-14-20636-100-026.png",
    "D:\\BreaKHis_v1\\histology_slides\\breast\\malignant\\SOB\\lobular_carcinoma\\SOB_M_LC_14-15570\\100X\\SOB_M_LC-14-15570-100-040.png",
    "D:\\BreaKHis_v1\\histology_slides\\breast\\malignant\\SOB\\ductal_carcinoma\\SOB_M_DC_14-17614\\100X\\SOB_M_DC-14-17614-100-022.png",
    "D:\\BreaKHis_v1\\histology_slides\\breast\\malignant\\SOB\\papillary_carcinoma\\SOB_M_PC_14-15687B\\100X\\SOB_M_PC-14-15687B-100-005.png",
    "D:\\BreaKHis_v1\\histology_slides\\breast\\malignant\\SOB\\lobular_carcinoma\\SOB_M_LC_14-13412\\100X\\SOB_M_LC-14-13412-100-024.png",
    "D:\\BreaKHis_v1\\histology_slides\\breast\\benign\\SOB\\phyllodes_tumor\\SOB_B_PT_14-21998AB\\100X\\SOB_B_PT-14-21998AB-100-065.png",
    "D:\\BreaKHis_v1\\histology_slides\\breast\\benign\\SOB\\phyllodes_tumor\\SOB_B_PT_14-21998AB\\100X\\SOB_B_PT-14-21998AB-100-038.png",
    "D:\\BreaKHis_v1\\histology_slides\\breast\\benign\\SOB\\fibroadenoma\\SOB_B_F_14-9133\\100X\\SOB_B_F-14-9133-100-013.png",
    "D:\\BreaKHis_v1\\histology_slides\\breast\\malignant\\SOB\\papillary_carcinoma\\SOB_M_PC_14-15704\\100X\\SOB_M_PC-14-15704-100-004.png",
    "D:\\BreaKHis_v1\\histology_slides\\breast\\malignant\\SOB\\mucinous_carcinoma\\SOB_M_MC_14-18842D\\100X\\SOB_M_MC-14-18842D-100-008.png",
    "D:\\BreaKHis_v1\\histology_slides\\breast\\malignant\\SOB\\mucinous_carcinoma\\SOB_M_MC_14-18842D\\100X\\SOB_M_MC-14-18842D-100-003.png",
    "D:\\BreaKHis_v1\\histology_slides\\breast\\benign\\SOB\\phyllodes_tumor\\SOB_B_PT_14-21998AB\\100X\\SOB_B_PT-14-21998AB-100-034.png",
    "D:\\BreaKHis_v1\\histology_slides\\breast\\malignant\\SOB\\papillary_carcinoma\\SOB_M_PC_14-15704\\100X\\SOB_M_PC-14-15704-100-016.png",
    "D:\\BreaKHis_v1\\histology_slides\\breast\\malignant\\SOB\\papillary_carcinoma\\SOB_M_PC_14-15704\\100X\\SOB_M_PC-14-15704-100-017.png",
    "D:\\BreaKHis_v1\\histology_slides\\breast\\malignant\\SOB\\ductal_carcinoma\\SOB_M_DC_14-20636\\100X\\SOB_M_DC-14-20636-100-019.png",
    "D:\\BreaKHis_v1\\histology_slides\\breast\\malignant\\SOB\\ductal_carcinoma\\SOB_M_DC_14-11951\\100X\\SOB_M_DC-14-11951-100-011.png",
    "D:\\BreaKHis_v1\\histology_slides\\breast\\benign\\SOB\\tubular_adenoma\\SOB_B_TA_14-3411F\\100X\\SOB_B_TA-14-3411F-100-012.png",
    "D:\\BreaKHis_v1\\histology_slides\\breast\\benign\\SOB\\phyllodes_tumor\\SOB_B_PT_14-21998AB\\100X\\SOB_B_PT-14-21998AB-100-027.png",
    "D:\\BreaKHis_v1\\histology_slides\\breast\\malignant\\SOB\\ductal_carcinoma\\SOB_M_DC_14-20636\\100X\\SOB_M_DC-14-20636-100-015.png",
    "D:\\BreaKHis_v1\\histology_slides\\breast\\malignant\\SOB\\ductal_carcinoma\\SOB_M_DC_14-11951\\100X\\SOB_M_DC-14-11951-100-005.png",
    "D:\\BreaKHis_v1\\histology_slides\\breast\\benign\\SOB\\fibroadenoma\\SOB_B_F_14-9133\\100X\\SOB_B_F-14-9133-100-031.png",
    "D:\\BreaKHis_v1\\histology_slides\\breast\\benign\\SOB\\fibroadenoma\\SOB_B_F_14-9133\\100X\\SOB_B_F-14-9133-100-036.png",
    "D:\\BreaKHis_v1\\histology_slides\\breast\\malignant\\SOB\\lobular_carcinoma\\SOB_M_LC_14-13412\\100X\\SOB_M_LC-14-13412-100-029.png",
    "D:\\BreaKHis_v1\\histology_slides\\breast\\benign\\SOB\\fibroadenoma\\SOB_B_F_14-9133\\100X\\SOB_B_F-14-9133-100-020.png",
    "D:\\BreaKHis_v1\\histology_slides\\breast\\malignant\\SOB\\mucinous_carcinoma\\SOB_M_MC_14-19979C\\100X\\SOB_M_MC-14-19979C-100-011.png",
    "D:\\BreaKHis_v1\\histology_slides\\breast\\benign\\SOB\\phyllodes_tumor\\SOB_B_PT_14-21998AB\\100X\\SOB_B_PT-14-21998AB-100-009.png"]


def run(params_path, list_of_imgs_path, y_trues=None):
    haonet = HaoNet(dataset=None, params=params_path)
    results = []
    testset = []
    for img_path in list_of_imgs_path:
        imgs = random_crop(path=img_path, patch_size=64, num_of_imgs=100, do_rotate=True, do_mirror=True, sub_mean=True)
        print(imgs[0].shape)
        testset.append(np.array(imgs))

    testset = np.array(testset)
    print(testset.shape)

    session = tf.Session()
    session.run(tf.global_variables_initializer())

    for i in range(testset.shape[0]):
        x = testset[i]
        y = []
        for j in range(testset.shape[1]):
            y.append([0,0])
        y = np.array(y)

        feed_dict = {haonet.x_image: x,
                           haonet.y_true: y}

        y_pre_cls = session.run(haonet.y_pred_cls, feed_dict=feed_dict)
        f = float(sum(y_pre_cls)) / len(y_pre_cls)

        if f > 0.5:
            y_pre_cls = 1
        else:
            y_pre_cls = 0
        results.append(y_pre_cls)

    return results

if __name__ == "__main__":
    results= run("C:\\Users\Hao\Projects\AI_Projects\project\saved_weights\params5.npy",list_of_imgs_path)
    print(len(results))