import pydicom
import cv2
from pathlib import Path
import sys
# if sys.platform == 'darwin':
#     import matplotlib
#     matplotlib.use('Qt5Agg')
# import matplotlib.pyplot as plt
# plt.interactive(False)
import json
import nibabel as nib
import numpy as np



def load_dicom(dcm_path):

    """
    Load x-rays in dicom format

    Args:
        dcm_path: path to the x-ray

    Returns:
        dcm_image: 2D array with the values pixel intensities of the x-ray

    """

    data = pydicom.dcmread(dcm_path)
    dcm_image = data.pixel_array

    # Saturate pixels higher than upper threshold
    try:
        upper_threshold = int(data[0x00281051].value)
        dcm_image[dcm_image > upper_threshold] = upper_threshold
    except:
        pass

    # Invert if MONOCHROME2
    photometricInterpretation = str(data['00280004'].value)
    if photometricInterpretation.find('MONOCHROME2'):
        dcm_image = cv2.bitwise_not(dcm_image)

    dcm_image = (((dcm_image - dcm_image.min()) / (dcm_image.max() - dcm_image.min())) * 255).astype('uint8')
    dcm_image = cv2.cvtColor(dcm_image, cv2.COLOR_GRAY2BGR)

    return dcm_image


def load_nifti(nii_path):

    """

    Load thorax segmentation masks

    Args:
        nii_path: 2D mask with the following labels:
            1 -> right lung
            2 -> left lung
            3 -> Mediastinum

    Returns:

    """

    nib_vol = nib.load(nii_path)
    nii_image = np.array(nib_vol.get_data())[:, :, 0] # Get just first 2 dimensions
    nii_image = np.transpose(nii_image)

    return nii_image


def nifti2rgb(nii_image):

    """

    Convert 3-regional nifto mask to rgb values

    Args:
        nii_image: 3-regional segmentation mask with right lung, left lung and mediastinum

    Returns:
        nii_rgb: 3-regional rgb mask

    """

    nii_rgb = np.zeros((nii_image.shape[0], nii_image.shape[1], 3)).astype("uint8")
    nii_rgb[nii_image == 1] = [255, 0, 0]
    nii_rgb[nii_image == 2] = [0, 255, 0]
    nii_rgb[nii_image == 3] = [0, 0, 255]

    return nii_rgb



if __name__ == "__main__":

    with open("../config/config_training.json") as config_file:
        config_json = json.load(config_file)
    data_path = Path(config_json["paths"]["path_to_data"])
    data_combined_path = Path(config_json["paths"]["path_to_data_combined"])

    ####################################################################################################################
    #                                             TEST FOR LOADING DICOM                                               #
    ####################################################################################################################

    # dcm_path = data_path / "images" / "000.dcm"
    # img = load_dicom(str(dcm_path))
    # plot_img(img)

    ####################################################################################################################
    #                                             TEST FOR LOADING NIFTI                                               #
    ####################################################################################################################

    # dcm_path = data_path / "images" / "000.dcm"
    # img = load_dicom(str(dcm_path))
    # nii_path = data_path / "masks" / "000.nii.gz"
    # mask = load_nifti(str(nii_path))
    # plot_annotation(img, mask)

    ####################################################################################################################
    #                                             TEST FOR COMBINED DATASET                                            #
    ####################################################################################################################

    img = cv2.imread(str(data_combined_path / "images_png" / "training" / "CHNCXR_0001_0.png"))
    mask = cv2.imread(str(data_combined_path / "masks_png" / "training" / "CHNCXR_0001_0.png"))
