from pathlib import Path
import json
import cv2
import numpy as np
import os
import io
# Import custom libraries
import models.neural_networks as net
import load_data as load
import data_generator_custom as datagen
from MaskRCNN_config import LungsConfig
from models.Mask_RCNN import model as modellib
import nibabel as nib
import pydicom
import nrrd
import matplotlib.pyplot as plt
import SimpleITK as sitk

def remove_noisy_labels(mask):

    """

    Remove noisy incorrectly labeled pixels from segmentation mask

    Args:
        mask: uint8 255-hot encoded segmentation mask

    Returns:
        mask_clean: 255-hot encoded segmentation mask without noisy labels

    """

    clean_mask = np.zeros(mask.shape)
    kernel = np.ones((5, 5), np.uint8)
    for label in range(mask.shape[2]):
        clean_mask[:, :, label] = cv2.morphologyEx(mask[:, :, label], cv2.MORPH_OPEN, kernel)

    return clean_mask


def keep_biggest_object(mask):

    """

    Args:
        mask: uint8 255-hot encoded segmentation mask

    Returns:
        biggest_mask: 255-hot encoded segmentation mask with just the biggest object of each label

    """

    biggest_masks = np.zeros(mask.shape).astype("uint8")
    for label in range(mask.shape[2]):
        biggest_mask = np.zeros((mask.shape[0], mask.shape[1], 1)).astype("uint8")
        contours, _ = cv2.findContours(mask[:, :, label], cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
        contour_areas = [cv2.contourArea(contour) for contour in contours]
        if contour_areas:
            biggest_contour = contours[np.argmax(contour_areas)]
            biggest_mask = cv2.fillPoly(biggest_mask, pts=[biggest_contour], color=(255))
            biggest_masks[:, :, label] = biggest_mask[:, :, 0]
            # registrar lista de blobs de mayor tamaño, ordenarlas y crear criterio de prevalencia (CT lung cancer, Chaimeleon, flat iron)
        else:
            print('take care')
            return 0
    return biggest_masks


def close_lungs(mask):

    """

    Opply a closing the segmentation masks

    Args:
        mask: uint8 255-hot encoded segmentation mask

    Returns:
        closed_masks: uint8 255-hot encoded segmentation mask with closed segmentations

    """

    closed_masks = np.zeros(mask.shape).astype("uint8")
    kernel = np.ones((25, 25), np.uint8)
    for label in range(mask.shape[2]):
        closed_masks[:, :, label] = cv2.morphologyEx(mask[:, :, label], cv2.MORPH_CLOSE, kernel)

    return closed_masks


def lung_segmentation_unet(img, img_size, model_path):

    """

    Function to get the segmentation mask of chest x-ray using UNET

    Args:
        img: uint8 rgb chest x-ray image in range [0, 255]
        img_size: input size of the model
        model_path: path to the model

    Returns:
        mask_res: one-hot encoded segmentation mask where:
            [1 0 0] --> background
            [0 1 0] --> right lung
            [0 0 1] --> left lung

    """

    img_res = cv2.resize(img, (img_size, img_size))
    img_res = datagen.clahe_imhisteq(img_res) / 255

    # Get network and predict
    labels = 3
    _, model = net.get_unet_2d_ds(img_size, img_size, labels, multi_gpu=False)
    model.load_weights(model_path)
    predictions = model.predict(img_res[np.newaxis, ...])
    mask = np.round(predictions[0])
    mask_cv2 = (mask * 255).astype("uint8")
    mask_cv2 = keep_biggest_object(mask_cv2)
    mask_cv2 = close_lungs(mask_cv2)
    mask_res = cv2.resize(mask_cv2, (img.shape[1], img.shape[0])) / 255

    return mask_res

model = None

def get_model(model_path):
    class InferenceConfig(LungsConfig):
        # Set batch size to 1 since we'll be running inference on
        # one image at a time. Batch size = GPU_COUNT * IMAGES_PER_GPU
        GPU_COUNT = 1
        IMAGES_PER_GPU = 1
    config = InferenceConfig()

    # Get model
    model = modellib.MaskRCNN(mode="inference", config=config, model_dir=str(model_path))
    model.load_weights(model_path, by_name=True)
    return model

def lung_segmentation_maskrcnn(img, img_size, model):
    """

    Function to get the segmentation mask of chest x-ray using MASK RCNN

    Args:
        img: uint8 rgb chest x-ray image in range [0, 255]
        img_size: input size of the model
        model_path: path to the model

    Returns:
        mask_res: one-hot encoded segmentation mask where:
            [1 0] --> right lung
            [0 1] --> left lung

    """

    img_resize = cv2.resize(img, (img_size, img_size))
    img_resize = datagen.clahe_imhisteq(img_resize)

    # Generate configuration for inference
    # class InferenceConfig(LungsConfig):
    #     # Set batch size to 1 since we'll be running inference on
    #     # one image at a time. Batch size = GPU_COUNT * IMAGES_PER_GPU
    #     GPU_COUNT = 1
    #     IMAGES_PER_GPU = 1
    # config = InferenceConfig()
    #
    # # Get model
    # if model == None:
    #     model = modellib.MaskRCNN(mode="inference", config=config, model_dir=str(model_path))
    #     model.load_weights(model_path, by_name=True)

    # Predict and process the results
    results = model.detect([img_resize], verbose=1)
    masks_pred = results[0]["masks"]
    # This is required because the resulting masks are ordered by class probability, therefore, each channel in the
    # resulting mask is not related to a class

    # Invert the channels because in this network right lung is labeled as class 2 and left as class 1

    predictions = masks_pred[:, :, ::-1]
    mask = np.round(predictions)
    mask_cv2 = (mask * 255).astype("uint8")
    mask_cv2 = keep_biggest_object(mask_cv2)
    if np.mean(mask_cv2) == 0:
        mask_cv2 = np.zeros(img.shape)
    mask_cv2 = close_lungs(mask_cv2)
    mask_res = cv2.resize(mask_cv2, (img.shape[1], img.shape[0])) / 255
    if len(mask_res.shape) == 3:
            l = mask_res[:,:,0]
            r = mask_res[:,:,1]
            final_mask = r + l
            final_mask = (final_mask * 255).astype("uint8")
            return final_mask

    else:
            final_mask = (mask_res * 255).astype("uint8")
            return final_mask


    # plt.imshow(final_mask)
    # plt.show()


def lung_segmentation(img, model):

    """

        Function to get the segmentation mask of chest x-ray

        Args:
            img: uint8 rgb chest x-ray image in range [0, 255]

        Returns:
            mask_res: one-hot encoded segmentation mask where:
                [1 0] --> right lung
                [0 1] --> left lung


    """

    img_size = 256
    mask_res = lung_segmentation_maskrcnn(img, img_size, model)

    return mask_res

def iterate_mask(img_path, mask_path):
        #patients = os.listdir(img_path)
        print('Creating training data')
        feature_list = []
        model_path = Path(__file__).parent / "best_models" / "mrcnn_lungs_exp20_0250.h5"
        model = get_model(str(model_path))
        # files = os.listdir(img_path)
        files = img_path
        for i in range(len(files)):
            #filename_mask = find_dcm(filename)
            # get the path of the folder "original" inside files[i]
            filename = os.path.join(files[i], os.listdir(files[i])[0])
        # dss = pydicom.dcmread(filename + filename_mask, force=True)
           # dss = pydicom.dcmread(filename, force=True)
        # Intenta leer el pixel array usando SimpleITK
            print(filename)
            try:
                data = sitk.GetArrayFromImage(sitk.ReadImage(filename))
            except Exception as e:
                print(f"Error al leer el pixel array: {e}")
                data = None

            #data = dss.pixel_array

            #convertir a 2D si es 3D
            if len(data.shape) == 3:
                data = data[0]
            original_shape = data.shape
            print(data.shape)
            #data = (normalnorm(data)) * 255
            data = data.astype('uint8')

            rgb = cv2.cvtColor(data, cv2.COLOR_GRAY2BGR)

            rgb = cv2.resize(rgb, (256, 256), interpolation=cv2.INTER_CUBIC)

            mask = lung_segmentation(rgb, model)
            mask = np.rot90(mask)
            mask = np.rot90(mask)


            mask = np.rot90(mask)
            mask = np.fliplr(mask)
            mask = cv2.resize(mask, original_shape, interpolation=cv2.INTER_CUBIC)

            filename_ = mask_path[i] + '/lungmask.nrrd'
            nrrd.write(filename_, mask)



    # Volúmenes docker
input_folder = '/input'

    # Obtener la lista de carpetas de pacientes dentro del directorio general
patient_folders = [folder for folder in os.listdir(input_folder) if os.path.isdir(os.path.join(input_folder, folder))]
array_images = []
array_masks = []
for patient_folder in patient_folders:
    patient_folder_path = os.path.join(input_folder, patient_folder)
    procedure_folder = os.listdir(patient_folder_path)[0]
    img_paths = os.path.join(patient_folder_path, procedure_folder, "original")
    array_images.append(img_paths)
    mask_paths = os.path.join(patient_folder_path, procedure_folder, "segmentation")
    os.makedirs(mask_paths, exist_ok=True)
    array_masks.append(mask_paths)
    # # Obtener la lista de carpetas de procedimientos para el paciente actual
    # procedure_folders = [folder for folder in os.listdir(patient_folder_path) if os.path.isdir(os.path.join(patient_folder_path, folder))]
    # # Iterar sobre las carpetas de procedimientos
    # for index, procedure_folder in enumerate(procedure_folders, start=1):
    #     procedure_folder_path = os.path.join(patient_folder_path, procedure_folder)
        
    #     original_dcm_path = os.path.join(procedure_folder_path, "original")
    #     img_path = original_dcm_path
    #     mask_path = os.path.join(procedure_folder_path, "segmentation")
    #     if not os.path.exists(mask_path):
    #         os.makedirs(mask_path, exist_ok=True)
    #         #print(f'carpeta {patient_folder} creada')
    #         print(f'Carpeta mask creada')
    #     iterate_mask(img_path, mask_path)
    #     print(f'Segmentado el paciente {patient_folder}, y su procedure {procedure_folder}')
iterate_mask(array_images, array_masks)