import numpy as np
from numpy import random
import cv2
from pathlib import Path
import json
import imgaug.augmenters as iaa
from imgaug.augmentables.segmaps import SegmentationMapOnImage

# Import custom libraries
import load_data as load


def mask2onehot(mask_rgb, labels):

    """
    Args:
        mask_rgb: rgb segmentation mask that can be right and left lungs or right lung, left lung and mediastinum
        labels: it can be 3 or 4 depending on the labels included in the segmentation mask

    Returns:
        mask_onehot: one hot encoded segmentation mask

    """

    mask_onehot = np.zeros((mask_rgb.shape[0], mask_rgb.shape[1], labels))
    if labels == 4:
        mask_onehot[mask_rgb[:, :, 0] == 1] = [0, 1, 0, 0]
        mask_onehot[mask_rgb[:, :, 1] == 1] = [0, 0, 1, 0]
        mask_onehot[mask_rgb[:, :, 2] == 1] = [0, 0, 0, 1]
        mask_onehot[np.logical_and(np.logical_and(mask_rgb[:, :, 0] == 0, mask_rgb[:, :, 1] == 0),
                                   mask_rgb[:, :, 2] == 0)] = [1, 0, 0, 0]
    elif labels == 3:
        mask_onehot[mask_rgb[:, :, 0] == 1] = [0, 1, 0]
        mask_onehot[mask_rgb[:, :, 1] == 1] = [0, 0, 1]
        mask_onehot[np.logical_and(mask_rgb[:, :, 0] == 0, mask_rgb[:, :, 1] == 0)] = [1, 0, 0]

    else:
        print("Wrong label number")

    return mask_onehot


def imhisteq(img):

    """

    Equalize histogram of an image considering global contrast

    Args:
        img: bgr uint8 matrix

    Returns:
        img_eq: bgr uint8 matrix with histogram equalized

    """

    # Histogram equalization
    img_lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    img_lab[:, :, 0] = cv2.equalizeHist(img_lab[:, :, 0])
    img_eq = cv2.cvtColor(img_lab, cv2.COLOR_LAB2BGR)

    return img_eq


def clahe_imhisteq(img):

    """

    Equalize histogram of an image considering local contrast

    Args:
        img: bgr uint8 matrix

    Returns:
        img_clahe: bgr uint8 matrix with histogram equalized

    """

    img_lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    img_lab[:, :, 0] = clahe.apply(img_lab[:, :, 0])
    img_clahe = cv2.cvtColor(img_lab, cv2.COLOR_LAB2BGR)

    return img_clahe


def data_generator(images_path, masks_path, seq, batch_size, img_size, labels):

    """

    Definition of custom data generator

    Args:
        images_path: path to the images directory
        masks_path: path to the masks directory
        seq: imgaug sequence of data augmentation
        batch_size: batch of augmented data per iteration
        img_size: desired size to which resize the loaded images and masks
        labels: number of segmentation labels

    Returns:
        batch_images: batch of augmented images
        batch_masks: batch of augmented masks

    """

    images_path_list = list(images_path.glob('*.png'))
    masks_path_list = list(masks_path.glob('*.png'))
    batch_images = np.zeros([batch_size, img_size, img_size, 3])
    batch_masks = np.zeros([batch_size, img_size, img_size, labels])
    index = 0
    shuffle_indexes = np.arange(len(images_path_list))
    random.shuffle(shuffle_indexes)
    while True:
        for b in range(batch_size):
            if index == len(images_path_list):
                index = 0
                shuffle_indexes = np.arange(len(images_path_list))
                random.shuffle(shuffle_indexes)
            image = cv2.resize(cv2.imread(str(images_path_list[shuffle_indexes[index]])),
                             (img_size, img_size)).astype("uint8") # img aug needs uint8 as input for images
            mask = cv2.resize(cv2.imread(str(masks_path_list[shuffle_indexes[index]])),
                              (img_size, img_size)) / 255
            mask = np.argmax(mask2onehot(mask, labels), axis=2) # img aug needs int labels from 1 to n as input
            segmap = SegmentationMapOnImage(mask, nb_classes=labels, shape=image.shape)
            image_aug, segmap_aug = seq(image=image, segmentation_maps=segmap)
            # Apply CLAHE for histogram equalization
            batch_images[b] = clahe_imhisteq(image_aug) / 255
            batch_masks[b] = segmap_aug.arr
            index += 1
        yield batch_images, batch_masks


def create_generators(data_path, train_seq, val_seq, batch_size, img_size, labels):

    """

    Create training and validation data generators

    Args:
        data_path: Pathlib path to folder with data
        train_seq: imgaug augmentation sequence for training images
        val_seq: imgaug augmentation sequence for validation images
        batch_size: size of batch of data to pass to the neural network
        img_size: size of the images to input to the neural network
        labels: number of segmentation labels

    Returns:
        gen_train: generator of training samples
        gen_val: generator of validation samples
        train_samples: number of training samples
        val_samples: number of validation samples

    """

    images_path = data_path / "images_png"
    masks_path = data_path / "masks_png"

    gen_train = data_generator(images_path=images_path / "training", masks_path=masks_path / "training",
                               seq=train_seq, batch_size=batch_size, img_size=img_size, labels=labels)
    gen_val = data_generator(images_path=images_path / "validation", masks_path=masks_path / "validation",
                             seq=val_seq, batch_size=batch_size, img_size=img_size, labels=labels)

    train_samples = len(list((images_path / "training").glob("*.png")))
    val_samples = len(list((images_path / "validation").glob("*.png")))

    return gen_train, gen_val, train_samples, val_samples


if __name__ == "__main__":

    with open("../config/config_training.json") as config_file:
        config_json = json.load(config_file)
    data_path = Path(config_json["paths"]["path_to_data"])

    ####################################################################################################################
    #                                             TEST FOR CREATE GENERATORS                                           #
    ####################################################################################################################

    # images_path = data_path / "images_png" / "images"
    # masks_path = data_path / "masks_png" / "masks"
    # batch_size = 25
    # input_size = 256
    # labels = 4
    # seq = iaa.Sequential([
    #     iaa.Crop(percent=(0, 0.1)),  # random crops
    #     iaa.Sometimes(0.5,
    #                   iaa.GaussianBlur(sigma=(0, 0.5))
    #                   ),
    #     iaa.ContrastNormalization((0.75, 1.5)), # Strengthen or weaken the contrast in each image.
    #     iaa.AdditiveGaussianNoise(loc=0, scale=(0.0, 0.05 * 255)), # Add gaussian noise.
    #     iaa.Multiply((0.8, 1.2)), # Make some images brighter and some darker.
    #     iaa.Affine( # Apply affine transformations to each image.
    #         scale={"x": (0.95, 1), "y": (0.95, 1.05)},
    #         rotate=(-10, 10),
    #         shear=(-4, 4)
    #     )
    # ], random_order=True)  # apply augmenters in random order
    # # seq = iaa.Sequential([])
    # gen = data_generator(images_path, masks_path, seq, batch_size, input_size, labels)
    # batch = next(gen)
    # images = batch[0]
    # masks = batch[1]
    # load.plot_annotation(images[2], masks[2])

    ####################################################################################################################
    #                                             TEST FOR IMGAUG                                                      #
    ####################################################################################################################

    # # Single image
    # image_path = data_path / "images_png" / "images" / "000.png"
    # mask_path = data_path / "masks_png" / "masks" / "000.png"
    # image = cv2.imread(str(image_path)).astype("uint8")
    # mask = cv2.imread(str(mask_path)) / 255
    # labels = 4
    # segmap = SegmentationMapOnImage(np.argmax(mask2onehot(mask, labels), axis=2), nb_classes=4, shape=image.shape)
    # print(segmap.arr.shape)
    # ia.imshow(segmap.draw_on_image(image))
    # ia.seed(1)
    # # Example batch of augmented images.
    # seq = iaa.Sequential([
    #     iaa.Crop(percent=(0, 0.1)),  # random crops
    #     # Small gaussian blur with random sigma between 0 and 0.5.
    #     # But we only blur about 50% of all images.
    #     iaa.Sometimes(0.5,
    #                   iaa.GaussianBlur(sigma=(0, 0.5))
    #                   ),
    #     # Strengthen or weaken the contrast in each image.
    #     iaa.ContrastNormalization((0.75, 1.5)),
    #     # Add gaussian noise.
    #     # For 50% of all images, we sample the noise once per pixel.
    #     iaa.AdditiveGaussianNoise(loc=0, scale=(0.0, 0.05 * 255)),
    #     # Make some images brighter and some darker.
    #     iaa.Multiply((0.8, 1.2)),
    #     # Apply affine transformations to each image.
    #     # Scale/zoom them, translate/move them, rotate them and shear them.
    #     iaa.Affine(
    #         scale={"x": (0.9, 1.1), "y": (0.9, 1.1)},
    #         translate_percent={"x": (-0.1, 0.1), "y": (-0.1, 0.1)},
    #         rotate=(-10, 10),
    #         shear=(-8, 8)
    #     )
    # ], random_order=True)  # apply augmenters in random order
    # image_aug, segmap_aug = seq(image=image, segmentation_maps=segmap)
    # ia.imshow(np.hstack([
    #     segmap_aug.draw_on_image(image_aug),  # show blend of (augmented) image and segmentation map
    #     segmap_aug.draw()  # show only the augmented segmentation map
    # ]))

    ####################################################################################################################
    #                                             TEST FOR create_generators                                           #
    ####################################################################################################################

    # batch_size = 25
    # img_size = 256
    # labels = 4
    # train_seq = iaa.Sequential([
    #         iaa.Crop(percent=(0, 0.1)),  # Random crops
    #         iaa.Sometimes(0.5,
    #                       iaa.GaussianBlur(sigma=(0, 0.5))
    #                       ),
    #         iaa.ContrastNormalization((0.75, 1.5)),  # Strengthen or weaken the contrast in each image.
    #         iaa.AdditiveGaussianNoise(loc=0, scale=(0.0, 0.05 * 255)),  # Add gaussian noise.
    #         iaa.Multiply((0.8, 1.2)),  # Make some images brighter and some darker.
    #         iaa.Affine(  # Apply affine transformations to each image.
    #             scale={"x": (0.95, 1), "y": (0.95, 1.05)},
    #             rotate=(-10, 10),
    #             shear=(-4, 4)
    #         )
    # ], random_order=True)  # Apply augmenters in random order
    # val_seq = iaa.Sequential([])
    # gen_train, gen_val, train_samples, val_samples = create_generators(data_path, train_seq, val_seq, batch_size,
    #                                                                    img_size, labels)
    # images_train, masks_train = next(gen_train)
    # images_val, masks_val = next(gen_val)
    # print(train_samples)
    # print(val_samples)
    # show_index = 11
    # load.plot_annotation(images_train[show_index], masks_train[show_index])
    # load.plot_annotation(images_val[show_index], masks_val[show_index])


    ####################################################################################################################
    #                                             TEST FOR create_generators COMBINED LUNG DATASETS                    #
    ####################################################################################################################

    data_path = Path(config_json["paths"]["path_to_data_combined3"])
    batch_size = 25
    img_size = 256
    labels = 3
    train_seq = iaa.Sequential([
            iaa.Crop(percent=(0, 0.1)),  # Random crops
            iaa.Sometimes(0.5,
                          iaa.GaussianBlur(sigma=(0, 0.5))
                          ),
            iaa.ContrastNormalization((0.75, 1.5)),  # Strengthen or weaken the contrast in each image.
            iaa.AdditiveGaussianNoise(loc=0, scale=(0.0, 0.05 * 255)),  # Add gaussian noise.
            iaa.Multiply((0.8, 1.2)),  # Make some images brighter and some darker.
            iaa.Affine(  # Apply affine transformations to each image.
                scale={"x": (0.45, 1.15), "y": (0.45, 1.15)},
                rotate=(-10, 10),
                shear=(-4, 4)
            )
    ], random_order=True)  # Apply augmenters in random order
    val_seq = iaa.Sequential([])
    gen_train, gen_val, train_samples, val_samples = create_generators(data_path, train_seq, val_seq, batch_size,
                                                                       img_size, labels)
    images_train, masks_train = next(gen_train)
    images_val, masks_val = next(gen_val)
    print(train_samples)
    print(val_samples)
    for show_index in range(batch_size):
        load.plot_annotation(images_train[show_index], masks_train[show_index])
        load.plot_annotation(images_val[show_index], masks_val[show_index])


    ####################################################################################################################
    #                                             TEST FOR HISTOGRAM EQUALIZATION                                      #
    ####################################################################################################################

    # data_path = Path(config_json["paths"]["path_to_data_combined"])
    # img_path = data_path / 'images_png' / "training" / "CHNCXR_0635_1.png"
    # img = cv2.imread(str(img_path))
    # img_eq = imhisteq(img)
    # img_clahe = clahe_imhisteq(img)
    # load.plot_img(np.hstack([img, img_eq, img_clahe]))

    ### IMPORTANT: THE CONCLUSION OF THE EXPERIMENT IS THAT CLAHE IS BETTER BECAUSE IT KEEPS BETTER THE INFORMATION IN
    ### THE IMAGE