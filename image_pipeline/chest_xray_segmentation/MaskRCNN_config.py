from tensorflow.python.client import device_lib
from pathlib import Path
import numpy as np
from PIL import Image
import skimage

# Import custom libraries
import data_generator_custom as datagen

# Import Mask RCNN
from models.Mask_RCNN.config import Config
from models.Mask_RCNN import model as modellib
from models.Mask_RCNN import utils


class LungsConfig(Config):
    """Configuration for training.
    Derives from the base Config class and overrides some values.
    """
    # Give the configuration a recognizable name
    NAME = "lungs"

    # Adjust down if you use a smaller GPU.
    local_device_protos = device_lib.list_local_devices()
    GPU_COUNT = len([x.name for x in local_device_protos if x.device_type == 'GPU'])
    IMAGES_PER_GPU = 10

    # Number of classes (including background)
    NUM_CLASSES = 1 + 2  # Background + lungs

    # Dimensions of training images.
    IMAGE_MIN_DIM = 256
    IMAGE_MAX_DIM = 256

    # Number of training steps per epoch
    STEPS_PER_EPOCH = 300

    # Number of validation steps to run at the end of every training epoch.
    # A bigger number improves accuracy of validation stats, but slows
    # down the training.
    VALIDATION_STEPS = 50

    # Skip detections with < 90% confidence
    DETECTION_MIN_CONFIDENCE = 0.8

    # Length of square anchor side in pixels
    RPN_ANCHOR_SCALES = (125, 150, 175, 225, 256)

    # Ratios of anchors at each cell (width/height)
    RPN_ANCHOR_RATIOS = [0.6, 0.7, 0.8]

    # Number of ROIs per image to feed to classifier/mask heads
    # The Mask RCNN paper uses 512 but often the RPN doesn't generate
    # enough positive proposals to fill this and keep a positive:negative
    # ratio of 1:3. You can increase the number of proposals by adjusting
    # the RPN NMS threshold.
    TRAIN_ROIS_PER_IMAGE = 100

    # Max number of final detections
    DETECTION_MAX_INSTANCES = 2


class LungsDataset(utils.Dataset):

    def load_lungs(self, dataset_dir, subset):

        """
        Load a subset of the Balloon dataset.

        dataset_dir: Root directory of the dataset.
        subset: Subset to load: train or val

        """
        # Add classes. Right lung is class two because opencv stores the iamges in BGR format and in this code skimage
        # is used to load the images, therefore they are loaded in RGB format.
        self.add_class("lungs", 2, "right_lung")
        self.add_class("lungs", 1, "left_lung")

        # Train or validation dataset?
        assert subset in ["training", "validation"]
        images_dir = Path(dataset_dir) / "images_png" / subset

        for image_path in images_dir.glob("*.png"):
            self.add_image("lungs",
                           image_id=image_path.name,  # use file name as a unique image id
                           path=str(image_path),
                           mask_path=str(Path(dataset_dir) / "masks_png" / subset / image_path.name)
                           )


    def load_image(self, image_id):
        """Load the specified image and return a [H,W,3] Numpy array.
        """
        # Load image
        image = skimage.io.imread(self.image_info[image_id]['path'])
        # If grayscale. Convert to RGB for consistency.
        if image.ndim != 3:
            image = skimage.color.gray2rgb(image)
        image = datagen.clahe_imhisteq(image)

        return image


    def load_mask(self, image_id):

        """
        Generate instance masks for an image.

        Returns:
        masks: A bool array of shape [height, width, instance count] with
            one mask per instance.
        class_ids: a 1D array of class IDs of the instance masks.
        """

        info = self.image_info[image_id]
        if info["source"] != "lungs":
            return super(self.__class__, self).load_mask(image_id)
        mask_path = info["mask_path"]
        mask = np.array(Image.open(mask_path))
        #Remove background channel
        mask = mask[:, :, 1:]
        class_ids = np.array([1, 2], dtype=np.int32)

        return mask, class_ids



    def image_reference(self, image_id):

        """Return the path of the image."""
        info = self.image_info[image_id]
        if info["source"] == "lungs":
            return info["path"]
        else:
            super(self.__class__, self).image_reference(image_id)

#
#
# if __name__ == "__main__":
#
#     ####################################################################################################################
#     #                                             TEST CONFIGURATION                                                   #
#     ####################################################################################################################
#
#     import os
#     import itertools
#     import math
#     import logging
#     import json
#     import re
#     import random
#     from collections import OrderedDict
#     if sys.platform == 'darwin':
#         import matplotlib
#         matplotlib.use('Qt5Agg')
#     import matplotlib.pyplot as plt
#     import matplotlib.patches as patches
#     import matplotlib.lines as lines
#     from matplotlib.patches import Polygon
#
#     # Import Mask RCNN
#     from Mask_RCNN import visualize
#     from Mask_RCNN.visualize import display_images
#     from Mask_RCNN.model import log
#
#     ###### 1 ######
#
#     dataset_dir = "/Users/rafalopez/Documents/Data/ChestXray/Lung segmentation/montgomery_china_xlsor_jsrt"
#     config = LungsConfig()
#     dataset = LungsDataset()
#     dataset.load_lungs(dataset_dir, "training")
#     dataset.prepare()
#     print("Image Count: {}".format(len(dataset.image_ids)))
#     print("Class Count: {}".format(dataset.num_classes))
#     for i, info in enumerate(dataset.class_info):
#         print("{:3}. {:50}".format(i, info['name']))
#
#
#     ###### 2 ######
#
#     # Load and display random samples
#     image_ids = np.random.choice(dataset.image_ids, 4)
#     for image_id in image_ids:
#         image = dataset.load_image(image_id)
#         mask, class_ids = dataset.load_mask(image_id)
#         visualize.display_top_masks(image, mask, class_ids, dataset.class_names)
#
#
#     ###### 3 ######
#
#     # Load random image and mask.
#     image_id = random.choice(dataset.image_ids)
#     image = dataset.load_image(image_id)
#     mask, class_ids = dataset.load_mask(image_id)
#     # Compute Bounding box
#     bbox = utils.extract_bboxes(mask)
#
#     # Display image and additional stats
#     print("image_id ", image_id, dataset.image_reference(image_id))
#     log("image", image)
#     log("mask", mask)
#     log("class_ids", class_ids)
#     log("bbox", bbox)
#     # Display image and instances
#     visualize.display_instances(image, bbox, mask, class_ids, dataset.class_names)
#
#
#     ###### 4 ######
#
#     # Load random image and mask.
#     image_id = np.random.choice(dataset.image_ids, 1)[0]
#     image = dataset.load_image(image_id)
#     mask, class_ids = dataset.load_mask(image_id)
#     original_shape = image.shape
#     # Resize
#     image, window, scale, padding = utils.resize_image(
#         image,
#         min_dim=config.IMAGE_MIN_DIM,
#         max_dim=config.IMAGE_MAX_DIM,
#         padding=config.IMAGE_PADDING)
#     mask = utils.resize_mask(mask, scale, padding)
#     # Compute Bounding box
#     bbox = utils.extract_bboxes(mask)
#
#     # Display image and additional stats
#     print("image_id: ", image_id, dataset.image_reference(image_id))
#     print("Original shape: ", original_shape)
#     log("image", image)
#     log("mask", mask)
#     log("class_ids", class_ids)
#     log("bbox", bbox)
#     # Display image and instances
#     visualize.display_instances(image, bbox, mask, class_ids, dataset.class_names)
#
#
#     ###### 5 ######
#
#     image_id = np.random.choice(dataset.image_ids, 1)[0]
#     image, image_meta, class_ids, bbox, mask = modellib.load_image_gt(
#         dataset, config, image_id, use_mini_mask=False)
#
#     log("image", image)
#     log("image_meta", image_meta)
#     log("class_ids", class_ids)
#     log("bbox", bbox)
#     log("mask", mask)
#
#     display_images([image] + [mask[:, :, i] for i in range(min(mask.shape[-1], 7))])
#     visualize.display_instances(image, bbox, mask, class_ids, dataset.class_names)
#
#     # Add augmentation and mask resizing.
#     image, image_meta, class_ids, bbox, mask = modellib.load_image_gt(
#         dataset, config, image_id, augment=True, use_mini_mask=True)
#     log("mask", mask)
#     display_images([image] + [mask[:, :, i] for i in range(min(mask.shape[-1], 7))])
#     mask = utils.expand_mask(bbox, mask, image.shape)
#     visualize.display_instances(image, bbox, mask, class_ids, dataset.class_names)
#
#
#     ###### 6 ######
#
#     # Generate Anchors
#     anchors = utils.generate_pyramid_anchors(config.RPN_ANCHOR_SCALES,
#                                              config.RPN_ANCHOR_RATIOS,
#                                              config.BACKBONE_SHAPES,
#                                              config.BACKBONE_STRIDES,
#                                              config.RPN_ANCHOR_STRIDE)
#
#     # Print summary of anchors
#     num_levels = len(config.BACKBONE_SHAPES)
#     anchors_per_cell = len(config.RPN_ANCHOR_RATIOS)
#     print("Count: ", anchors.shape[0])
#     print("Scales: ", config.RPN_ANCHOR_SCALES)
#     print("ratios: ", config.RPN_ANCHOR_RATIOS)
#     print("Anchors per Cell: ", anchors_per_cell)
#     print("Levels: ", num_levels)
#     anchors_per_level = []
#     for l in range(num_levels):
#         num_cells = config.BACKBONE_SHAPES[l][0] * config.BACKBONE_SHAPES[l][1]
#         anchors_per_level.append(anchors_per_cell * num_cells // config.RPN_ANCHOR_STRIDE ** 2)
#         print("Anchors in Level {}: {}".format(l, anchors_per_level[l]))
#
#     ## Visualize anchors of one cell at the center of the feature map of a specific level
#
#     # Load and draw random image
#     image_id = np.random.choice(dataset.image_ids, 1)[0]
#     image, image_meta, _, _, _ = modellib.load_image_gt(dataset, config, image_id)
#     fig, ax = plt.subplots(1, figsize=(10, 10))
#     ax.imshow(image)
#     levels = len(config.BACKBONE_SHAPES)
#
#     for level in range(levels):
#         colors = visualize.random_colors(levels)
#         # Compute the index of the anchors at the center of the image
#         level_start = sum(anchors_per_level[:level])  # sum of anchors of previous levels
#         level_anchors = anchors[level_start:level_start + anchors_per_level[level]]
#         print("Level {}. Anchors: {:6}  Feature map Shape: {}".format(level, level_anchors.shape[0],
#                                                                       config.BACKBONE_SHAPES[level]))
#         center_cell = config.BACKBONE_SHAPES[level] // 2
#         center_cell_index = (center_cell[0] * config.BACKBONE_SHAPES[level][1] + center_cell[1])
#         level_center = center_cell_index * anchors_per_cell
#         center_anchor = anchors_per_cell * (
#                 (center_cell[0] * config.BACKBONE_SHAPES[level][1] / config.RPN_ANCHOR_STRIDE ** 2) \
#                 + center_cell[1] / config.RPN_ANCHOR_STRIDE)
#         level_center = int(center_anchor)
#
#         # Draw anchors. Brightness show the order in the array, dark to bright.
#         for i, rect in enumerate(level_anchors[level_center:level_center + anchors_per_cell]):
#             y1, x1, y2, x2 = rect
#             p = patches.Rectangle((x1, y1), x2 - x1, y2 - y1, linewidth=2, facecolor='none',
#                                   edgecolor=(i + 1) * np.array(colors[level]) / anchors_per_cell)
#             ax.add_patch(p)
#
#
#     ###### 7 ######
#
#     random_rois = 2000
#     g = modellib.data_generator(
#         dataset, config, shuffle=True, random_rois=random_rois,
#         batch_size=4,
#         detection_targets=True)
#
#     # # Uncomment to run the generator through a lot of images
#     # # to catch rare errors
#     #  for i in range(1000):
#     #      print(i)
#     #      _, _ = next(g)
#
#     # Get Next Image
#     if random_rois:
#         [normalized_images, image_meta, rpn_match, rpn_bbox, gt_class_ids, gt_boxes, gt_masks, rpn_rois, rois], \
#         [mrcnn_class_ids, mrcnn_bbox, mrcnn_mask] = next(g)
#
#         log("rois", rois)
#         log("mrcnn_class_ids", mrcnn_class_ids)
#         log("mrcnn_bbox", mrcnn_bbox)
#         log("mrcnn_mask", mrcnn_mask)
#     else:
#         [normalized_images, image_meta, rpn_match, rpn_bbox, gt_boxes, gt_masks], _ = next(g)
#
#     log("gt_class_ids", gt_class_ids)
#     log("gt_boxes", gt_boxes)
#     log("gt_masks", gt_masks)
#     log("rpn_match", rpn_match, )
#     log("rpn_bbox", rpn_bbox)
#     image_id = image_meta[0][0]
#     print("image_id: ", image_id, dataset.image_reference(image_id))
#
#     # Remove the last dim in mrcnn_class_ids. It's only added
#     # to satisfy Keras restriction on target shape.
#     mrcnn_class_ids = mrcnn_class_ids[:, :, 0]
#
#     b = 0
#
#     # Restore original image (reverse normalization)
#     sample_image = modellib.unmold_image(normalized_images[b], config)
#
#     # Compute anchor shifts.
#     indices = np.where(rpn_match[b] == 1)[0]
#     refined_anchors = utils.apply_box_deltas(anchors[indices], rpn_bbox[b, :len(indices)] * config.RPN_BBOX_STD_DEV)
#     log("anchors", anchors)
#     log("refined_anchors", refined_anchors)
#
#     # Get list of positive anchors
#     positive_anchor_ids = np.where(rpn_match[b] == 1)[0]
#     print("Positive anchors: {}".format(len(positive_anchor_ids)))
#     negative_anchor_ids = np.where(rpn_match[b] == -1)[0]
#     print("Negative anchors: {}".format(len(negative_anchor_ids)))
#     neutral_anchor_ids = np.where(rpn_match[b] == 0)[0]
#     print("Neutral anchors: {}".format(len(neutral_anchor_ids)))
#
#     # ROI breakdown by class
#     for c, n in zip(dataset.class_names, np.bincount(mrcnn_class_ids[b].flatten())):
#         if n:
#             print("{:23}: {}".format(c[:20], n))
#
#     # Show positive anchors
#     fig, ax = plt.subplots(1, figsize=(16, 16))
#     visualize.draw_boxes(sample_image, boxes=anchors[positive_anchor_ids],
#                          refined_boxes=refined_anchors, ax=ax)
#     # Show negative anchors
#     visualize.draw_boxes(sample_image, boxes=anchors[negative_anchor_ids])
#
#     if random_rois:
#         # Class aware bboxes
#         bbox_specific = mrcnn_bbox[b, np.arange(mrcnn_bbox.shape[1]), mrcnn_class_ids[b], :]
#
#         # Refined ROIs
#         refined_rois = utils.apply_box_deltas(rois[b].astype(np.float32), bbox_specific[:, :4] * config.BBOX_STD_DEV)
#
#         # Class aware masks
#         mask_specific = mrcnn_mask[b, np.arange(mrcnn_mask.shape[1]), :, :, mrcnn_class_ids[b]]
#
#         visualize.draw_rois(sample_image, rois[b], refined_rois, mask_specific, mrcnn_class_ids[b], dataset.class_names)
#
#         # Any repeated ROIs?
#         rows = np.ascontiguousarray(rois[b]).view(np.dtype((np.void, rois.dtype.itemsize * rois.shape[-1])))
#         _, idx = np.unique(rows, return_index=True)
#         print("Unique ROIs: {} out of {}".format(len(idx), rois.shape[1]))
#
#     if random_rois:
#         # Dispalay ROIs and corresponding masks and bounding boxes
#         ids = random.sample(range(rois.shape[1]), 8)
#
#         images = []
#         titles = []
#         for i in ids:
#             image = visualize.draw_box(sample_image.copy(), rois[b, i, :4].astype(np.int32), [255, 0, 0])
#             image = visualize.draw_box(image, refined_rois[i].astype(np.int64), [0, 255, 0])
#             images.append(image)
#             titles.append("ROI {}".format(i))
#             images.append(mask_specific[i] * 255)
#             titles.append(dataset.class_names[mrcnn_class_ids[b, i]][:20])
#
#         display_images(images, titles, cols=4, cmap="Blues", interpolation="none")