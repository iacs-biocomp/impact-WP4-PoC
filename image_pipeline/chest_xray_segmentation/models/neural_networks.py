from keras.models import Model
from keras.layers import Input, concatenate, Conv2D, MaxPooling2D, Conv2DTranspose,\
                            BatchNormalization, Activation, UpSampling2D, Add
from keras.optimizers import Adam
from keras import backend as K
# from keras.utils import multi_gpu_model
from tensorflow.python.client import device_lib


def dice_coef(y_true, y_pred):

    """

    Calculate dice score

    Args:
        y_true: tensor with the ground truth segmentation mask
        y_pred: tensor with the ground truth segmentation mask

    Returns:
        dice_score: dice score between ground truth and prediction

    """

    smooth = 1e-7
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)



def dice_coef_loss_multiclass_4(y_true, y_pred):

    """

    Calculate dice score for 4 labels and average it

    Args:
        y_true: tensor with the ground truth segmentation mask
        y_pred: tensor with the ground truth segmentation mask

    Returns:
        dice_score_4_labels: average dice score of the 4 labels

    """

    dice_0 = dice_coef(y_true[:, :, :, 0], y_pred[:, :, :, 0])
    dice_1 = dice_coef(y_true[:, :, :, 1], y_pred[:, :, :, 1])
    dice_2 = dice_coef(y_true[:, :, :, 2], y_pred[:, :, :, 2])
    dice_3 = dice_coef(y_true[:, :, :, 3], y_pred[:, :, :, 3])

    mean_dice = (dice_0 + dice_1 + dice_2 + dice_3) / 4

    return 1.-mean_dice


def dice_coef_loss_multiclass_3(y_true, y_pred):

    """

    Calculate dice score for 3 labels and average it

    Args:
        y_true: tensor with the ground truth segmentation mask
        y_pred: tensor with the ground truth segmentation mask

    Returns:
        dice_score_3_labels: average dice score of the 3 labels

    """

    dice_0 = dice_coef(y_true[:, :, :, 0], y_pred[:, :, :, 0])
    dice_1 = dice_coef(y_true[:, :, :, 1], y_pred[:, :, :, 1])
    dice_2 = dice_coef(y_true[:, :, :, 2], y_pred[:, :, :, 2])

    mean_dice = (dice_0 + dice_1 + dice_2) / 3

    return 1.-mean_dice


# Unet 2D with batch normalization and Deep supervision layers for multi-gpu
def get_unet_2d_ds(img_rows, img_cols, labels, learning_rate=0.001, multi_gpu=True):

    """

    Generate UNET 2D model with deep supervision

    Args:
        img_rows: rows of the input images
        img_cols: columns of the input images
        labels: number of labels of the segmentation mask
        learning_rate: learning rate for the model
        multi_gpu: flag indicating if the network will be executed in several GPUs

    Returns:
        model: keras model of the network

    """

    # Encoding phase
    inputs = Input((img_rows, img_cols, 3))
    conv1 = Conv2D(32, (3, 3), padding='same')(inputs)
    conv1 = BatchNormalization()(conv1)
    conv1 = Activation('relu')(conv1)
    conv1 = Conv2D(32, (3, 3), padding='same')(conv1)
    conv1 = BatchNormalization()(conv1)
    conv1 = Activation('relu')(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)

    conv2 = Conv2D(64, (3, 3), padding='same')(pool1)
    conv2 = BatchNormalization()(conv2)
    conv2 = Activation('relu')(conv2)
    conv2 = Conv2D(64, (3, 3), padding='same')(conv2)
    conv2 = BatchNormalization()(conv2)
    conv2 = Activation('relu')(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)

    conv3 = Conv2D(128, (3, 3), padding='same')(pool2)
    conv3 = BatchNormalization()(conv3)
    conv3 = Activation('relu')(conv3)
    conv3 = Conv2D(128, (3, 3), padding='same')(conv3)
    conv3 = BatchNormalization()(conv3)
    conv3 = Activation('relu')(conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)

    conv4 = Conv2D(256, (3, 3), padding='same')(pool3)
    conv4 = BatchNormalization()(conv4)
    conv4 = Activation('relu')(conv4)
    conv4 = Conv2D(256, (3, 3), padding='same')(conv4)
    conv4 = BatchNormalization()(conv4)
    conv4 = Activation('relu')(conv4)
    pool4 = MaxPooling2D(pool_size=(2, 2))(conv4)

    # Decoding phase
    conv5 = Conv2D(512, (3, 3), padding='same')(pool4)
    conv5 = BatchNormalization()(conv5)
    conv5 = Activation('relu')(conv5)
    conv5 = Conv2D(512, (3, 3), padding='same')(conv5)
    conv5 = BatchNormalization()(conv5)
    conv5 = Activation('relu')(conv5)

    up6 = concatenate([Conv2DTranspose(256, (2, 2), strides=(2, 2), padding='same')(conv5), conv4], axis=3)
    conv6 = Conv2D(256, (3, 3), padding='same')(up6)
    conv6 = BatchNormalization()(conv6)
    conv6 = Activation('relu')(conv6)
    conv6 = Conv2D(256, (3, 3), padding='same')(conv6)
    conv6 = BatchNormalization()(conv6)
    conv6 = Activation('relu')(conv6)

    up7 = concatenate([Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same')(conv6), conv3], axis=3)
    conv7 = Conv2D(128, (3, 3), padding='same')(up7)
    conv7 = BatchNormalization()(conv7)
    conv7 = Activation('relu')(conv7)
    conv7 = Conv2D(128, (3, 3), padding='same')(conv7)
    conv7 = BatchNormalization()(conv7)
    conv7 = Activation('relu')(conv7)

    up8 = concatenate([Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same')(conv7), conv2], axis=3)
    conv8 = Conv2D(64, (3, 3), padding='same')(up8)
    conv8 = BatchNormalization()(conv8)
    conv8 = Activation('relu')(conv8)
    conv8 = Conv2D(64, (3, 3), padding='same')(conv8)
    conv8 = BatchNormalization()(conv8)
    conv8 = Activation('relu')(conv8)

    up9 = concatenate([Conv2DTranspose(32, (2, 2), strides=(2, 2), padding='same')(conv8), conv1], axis=3)
    conv9 = Conv2D(32, (3, 3), padding='same')(up9)
    conv9 = BatchNormalization()(conv9)
    conv9 = Activation('relu')(conv9)
    conv9 = Conv2D(32, (3, 3), padding='same')(conv9)
    conv9 = BatchNormalization()(conv9)
    conv9 = Activation('relu')(conv9)

    # Deep supervision phase
    deep_supervision_6 = Conv2D(1, (1, 1))(conv6)
    deep_supervision_7 = Conv2D(1, (1, 1))(conv7)
    deep_supervision_7 = Add()([deep_supervision_7, UpSampling2D()(deep_supervision_6)])
    deep_supervision_8 = Conv2D(1, (1, 1))(conv8)
    deep_supervision_8 = Add()([deep_supervision_8, UpSampling2D()(deep_supervision_7)])
    deep_supervision_out = Conv2D(1, (1, 1))(conv9)
    deep_supervision_out = Add()([deep_supervision_out, UpSampling2D()(deep_supervision_8)])

    # Added Layer for multi region
    deep_supervision_out = Conv2D(labels, (1, 1))(deep_supervision_out)

    # Output
    if labels == 1:
        output = Activation('sigmoid')(deep_supervision_out)
    else:
        output = Activation('softmax')(deep_supervision_out)

    # Set inputs and outputs
    model = Model(inputs=[inputs], outputs=[output])

    # # Using model for multi-gpu
    # if multi_gpu:
    #     #Get number of GPUs
    #     local_device_protos = device_lib.list_local_devices()
    #     gpus = len([x.name for x in local_device_protos if x.device_type == 'GPU'])
    #     print("=================================", "Number of GPUs: " + str(gpus), "=================================")
    #     model_mgpu = multi_gpu_model(model, gpus=gpus)
    #     if labels == 3:
    #         model_mgpu.compile(optimizer=Adam(lr=learning_rate), loss=dice_coef_loss_multiclass_3,
    #                            metrics=[dice_coef_loss_multiclass_3])
    #     elif labels == 4:
    #         model_mgpu.compile(optimizer=Adam(lr=learning_rate), loss=dice_coef_loss_multiclass_4,
    #                            metrics=[dice_coef_loss_multiclass_4])
    #     return model_mgpu, model

    # Compile model
    if labels == 3:
        model.compile(optimizer=Adam(lr=learning_rate), loss=dice_coef_loss_multiclass_3,
                      metrics=[dice_coef_loss_multiclass_3])
    elif labels == 4:
        model.compile(optimizer=Adam(lr=learning_rate), loss=dice_coef_loss_multiclass_4,
                      metrics=[dice_coef_loss_multiclass_4])

    return None, model
