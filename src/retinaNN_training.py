import sys
import matplotlib.pyplot as plt
sys.path.append('..\lib\\')
from extract_datapatches import *
from help_function import *
from keras.models import Model
from keras.layers import Input, concatenate, Conv2D, MaxPooling2D, UpSampling2D, Reshape, Activation, Dropout,BatchNormalization,LeakyReLU
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint, LearningRateScheduler
from keras import backend as K
from keras.utils.vis_utils import plot_model as plot
from keras.optimizers import SGD

orginal_image_dir = '..\h5d_data\hdf5_images_train.hdf5'
groundTure_image_dir = '..\h5d_data\hdf5_groundTures_train.hdf5'
image_test_dir = '..\h5d_data\hdf5_images_test.hdf5'
groundTure_test_dir = '..\h5d_data\hdf5_groundTures_test.hdf5'


def get_unet(n_ch, patch_height, patch_width):
    inputs = Input(shape=(n_ch, patch_height, patch_width))
    conv1 = Conv2D(64, (3, 3), activation='relu', padding='same', data_format='channels_first')(inputs)
    # conv1 = Dropout(0.2)(conv1)
    conv1 = BatchNormalization()(conv1)
    conv1 = LeakyReLU()(conv1)
    conv1 = Conv2D(64, (3, 3), activation='relu', padding='same', data_format='channels_first')(conv1)
    pool1 = MaxPooling2D((2, 2),data_format='channels_first')(conv1)
    #
    conv2 = Conv2D(128, (3, 3), activation='relu', padding='same', data_format='channels_first')(pool1)
    # conv2 = Dropout(0.2)(conv2)
    conv2 = BatchNormalization()(conv2)
    conv2 = LeakyReLU()(conv2)
    conv2 = Conv2D(128, (3, 3), activation='relu', padding='same', data_format='channels_first')(conv2)
    pool2 = MaxPooling2D((2, 2),data_format='channels_first')(conv2)
    #
    conv3 = Conv2D(256, (3, 3), activation='relu', padding='same', data_format='channels_first')(pool2)
    # conv3 = Dropout(0.2)(conv3)
    conv3=BatchNormalization()(conv3)
    conv3=LeakyReLU()(conv3)
    conv3 = Conv2D(256, (3, 3), activation='relu', padding='same', data_format='channels_first')(conv3)

    up1 = UpSampling2D(size=(2, 2),data_format='channels_first')(conv3)
    up1 = concatenate([conv2, up1], axis=1)
    conv4 = Conv2D(128, (3, 3), activation='relu', padding='same', data_format='channels_first')(up1)
    conv4 = Dropout(0.2)(conv4)
    conv4 = BatchNormalization()(conv4)
    conv4 = LeakyReLU()(conv4)
    conv4 = Conv2D(128, (3, 3), activation='relu', padding='same', data_format='channels_first')(conv4)
    #
    up2 = UpSampling2D(size=(2, 2),data_format='channels_first')(conv4)
    up2 = concatenate([conv1, up2], axis=1)
    conv5 = Conv2D(64, (3, 3), activation='relu', padding='same', data_format='channels_first')(up2)
    conv5 = Dropout(0.2)(conv5)
    conv5 = Conv2D(64, (3, 3), activation='relu', padding='same', data_format='channels_first')(conv5)
    #
    conv6 = Conv2D(1, (1, 1), padding='same', data_format='channels_first')(conv5)

    ############
    conv7 = Activation('sigmoid')(conv6)

    model = Model(inputs=inputs, outputs=conv7)

    sgd = SGD(lr=0.02, decay=1e-6, momentum=0.3, nesterov=False)
    model.compile(optimizer=sgd, loss=[DSC_loss], metrics=['accuracy',DSC,sensitive,PPV])

    return model

'''
定义损失函数
'''

smooth = 1.
from keras import backend as K
######################################### Dice Similarity Coefficient  Dice和DSC一个东西 ######################################################
#https://blog.csdn.net/a362682954/article/details/81179276
def DSC(y_true,y_pred):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (K.sum(y_true_f * y_true_f) + K.sum(y_pred_f * y_pred_f) + smooth)

def DSC_loss(y_true, y_pred):
    return 1-DSC(y_true, y_pred)
############################################# sensitive 和 recall 是一个东西 ###############################################

def sensitive(y_true, y_pred):
  """Recall metric.

  Only computes a batch-wise average of recall.

  Computes the recall, a metric for multi-label classification of
  how many relevant items are selected.
  """
  true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
  possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
  recall = true_positives / (possible_positives + K.epsilon())
  return recall
############################################## PPV 和 precision 是一个东西 ###############################################
def PPV(y_true, y_pred):
  """Precision metric.

  Only computes a batch-wise average of precision.

  Computes the precision, a metric for multi-label classification of
  how many selected items are relevant.
  """
  true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
  predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
  precision = true_positives / (predicted_positives + K.epsilon())
  return precision


if __name__ == '__main__':
    patch_image_train, patch_ground_train = get_data_train(orginal_image_dir, groundTure_image_dir, patch_h=48, patch_w=48,
                                                         N_patch=1300)
    print(patch_ground_train.max())
    # plt.subplot(121)
    # plt.imshow(patch_ground_train[204,0],'gray')
    # plt.subplot(122)
    # plt.imshow(patch_image_train[204, 0], 'gray')
    # plt.show()
    # new_patch_ground_train=masks_Unet(patch_ground_train)
    #得到模型
    model = get_unet(patch_image_train.shape[1], patch_height=patch_ground_train.shape[2],
                     patch_width=patch_image_train.shape[3])
    #打印输出
    print("Check: final output of the network:")
    print(model.output_shape)
    model.summary()
    #保存模型
    json_string = model.to_json()
    open('../' + 'test' + '/' + 'test' + '_architecture.json', 'w').write(json_string)
    # ============  Training ==================================
    model.load_weights('../test/test_best_weights.h5')
    checkpointer = ModelCheckpoint(filepath='../' + 'test' + '/' + 'test' + '_best_weights.h5',
                                   verbose=1, monitor='val_loss', mode='auto',
                                   save_best_only=True)  # save at each epoch if the validation decreased
    model.fit(patch_image_train, patch_ground_train, epochs=20, batch_size=12,
              verbose=1, shuffle=True,
              validation_split=0.3, callbacks=[checkpointer])
#========== Save and test the last model ===================
    model.save_weights('../'+'test'+'/'+'test' +'_last_weights.h5', overwrite=True)