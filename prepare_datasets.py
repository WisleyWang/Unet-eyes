import h5py
import os
import numpy as np
from PIL import Image


# 将数据写成hd5
def write_hd5py(data, file):
    with h5py.File(file, 'w') as f:
        f.create_dataset(data=data, dtype=data.dtype, name='image')


# -------------------------文件路径------------------------
# train
original_image_train = r'.\DRIVE\training\images\\'
groundTure_image_train = r'.\DRIVE\training\1st_manual\\'
borderMasker_image_train = r'.\DRIVE\training\mask\\'
# test
original_image_test = r'.\DRIVE\test\images\\'
groundTure_image_test = r'.\DRIVE\test\1st_manual\\'
borderMasker_image_test = r'.\DRIVE\test\mask\\'
# ----------------------------------------------------------
Nimage = 20
channels = 3
height = 584
width = 565


# 读取数据
def get_datasets(orginal_image_dir, groundTure_image_dir, boderMasker_image_dir, train_test=''):
    images = np.empty((Nimage, height, width, channels))
    groundTures = np.empty((Nimage, height, width))
    boderMasker = np.empty((Nimage, height, width))
    files = os.listdir(orginal_image_dir)
    for i in range(len(files)):
        image = Image.open(orginal_image_dir + files[i])
        images[i] = np.asarray(image)
        print('read ' + orginal_image_dir + files[i] + 'sucessed')
        groundTure_name = files[i][0:2] + '_manual1.gif'
        g_image = Image.open(groundTure_image_dir + groundTure_name)
        groundTures[i] = np.asarray(g_image)
        print('read' + groundTure_image_dir + groundTure_name + 'sucessed')
        boderMasker_name=''
        if train_test == 'train':
            boderMasker_name = files[i][0:2] + '_training_mask.gif'
        elif train_test == 'test':
            boderMasker_name = files[i][0:2] + '_test_mask.gif'
        else:
            print('train_test must is train or test!')
        b_image = Image.open(boderMasker_image_dir + boderMasker_name)
        boderMasker[i] = np.asarray(b_image)
        print('read' + boderMasker_image_dir + boderMasker_name + 'sucessed')
    print("imgs max: " + str(np.max(images)))
    print("imgs min: " + str(np.min(images)))
    images = np.transpose(images, (0, 3, 1, 2))
    groundTures = np.reshape(groundTures, (Nimage, 1, height, width))
    boderMasker = np.reshape(boderMasker, (Nimage, 1, height, width))
    return images, groundTures, boderMasker


if __name__ == '__main__':
    data_path = './h5d_data/'
    if not os.path.exists(data_path):
        os.mkdir(data_path)
    else:
        for file in os.listdir(data_path):
            os.remove(data_path + file)
    #写训练集数据
    image, groumdTure, borderMaske = get_datasets(orginal_image_dir=original_image_train,
                                                  groundTure_image_dir=groundTure_image_train, boderMasker_image_dir=
                                                  borderMasker_image_train, train_test='train')
    write_hd5py(image, data_path + 'hdf5_images_train.hdf5')
    write_hd5py(groumdTure, data_path + 'hdf5_groundTures_train.hdf5')
    write_hd5py(borderMaske, data_path + 'hdf5_borderMasks_train.hdf5')
    #写测试集数据
    image, groumdTure, borderMaske = get_datasets(orginal_image_dir=original_image_test,
                                                  groundTure_image_dir=groundTure_image_test, boderMasker_image_dir=
                                                  borderMasker_image_test, train_test='test')
    write_hd5py(image, data_path + 'hdf5_images_test.hdf5')
    write_hd5py(groumdTure, data_path + 'hdf5_groundTures_test.hdf5')
    write_hd5py(borderMaske, data_path + 'hdf5_borderMasks_test.hdf5')
