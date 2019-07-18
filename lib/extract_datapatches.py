from help_function import *
import matplotlib.pyplot as plt
import numpy as np
orginal_image_dir='E:\Tensorlow\Project\深度学习练习\Practical Unet\h5d_data\hdf5_images_train.hdf5'
groundTure_image_dir='E:\Tensorlow\Project\深度学习练习\Practical Unet\h5d_data\hdf5_groundTures_train.hdf5'
image_test_dir='E:\Tensorlow\Project\深度学习练习\Practical Unet\h5d_data\hdf5_images_test.hdf5'
groundTure_test_dir='E:\Tensorlow\Project\深度学习练习\Practical Unet\h5d_data\hdf5_groundTures_test.hdf5'
def get_data_train(orginal_image_dir,groundTure_image_dir,patch_h,patch_w,N_patch):
    #载入图片
    orginal_image_train=load_hdf5(orginal_image_dir)
    groundTure_image_train=load_hdf5(groundTure_image_dir)
    #d对图片进行预处理
    images=Preprocess(orginal_image_train)#标准化，均衡化，归一化 584*565
    groundTures=groundTure_image_train/255
    # images = images[:, :, 9:574, :]  # cut bottom and top so now it is 565*565
    # groundTures = groundTures[:, :, 9:574, :]  # cut bottom and top so now it is 565*565
    # 随机截取patch_size大小的图片
    patch_image_train, patch_groundTure_train = extract_random(images, groundTures,
                                                         patch_h=patch_h, patch_w=patch_w, N_patches=N_patch)
    return patch_image_train,patch_groundTure_train

def get_data_test(orginal_test_dir,groundTure_test_dir,image_to_test,patch_h,patch_w):
    orginal_test=load_hdf5(orginal_test_dir)
    groundTure_test=load_hdf5(groundTure_test_dir)

    test_ground=groundTure_test[0:image_to_test,:,:,:]
    test_ground=test_ground/255
    test_image=orginal_test[0:image_to_test,:,:]
    test_image = Preprocess(test_image)

    test_image_patch,_,_=get_order_patch(test_image,patch_h,patch_w)
    test_ground_patch,patch_N_h,patch_N_w=get_order_patch(test_ground,patch_h,patch_w)
    # print('test one image number of patch='+str(per_N_image_patch))
    # print('test one ground number of patch=' + str(per_N_ground_patch))
    return test_image_patch,test_ground_patch,patch_N_h,patch_N_w

# patch_image_train,patch_ground_train=get_data_train(orginal_image_dir,groundTure_image_dir,patch_h=48,
#                                                     patch_w=48,N_patch=1300)
# patch_image_test,patch_ground_test=get_data_test(image_test_dir,groundTure_test_dir,patch_h=48,patch_w=48)
# image=recoveImage(patch_image_test,per_h=13,per_w=12)

# print(patch_image_train.shape,patch_ground_train.shape)
# plt.subplot(121)
# plt.imshow(patch_image_train[0,0,:,:],cmap='gray')
# print(np.max(patch_image_train))
# plt.subplot(122)
# plt.imshow(patch_ground_train[0,0,:,:],cmap='gray')
# print(np.max(patch_ground_train))
#
#
# plt.show()
