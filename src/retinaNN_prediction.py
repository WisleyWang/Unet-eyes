from keras.models import *
import sys
# sys.path.append('./lib/')
sys.path.append('../lib/')
from extract_datapatches import *
from help_function import *
import matplotlib.pyplot as plt
orginal_image_dir='..\h5d_data\hdf5_images_train.hdf5'
groundTure_image_dir='..\h5d_data\hdf5_groundTures_train.hdf5'
image_test_dir='..\h5d_data\hdf5_images_test.hdf5'
groundTure_test_dir='..\h5d_data\hdf5_groundTures_test.hdf5'
if __name__ == '__main__':

    model=model_from_json(open('../test/test_architecture.json').read())
    model.summary()
    model.load_weights('../test/test_best_weights.h5')
    test_image_patch, test_ground_patch,patch_N_h,patch_N_w=get_data_test(image_test_dir, groundTure_test_dir,image_to_test=10, patch_h=48, patch_w=48)
    prediction=model.predict(test_image_patch[0:156])
    predict_img=pred_to_imgs(prediction, patch_height=48, patch_width=48, mode="original")
    predict_img=recoveImage(predict_img, image_n=1, per_h=patch_N_h, per_w=patch_N_w)
    print(predict_img.shape)
    plt.imshow(predict_img,'gray')
    plt.show()
