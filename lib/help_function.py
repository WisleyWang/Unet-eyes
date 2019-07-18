import h5py
import numpy as np
from PIL import Image
import cv2
import random

def load_hdf5(file):
    with h5py.File(file,'r') as f:
        return f['image'][:]
# image=load_hdf5('E:\Tensorlow\Project\深度学习练习\Practical Unet\h5d_data\hdf5_images_train.hdf5')
# print(image.shape)

#灰度处理
def rgb2gracy(rgb):
    assert rgb.shape[1]==3,"not 3 channels image[:,3,:,:]"
    image=rgb[:,0,:,:]*0.299 + rgb[:,1,:,:]*0.587 + rgb[:,2,:,:]*0.114
    image=np.reshape(image,(rgb.shape[0],1,rgb.shape[2],rgb.shape[3]))
    return image
#标准化
def data_normalized(image):

    image_mean=np.mean(image)
    image_std=np.std(image)
    imgs_normalized = (image - image_mean) / image_std
    for i in range(image.shape[0]):
        imgs_normalized[i] = ((imgs_normalized[i,0] - np.min(imgs_normalized[i,0])) / (
                    np.max(imgs_normalized[i,0]) - np.min(imgs_normalized[i,0]))) * 255
    return imgs_normalized

#受限的直方图均衡化clahe方法
def clahe_equalized(imgs):
    assert (len(imgs.shape)==4)  #4D arrays
    assert (imgs.shape[1]==1)  #check the channel is 1
    #create a CLAHE object (Arguments are optional).
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    imgs_equalized = np.empty(imgs.shape)
    for i in range(imgs.shape[0]):
        imgs_equalized[i,0] = clahe.apply(np.array(imgs[i,0], dtype = np.uint8))
    return imgs_equalized

def Preprocess(image_data):
    image_data=rgb2gracy(image_data)
    image_data=data_normalized(image_data)
    image_data=clahe_equalized(image_data)
    image_data=image_data/255
    return image_data

def extract_random(full_imgs,full_ground, patch_h,patch_w, N_patches, inside=True):
    if (N_patches%full_imgs.shape[0] != 0):
        print("N_patches: plase enter a multiple of 20")
        exit()
    assert (len(full_imgs.shape)==4 and len(full_ground.shape)==4)  #4D arrays
    assert (full_imgs.shape[1]==1 or full_imgs.shape[1]==3)  #check the channel is 1 or 3
    assert (full_ground.shape[1]==1)   #masks only black and white
    assert (full_imgs.shape[2] == full_ground.shape[2] and full_imgs.shape[3] == full_ground.shape[3])
    patches = np.empty((N_patches,full_imgs.shape[1],patch_h,patch_w))
    patch_groundTures = np.empty((N_patches,full_ground.shape[1],patch_h,patch_w))
    img_h = full_imgs.shape[2]  #height of the full image
    img_w = full_imgs.shape[3] #width of the full image
    # (0,0) in the center of the image
    patch_per_img = int(N_patches/full_imgs.shape[0])  #N_patches equally divided in the full images
    print("patches per full image: " +str(patch_per_img))
    iter_tot = 0   #iter over the total numbe rof patches (N_patches)
    for i in range(full_imgs.shape[0]):  #loop over the full images
        k=0
        while k <patch_per_img:
            x_center = random.randint(0+int(patch_w/2),img_w-int(patch_w/2))
            # print "x_center " +str(x_center)
            y_center = random.randint(0+int(patch_h/2),img_h-int(patch_h/2))
            # print "y_center " +str(y_center)
            if is_patch_inside_FOV(x_center, y_center, img_w, img_h, patch_h)==False:
                continue

            patch = full_imgs[i,:,y_center-int(patch_h/2):y_center+int(patch_h/2),x_center-int(patch_w/2):x_center+int(patch_w/2)]
            patch_groundTure = full_ground[i,:,y_center-int(patch_h/2):y_center+int(patch_h/2),x_center-int(patch_w/2):x_center+int(patch_w/2)]
            patches[iter_tot]=patch
            patch_groundTures[iter_tot]=patch_groundTure
            iter_tot +=1   #total
            k+=1  #per full_img
    return patches, patch_groundTures

#判断是不是中心
def is_patch_inside_FOV(x,y,img_w,img_h,patch_h):
    x_ = x - int(img_w/2) # origin (0,0) shifted to image center
    y_ = y - int(img_h/2)  # origin (0,0) shifted to image center
    R_inside = 270 - int(patch_h * np.sqrt(2.0) / 2.0) #radius is 270 (from DRIVE db docs), minus the patch diagonal (assumed it is a square #this is the limit to contain the full patch in the FOV
    radius = np.sqrt((x_*x_)+(y_*y_))
    if radius < R_inside:
        return True
    else:
        return False

#重新设定图片的大小
def get_order_patch(image_test,patch_h,patch_w):
    over_patch_h=image_test.shape[2]%patch_h
    over_patch_w=image_test.shape[3]%patch_w
    new_image=np.zeros((image_test.shape[0],image_test.shape[1],
                        image_test.shape[2]-over_patch_h+patch_h,image_test.shape[3]-over_patch_w+patch_w))
    # print(new_image.shape)
    new_image[:,:,0:image_test.shape[2],0:image_test.shape[3]]=image_test
    patchs,patch_N_h,patch_N_w=extract_ordered_overlap(new_image,patch_h,patch_w)

    return patchs,patch_N_h,patch_N_w

#按顺序裁剪
def extract_ordered_overlap(full_imgs, patch_h, patch_w):
    assert (len(full_imgs.shape)==4)  #4D arrays
    assert (full_imgs.shape[1]==1 or full_imgs.shape[1]==3)  #check the channel is 1 or 3
    img_h = full_imgs.shape[2]  #height of the full image
    img_w = full_imgs.shape[3] #width of the full image

    N_patches_img = (img_h//patch_h)*(img_w//patch_w)  #// --> division between integers
    N_patches_tot = N_patches_img*full_imgs.shape[0]
    print("Number of patches on h : " +str((img_h//patch_h)))
    print("Number of patches on w : " +str((img_w//patch_w)))
    print("number of patches per image: " +str(N_patches_img) +", totally for this dataset: " +str(N_patches_tot))
    patches = np.empty((N_patches_tot,full_imgs.shape[1],patch_h,patch_w))
    iter_tot = 0   #iter over the total number of patches (N_patches)
    for i in range(full_imgs.shape[0]):  #loop over the full images
        for h in range(img_h//patch_h):
            for w in range(img_w//patch_w):
                patch = full_imgs[i,:,h*patch_h:(h+1)*patch_h,w*patch_w:(w+1)*patch_w]
                patches[iter_tot]=patch
                iter_tot +=1   #total
    assert (iter_tot==N_patches_tot)
    return patches,(img_h//patch_h),(img_w//patch_w) #array with all the full_imgs divided in patches
                                                    #return patch_N_h and patch_N_w
#讲规则裁剪的数据还原的图片
def recoveImage(patch_image,image_n,per_h,per_w):
    all_strip=[]
    for i in range(per_h*image_n):
        strip = patch_image[i * per_w, 0]
        for j in range(i * per_w + 1, (i + 1) * per_w):
            strip = np.concatenate((strip, patch_image[j, 0]), axis=1)
        all_strip.append(strip)
    total = all_strip[0]

    for i in range(1, len(all_strip)):
        total = np.concatenate((total, all_strip[i]), axis=0)
    return total

#将得到的预测标签shape=(N,pix,2)，还原成训练批次的图片，shape=(N,1,patch_h,patch_w)
def pred_to_imgs(pred, patch_height, patch_width, mode="original"):
    assert (len(pred.shape)==3)  #3D array: (Npatches,height*width,2)
    assert (pred.shape[2]==2 )  #check the classes are 2
    pred_images = np.empty((pred.shape[0],pred.shape[1]))  #(Npatches,height*width)
    if mode=="original":
        for i in range(pred.shape[0]):
            for pix in range(pred.shape[1]):
                pred_images[i,pix]=pred[i,pix,1]
    elif mode=="threshold":
        for i in range(pred.shape[0]):
            for pix in range(pred.shape[1]):
                if pred[i,pix,1]>0.5:
                    pred_images[i,pix]=1
                else:
                    pred_images[i,pix]=0
    else:
        print("mode " +str(mode) +" not recognized, it can be 'original' or 'threshold'")
        exit()
    pred_images = np.reshape(pred_images,(pred_images.shape[0],1, patch_height, patch_width))
    return pred_images

#将标签平铺
def masks_Unet(masks):
    assert (len(masks.shape)==4)  #4D arrays
    assert (masks.shape[1]==1 )  #check the channel is 1
    im_h = masks.shape[2]
    im_w = masks.shape[3]
    masks = np.reshape(masks,(masks.shape[0],im_h*im_w))
    new_masks = np.empty((masks.shape[0],im_h*im_w,2))
    for i in range(masks.shape[0]):
        for j in range(im_h*im_w):
            if  masks[i,j] == 0:
                new_masks[i,j,0]=1
                new_masks[i,j,1]=0
            else:
                new_masks[i,j,0]=0
                new_masks[i,j,1]=1
    return new_masks