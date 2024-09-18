import tensorflow as tf
import numpy as np
import tensorflow_addons as tfa

###########################################################################
#
# FUNCTIONS FOR DATA AUGMENTATION
#
###########################################################################



###################################################################################################
# FUNCTION FOR APPLYING RANDOM ERASURE
#
# INPUTS
# See comments inside function
#
# OUTPUTS
# 1) img: image with random erasing augmentation
#
def random_erasing(img, probability = 0.25, sl = 0.02, sh = 0.1, r1 = 0.3, method = 'random'):
    #Motivated by https://github.com/Amitayus/Random-Erasing-TensorFlow.git
    #Motivated by https://github.com/zhunzhong07/Random-Erasing/blob/master/transforms.py
    '''
    Class that performs Random Erasing in Random Erasing Data Augmentation by Zhong et al. 
    -------------------------------------------------------------------------------------
    img : 3D Tensor data (H,W,Channels) normalized value [0,1]
    probability: The probability that the operation will be performed.
    sl: min erasing area
    sh: max erasing area
    r1: min aspect ratio
    method : 'black', 'white' or 'random'. Erasing type
    -------------------------------------------------------------------------------------
    '''
    assert method in ['random', 'white', 'black'], 'Wrong method parameter'

    if tf.random.uniform([]) > probability:
        return img

    # HWC order
    height = tf.shape(img)[0]
    width = tf.shape(img)[1]
    channels = tf.shape(img)[2]
    area = tf.cast(width*height, tf.float32)

    target_area = tf.random.uniform([],minval=sl, maxval=sh) * area
    aspect_ratio = tf.random.uniform([],minval=r1, maxval=1/r1)
    h = tf.cast(tf.math.round(tf.math.sqrt(target_area * aspect_ratio)), tf.int32)
    w = tf.cast(tf.math.round(tf.math.sqrt(target_area / aspect_ratio)), tf.int32)

    while tf.constant(True, dtype=tf.bool):
        if h > height or w > width:
            target_area = tf.random.uniform([],minval=sl, maxval=sh) * area
            aspect_ratio = tf.random.uniform([],minval=r1, maxval=1/r1)
            h = tf.cast(tf.math.round(tf.math.sqrt(target_area * aspect_ratio)), tf.int32)
            w = tf.cast(tf.math.round(tf.math.sqrt(target_area / aspect_ratio)), tf.int32)
        else:
            break

    x1 = tf.cond(height == h, lambda:0, lambda:tf.random.uniform([], minval=0, maxval=height - h, dtype=tf.int32))
    y1 = tf.cond(width  == w, lambda:0, lambda:tf.random.uniform([], minval=0, maxval=width - w, dtype=tf.int32))
    
    part1 = tf.slice(img, [0,0,0], [x1,width,channels]) # first row
    part2 = tf.slice(img, [x1,0,0], [h,y1,channels]) # second row 1

    if method is 'black':
        part3 = tf.zeros((h,w,channels), dtype=tf.float32) # second row 2
    elif method is 'white':
        part3 = tf.ones((h,w,channels), dtype=tf.float32)
    elif method is 'random':
        part3 = tf.random.uniform((h,w,channels), dtype=tf.float32)
    
    part4 = tf.slice(img,[x1,y1+w,0], [h,width-y1-w,channels]) # second row 3
    part5 = tf.slice(img,[x1+h,0,0], [height-x1-h,width,channels]) # third row

    middle_row = tf.concat([part2,part3,part4], axis=1)
    img = tf.concat([part1,middle_row,part5], axis=0)

    return img
###################################################################################################


###################################################################################################
# FUNCTION FOR APPLYING COLOR JITTER AUGMENTATION
# APPLIES RANDOM BRIGHTNESS, CONTRAST, SATURATION, AND HUE
#
# INPUTS
# 1) image: image as TF tensor
# 2) p: float, probability of applying augmentation
# 3) s: parameter used to determine range used by augmentation methods to sample
#
# OUTPUTS
# 1) image: image with augmentation
#
def color_jitter(image, p=0.5, s=0.1):
    rand_ = tf.random.uniform(shape=(), minval=0, maxval=1)
    if rand_ < p:
        image = tf.image.random_brightness(image, max_delta=0.8 * s)
        image = tf.image.random_contrast(image, lower=1 - 0.8 * s, upper=1 + 0.8 * s)
        image = tf.image.random_saturation(image, lower=1 - 0.8 * s, upper=1 + 0.8 * s)
        #image = tf.image.random_hue(image, max_delta=0.2 * s)
        image = tf.clip_by_value(image, 0, 1)
    return image
###################################################################################################


###################################################################################################
# FUNCTION APPLYING COLOR DROP
#
# uses TF augmentation function that accepts images in float or int
# If image is float, values must be between [0,1]; if int, must be between [0, MAX]
#
# INPUTS
# 1) p: float, probability of applying augmentation
#
# OUTPUTS
# 1) image: image with augmentation
#
def color_drop(image, p=0.2):
    rand_ = tf.random.uniform(shape=(), minval=0, maxval=1)
    if rand_ < p:
        image = tf.image.rgb_to_grayscale(image)
        image = tf.tile(image, [1, 1, 3])
    return image
###################################################################################################


###################################################################################################
# FUNCTION APPLYING GAUSSIAN BLUR
#
# uses TF augmentation function
#
# INPUTS
# 1) image: image as TF tensor
# 2) p: float, probability of applying augmentation
# 3) kernel_size: int or int tuple, specifies height and width of kernel
# 4) sigma_min: float, used to determine std for gaussian filter
# 5) sigma_max: float, used to determine std for gaussian filter
#
# OUTPUTS
# 1) image: image with augmentation
#
def gaussian_blur(image, p=0.25, kernel_size = (10,10), sigma_min=0.1, sigma_max=2.0):
    rand_ = tf.random.uniform(shape=(), minval=0, maxval=1)
    if rand_ < p:
        # randomly generate std within range (sigma_min, sigma_max)
        sigma = (sigma_max - sigma_min) * np.random.random_sample() + sigma_min
        image = tfa.image.gaussian_filter2d(image, kernel_size, sigma)
        image = tf.clip_by_value(image, 0, 1)
    return image
###################################################################################################


###################################################################################################
# FUNCTION APPLYING RANDOM SATURATION
#
# uses TF augmentation function that expects pixel values to be between [0,1]
# TF function will sample a saturation factor from (lower, upper), then it will convert image to
# HSV and multiply the saturation by saturation_factor
# A lower factor results in making color more white, while a higher factor makes it more vibrant/strong
#
# INPUTS
# 1) image: image as TF tensor
# 2) p: float, probability of applying augmentation
# 3) lower: float, must be in (0 inf) range
# 4) upper: float, must be in (0, inf) range and > lower
#
# OUTPUTS
# image: image with augmentation
#
def random_saturation(image, p =0.25, lower=0.5, upper=2):
    rand_ = tf.random.uniform(shape=(), minval=0, maxval=1)
    if rand_ < p:
        image = tf.image.random_saturation(image, lower=lower, upper=upper)
        image = tf.clip_by_value(image, 0, 1)
    return image
###################################################################################################


###################################################################################################
# FUNCTION APPLYING RANDOM HUE
#
# uses TF augmentation function that expects pixel values to be between [0,1]
# TF function will sample from range (-max_delta, max_delta), then it will convert image to
# hue channel and rotate it there by sampled value
#
# INPUTS
# 1) image: image as TF tensor
# 2) p: float, probability of applying augmentation
# 3) max_delta: float, in range [0, 0.5]
#
# OUTPUTS
# 1) image: image with augmentation
#
def random_hue(image, p=0.25, max_delta=0.08):
    rand_ = tf.random.uniform(shape=(), minval=0, maxval=1)
    if rand_ < p:
        image = tf.image.random_hue(image, max_delta=max_delta)
        image = tf.clip_by_value(image, 0, 1)
    return image
###################################################################################################


###################################################################################################
# FUNCTION APPLYING RANDOM BRIGHTNESS
#
# uses TF augmentation function that expects pixel values to be between [0,1]
# TF function will sample from range (-max_delta, max_delta) and adds it to the pixels
#
# INPUTS
# 1) image: image as TF tensor
# 2) p: float, probability of applying augmentation
# 3) max_delta: float, argument for TF function
#
# OUTPUTS
# 1) image: image with augmentation
#
def random_brightness(image, p=0.25, max_delta=0.35):
    rand_ = tf.random.uniform(shape=(), minval=0, maxval=1)
    if rand_ < p:
        image = tf.image.random_brightness(image, max_delta=max_delta)
        image = tf.clip_by_value(image, 0, 1)
    return image
###################################################################################################


###################################################################################################
# FUNCTION APPLYING RANDOM CONTRAST
#
# uses TF augmentation function that expects pixel values to be between [0,1]
# TF function will sample a contrast factor from range (lower, upper)
#    and applies image = (image - mean)* contrast_factor + mean
#    where mean is calculated per channel for each image
#
# INPUTS
# 1) image: image as TF tensor
# 2) p: float, probability of applying augmentation
# 3) lower: float, lower bound for contrast factor
# 4) upper: float, upper bound for contrast factor
#
# OUTPUTS
# 1) image: image with augmentation
#
def random_contrast(image, p=0.25, lower=0.75, upper=1.5): # center bounds around 1
    rand_ = tf.random.uniform(shape=(), minval=0, maxval=1)
    if rand_ < p:
        image = tf.image.random_contrast(image, lower=lower, upper=upper)
        image = tf.clip_by_value(image, 0, 1)
    return image
###################################################################################################


###################################################################################################
# FUNCTION APPLYING RANDOM ROTATION
# 
# uses TF augmentation function that expects pixel values to be between [0,1]
# TF function rotates image by an angle in radians, counter-clockwise
#
# INPUTS
# 1) image: image as TF tensor
# 2) p: float, probability of applying augmentation
# 3) minval: float, minimum possible angle (in radians) for rotation
# 4) maxval: float, maximum possible angle (in radians) for rotation
#
# OUTPUTS
# 1) image: image with augmentation
#
def random_rotation(image, p=0.25, minval=0, maxval=2*np.pi):
    rand = tf.random.uniform(shape=(), minval=0, maxval=1)
    if rand < p:
        # randomly generate an angle within range (minval, maxval)
        angle = tf.random.uniform(shape=(), minval=minval, maxval=maxval)
        image = tfa.image.rotate(image, angle)
    return image
###################################################################################################



###################################################################################################
# FUNCTION TO APPLY RANDOM TRANSLATION


def random_translation(image, p=0.25, height_range=[0,10], width_range=[0,10]):
    rand = tf.random.uniform(shape=(), minval=0, maxval=1)
    if rand < p:
        h = tf.random.uniform(shape=(), minval=height_range[0], maxval=height_range[1])
        w = tf.random.uniform(shape=(), minval=width_range[0], maxval=width_range[1])
        image = tfa.image.translate(image, [h,w])
    return image
###################################################################################################



    
###################################################################################################
# FUNTION TO APPLY SPECIFIC OCCLUSION TO IMAGE
#
# INPUTS
# 1) image: image as TF tensor
# 2) h_range: int list, range of height to occlude; can contain more than one range
# 3) w_range: int list, range of width to occlude; can contain more than one range
# 4) p: float, probability of applying augmentation
#
# OUTPUTS
# 1) image: image with augmentation
#   
def occlude_image(image, h_range, w_range, p=0.25):
    h_range = tf.cast(h_range, tf.int32)
    w_range = tf.cast(w_range, tf.int32)
    rand_ = tf.random.uniform(shape=(), minval=0, maxval=1)
    if rand_ < p:
        # image shape
        N_H = tf.shape(image)[0]
        N_W = tf.shape(image)[1]
        N_C = tf.shape(image)[2]
        if h_range.shape[0] == 1:
            H1 = h_range[0][0]
            dH = h_range[0][1] - h_range[index][0]
            W1 = w_range[0][0]
            dW = w_range[0][1] - w_range[0][0]
             # top
            top_part = tf.slice(image, [0,0,0], [H1, N_W, N_C])
            # middle
            left_part = tf.slice(image, [H1,0,0], [dH, W1, N_C])
            right_part = tf.slice(image, [H1, W1+dW, 0], [dH, N_W - W1 - dW, N_C])
            occluded_part = tf.zeros((dH, dW, N_C))
            middle_part = tf.concat([left_part, occluded_part, right_part], axis=1)
            # bottom
            bottom_part = tf.slice(image, [H1+dH, 0, 0], [N_H-H1-dH, N_W, N_C])
            # put together
            image = tf.concat([top_part, middle_part, bottom_part], axis=0)
        else:
            index = tf.random.uniform(shape=(), minval=0, maxval=h_range.shape[0], dtype=tf.int32)
            H1 = h_range[index][0]
            dH = h_range[index][1] - h_range[index][0]
            W1 = w_range[index][0]
            dW = w_range[index][1] - w_range[index][0]
            # top
            top_part = tf.slice(image, [0,0,0], [H1, N_W, N_C])
            # middle
            left_part = tf.slice(image, [H1,0,0], [dH, W1, N_C])
            right_part = tf.slice(image, [H1, W1+dW, 0], [dH, N_W - W1 - dW, N_C])
            occluded_part = tf.zeros((dH, dW, N_C))
            middle_part = tf.concat([left_part, occluded_part, right_part], axis=1)
            # bottom
            bottom_part = tf.slice(image, [H1+dH, 0, 0], [N_H-H1-dH, N_W, N_C])
            # put together
            image = tf.concat([top_part, middle_part, bottom_part], axis=0)
    return image
###################################################################################################


###################################################################################################
# STATISTICS OF COLORS OBTAINED FROM A CASUAL SAMPLING OF FLOWERPATCH IMAGES
# CALCULATED PER CHANNEL MEANS AND VARS USING PIXELS FROM SAMPLED PATCHES
color_stats = {'red':{'mean':[0.99262745, 0.51309804, 0.28141176], 'var':[9.14356017e-05, 3.72386005e-04, 1.09407766e-03]},
               'yellow':{'mean':[0.98995098, 0.99509804, 0.50710784], 'var':[9.41344675e-05, 1.82622068e-05, 9.14492022e-04]},
               'cyan':{'mean':[0.2754902, 0.7054902, 0.7254902], 'var':[0.00207286, 0.00063806, 0.00093964]},
               'green':{'mean':[0.22839216, 0.58760784, 0.38635294], 'var':[2.58706651e-04, 9.18785083e-05, 6.52204537e-04]},
               'white':{'mean':[0.9827451,  0.98996078, 0.96015686], 'var':[4.99500192e-04, 9.72918108e-05, 8.59583237e-04]},
              }
###################################################################################################


###################################################################################################
# FUNCTION FOR GENERATING SINC MASK
#
# INPUTS
# 1) N: int, size of sinc mask
#
# OUTPUTS
# 1) sinc: float numpy array
#
def get_sinc_mask(N):

    # Create a grid of x and y values
    x = np.linspace(-.75, .75, N)
    y = np.linspace(-.75, .75, N)
    xx, yy = np.meshgrid(x, y)

    # Calculate the 2D sinc function
    sinc = np.sinc(np.sqrt(xx**2 + yy**2))
    return sinc
###################################################################################################


###################################################################################################
# FUNCTION FOR APPLYING COLOR MASK AUGMENTATION
# GENERATES ARTIFICIAL COLOR MASK
# FUNCTION ASSUMES IMAGE HAS 3 CHANNELS
#
# FUNCTION ASSUMES PIXEL VALUES ARE BETWEEN (0,1)
#
# INPUTS
# 1) image: image as TF tensor
# 2) p: float, probability of applying augmentation
# 3) size_mean: int, specifies mean for sampling color mask size from a Normal(size_mean, size_std)
# 4) size_std: int, specifies std for sampling color mask size from a Normal(size_mean, size_std)
# 5-8) H1_mean, H1_std, W1_mean, W1_std: specify the means and stds for sampling the height and width of upper left corner of color mask
#               from Normal(H1_mean, H1_std) and Normal(W1_mean, H1_std) respectively
#
# OUTPUTS
# 1) image with color mask augmentation
#
def add_color_mask(image, p=0.25, size_mean = 20, size_std=5, H1_mean=65, H1_std=5, W1_mean=65, W1_std=5):
    # determine whether to apply augmentation method
    rand_ = tf.random.uniform(shape=(), minval=0, maxval=1)
    if rand_ < p:
        # randomly sample a color
        colors = list(color_stats.keys())
        C = np.random.choice(colors)
        mean_vals, var_vals = color_stats[C]['mean'], color_stats[C]['var']
        # randomly sample a color mask size
        N = int(np.random.normal(size_mean, size_std))
        # get image dimensions
        N_H = tf.shape(image)[0]
        N_W = tf.shape(image)[1]
        N_C = tf.shape(image)[2]
        # randomly sample coordinates for upper left corner
        H1 = tf.random.normal(shape=[], mean=H1_mean, stddev=5)
        H1 = tf.cast(H1, tf.int32)
        W1 = tf.random.normal(shape=[], mean=W1_mean, stddev=5)
        W1 = tf.cast(W1, tf.int32)
        # get sinc mask to give color mask shape
        sinc_mask = get_sinc_mask(N).astype(np.float32)
        sinc_mask = tf.convert_to_tensor(sinc_mask, dtype=tf.float32)
        # get top part of image
        top_part = tf.slice(image, [0,0,0], [H1, N_W, N_C])
        # get bottom part of image
        bottom_part = tf.slice(image, [H1+N, 0, 0], [N_H-H1-N, N_W, N_C])
        # construct middle part
        # extract unchanged left and right part
        left_part = tf.slice(image, [H1,0,0], [N, W1, N_C])
        right_part = tf.slice(image, [H1, W1+N, 0], [N ,N_W-W1-N, N_C])
        #middle_part = np.zeros((3,N,N))
        # construct paint patch
        paint_stack = []
        # hard coded range val
        for k in range(3):
            # initialize color mask
            temp = tf.random.normal([N,N], mean_vals[k], np.sqrt(var_vals[k]))
            # blend color mask with original image using sinc mask
            temp = tf.math.multiply(sinc_mask, temp) + tf.math.multiply((1.0-sinc_mask),image[H1:H1+N, W1:W1+N,k])
            paint_stack.append(temp)
        # construct color_mask aptch
        paint = tf.stack(paint_stack, axis=2)
        # stitch image back together
        middle_part = tf.concat([left_part, paint, right_part], axis=1)
        image = tf.concat([top_part, middle_part, bottom_part], axis=0)
        image = tf.clip_by_value(image, 0, 1)
    return image
###################################################################################################


