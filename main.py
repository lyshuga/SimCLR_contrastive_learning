import tensorflow as tf
print(tf.__version__)
import h5py
import tensorflow_addons as tfa

# Other imports
from tensorflow.keras.layers import *
from tensorflow.keras.models import *
import matplotlib.pyplot as plt
from imutils import paths
from tqdm import tqdm
import tensorflow as tf
import numpy as np
import cv2

# Random seed fixation
tf.random.set_seed(666)
np.random.seed(666)

mean=[0.485, 0.456, 0.406]
std=[0.229, 0.224, 0.225]

# class CustomAugment(object):
#     def __init__(self, sigma=0.4):
#         super(CustomAugment, self).__init__()

#         # M from [1] & https://blog.bham.ac.uk/intellimic/g-landini-software/colour-deconvolution-2/
#         #self.M = tf.constant(np.array([[0.650, 0.704, 0.286], [0.072, 0.990, 0.105], [0.268, 0.570, 0.776]], dtype='float32'))
#         self.M = tf.constant(np.array([[0.651, 0.701, 0.290], [0.269, 0.568, 0.778], [0.633, -0.713, 0.302]], dtype='float32'))
#         self.RGB2HED = tf.linalg.inv(self.M)
#         self.sigma = sigma

#     def __call__(self, sample):        
#         # Random flips
        
#         sample = self._random_apply(tf.image.flip_left_right, sample, p=0.5)
#         # sample = self._random_apply(tf.image.flip_left_right, sample, p=0.5)
#         sample = self._random_apply(tf.image.rot90, sample, p=0.5)
#         sample = self._random_apply(tf.image.rot90, sample, p=0.5)
#         sample = self._random_apply(tf.image.rot90, sample, p=0.5)
        
#         # Randomly apply transformation (color distortions) with probability p.
#         sample = self._random_apply(self._color_jitter, sample, p=0.8)
#         sample = self._random_apply(self._color_drop, sample, p=0.4)

#         sample = self._random_apply(self._hedaugm, sample, p=0.8)


#         sample = self._random_apply(self._cutout, sample, p=0.7)


#         sample = self._random_apply(self._gaus_noise, sample, p=0.9)
#         sample = self._random_apply(self._apply_blur, sample, p=0.8)
        
#         sample = self._random_apply(self._clip, sample, p=1)
#         sample = (sample - mean) / std
        
#         return sample

#     def _gaussian_kernel(self, kernel_size, sigma, n_channels, dtype):
#         x = tf.range(-kernel_size // 2 + 1, kernel_size // 2 + 1, dtype=dtype)
#         g = tf.math.exp(-(tf.pow(x, 2) / (2 * tf.pow(tf.cast(sigma, dtype), 2))))
#         g_norm2d = tf.pow(tf.reduce_sum(g), 2)
#         g_kernel = tf.tensordot(g, g, axes=0) / g_norm2d
#         g_kernel = tf.expand_dims(g_kernel, axis=-1)
#         return tf.expand_dims(tf.tile(g_kernel, (1, 1, n_channels)), axis=-1)

#     def _apply_blur(self, sample):
#         blur = self._gaussian_kernel(5, 2, 3, sample.dtype)
#         sample = tf.nn.depthwise_conv2d(sample, blur, [1, 1, 1, 1], 'SAME')
#         return sample

#     def _gaus_noise(self, sample):
#         noise = tf.random.normal(shape=tf.shape(sample), mean=0.0, stddev=1, dtype=tf.float32)
#         sample = tf.add(sample, noise)
#         return sample

#     def _cutout(self, sample):
#         sizes = [50, 60, 80, 90]
#         size = sizes[np.random.randint(0, 4)]
#         cutout_image = tfa.image.random_cutout(sample, (size, size))
#         cutout_image = tf.squeeze(cutout_image)
#         return cutout_image

#     def _clip(self, inputs):
#         return tf.clip_by_value(inputs, clip_value_min=0.0, clip_value_max=1.)
    
#     def _hedaugm(self, inputs):
#         epsilon = 3.14159
#         input_shape = inputs.shape

#         # Reshaped images P \in R^(bs,N,3)
#         P = tf.cast(tf.reshape(inputs, [-1,input_shape[1]*input_shape[2],3]),tf.float32)
        
#         # HED images
#         S = tf.matmul(-tf.math.log(P+epsilon), self.RGB2HED)
        
#         # Channel-wise pertubations
#         alpha = tf.random.normal([tf.shape(inputs)[0],1,3], mean=1, stddev=self.sigma)
#         beta = tf.random.normal([tf.shape(inputs)[0],1,3], mean=0, stddev=self.sigma)
#         Shat = alpha*S + beta

#         # Augmented RGB images
#         Phat = tf.math.exp(-tf.matmul(Shat,self.M))-epsilon

#         # Clip values to range [0, 255]
#         Phat_clipped = Phat#tf.clip_by_value(Phat, clip_value_min=0.0, clip_value_max=1.)
# #         Phat_uint8 = tf.cast(Phat_clipped, tf.uint8)

#         return tf.reshape(Phat_clipped, [-1,input_shape[1],input_shape[2],3])

#     def _color_jitter(self, x, s=1):
#         # one can also shuffle the order of following augmentations
#         # each time they are applied.
#         x = tf.image.random_brightness(x, max_delta=0.9*s)
#         x = tf.image.random_contrast(x, lower=1-0.9*s, upper=1+0.9*s)
#         x = tf.image.random_saturation(x, lower=1-0.9*s, upper=1+0.9*s)
#         x = tf.image.random_hue(x, max_delta=0.5*s)
#         x = tf.clip_by_value(x, 0, 1)
#         return x
    
#     def _color_drop(self, x):
#         x = tf.image.rgb_to_grayscale(x)
#         x = tf.tile(x, [1, 1, 1, 3])
#         return x
    
#     def _random_apply(self, func, x, p):
#         return tf.cond(
#           tf.less(tf.random.uniform([], minval=0, maxval=1, dtype=tf.float32),
#                   tf.cast(p, tf.float32)),
#           lambda: func(x),
#           lambda: x)

class CustomAugment(object):
    def __init__(self, sigma=0.25):
        super(CustomAugment, self).__init__()

        # M from [1] & https://blog.bham.ac.uk/intellimic/g-landini-software/colour-deconvolution-2/
        # self.M = tf.constant(np.array([[0.650, 0.704, 0.286], [0.072, 0.990, 0.105], [0.268, 0.570, 0.776]], dtype='float32'))
        self.M = tf.constant(
            np.array([[0.651, 0.701, 0.290], [0.269, 0.568, 0.778], [0.633, -0.713, 0.302]], dtype='float32'))
        self.RGB2HED = tf.linalg.inv(self.M)
        self.sigma = sigma

    def __call__(self, sample):
        # Random flips
        
        #sample = self._random_apply(self._random_crop, sample, p=0.6)
        
        sample = self._random_apply(tf.image.flip_left_right, sample, p=0.5)
        
#         sample = self._random_apply(self._rotate, sample, p=0.5)
        
        sample = self._random_apply(tf.image.rot90, sample, p=0.5)
        sample = self._random_apply(tf.image.rot90, sample, p=0.5)
        sample = self._random_apply(tf.image.rot90, sample, p=0.5)

        # Randomly apply transformation (color distortions) with probability p.
        sample = self._random_apply(self._color_jitter, sample, p=0.8)
        sample = self._random_apply(self._color_drop, sample, p=0.2)

        sample = self._random_apply(self._hedaugm, sample, p=0.7)

        sample = self._random_apply(self._cutout, sample, p=0.4)

        sample = self._random_apply(self._gaus_noise, sample, p=0.7)
        sample = self._random_apply(self._apply_blur, sample, p=0.7)

        return sample
    
    def _rotate(self, sample):
        its = np.random.randint(1, 4)
        for i in range(its):
            sample = tf.image.rot90(sample)
            
        return sample
        
    
    def _gaussian_kernel(self, kernel_size, sigma, n_channels, dtype):
        x = tf.range(-kernel_size // 2 + 1, kernel_size // 2 + 1, dtype=dtype)
        g = tf.math.exp(-(tf.pow(x, 2) / (2 * tf.pow(tf.cast(sigma, dtype), 2))))
        g_norm2d = tf.pow(tf.reduce_sum(g), 2)
        g_kernel = tf.tensordot(g, g, axes=0) / g_norm2d
        g_kernel = tf.expand_dims(g_kernel, axis=-1)
        return tf.expand_dims(tf.tile(g_kernel, (1, 1, n_channels)), axis=-1)

    def _apply_blur(self, sample):
        blur = self._gaussian_kernel(3, 2, 3, sample.dtype)
        sample = tf.nn.depthwise_conv2d(sample, blur, [1, 1, 1, 1], 'SAME')
        return sample

    def _gaus_noise(self, sample):
        noise = tf.random.normal(shape=tf.shape(sample), mean=0.0, stddev=1, dtype=tf.float32)
        sample = tf.add(sample, noise)
        return sample

    def _cutout(self, sample):
        sizes = [50, 60, 80, 90]
        size = sizes[np.random.randint(0, 4)]
        cutout_image = tfa.image.random_cutout(sample, (size, size))
        cutout_image = tf.squeeze(cutout_image)
        return cutout_image
    
    def _random_crop(self, sample):
        print(sample.shape)
        sample = tf.image.random_crop(sample, size=[98, 98, 3])
        sample = tf.image.resize(sample, [112, 112])
        return sample
    
    def _hedaugm(self, inputs):
        epsilon = 3.14159
        input_shape = inputs.shape

        # Reshaped images P \in R^(bs,N,3)
        P = tf.cast(tf.reshape(inputs, [-1, input_shape[1] * input_shape[2], 3]), tf.float32)

        # HED images
        S = tf.matmul(-tf.math.log(P + epsilon), self.RGB2HED)

        # Channel-wise pertubations
        alpha = tf.random.normal([tf.shape(inputs)[0], 1, 3], mean=1, stddev=self.sigma)
        beta = tf.random.normal([tf.shape(inputs)[0], 1, 3], mean=0, stddev=self.sigma)
        Shat = alpha * S + beta

        # Augmented RGB images
        Phat = tf.math.exp(-tf.matmul(Shat, self.M)) - epsilon

        # Clip values to range [0, 255]
        Phat_clipped = tf.clip_by_value(Phat, clip_value_min=0.0, clip_value_max=255.)
        Phat_uint8 = tf.cast(Phat_clipped, tf.uint8)

        return tf.reshape(Phat_clipped, [-1, input_shape[1], input_shape[2], 3])

    def _color_jitter(self, x, s=1):
        # one can also shuffle the order of following augmentations
        # each time they are applied.
        x = tf.image.random_brightness(x, max_delta=0.8 * s)
        x = tf.image.random_contrast(x, lower=1 - 0.8 * s, upper=1 + 0.8 * s)
        x = tf.image.random_saturation(x, lower=1 - 0.8 * s, upper=1 + 0.8 * s)
        x = tf.image.random_hue(x, max_delta=0.2 * s)
        x = tf.clip_by_value(x, 0, 1)
        return x

    def _color_drop(self, x):
        x = tf.image.rgb_to_grayscale(x)
        x = tf.tile(x, [1, 1, 1, 3])
        return x

    def _random_apply(self, func, x, p):
        return tf.cond(
            tf.less(tf.random.uniform([], minval=0, maxval=1, dtype=tf.float32),
                    tf.cast(p, tf.float32)),
            lambda: func(x),
            lambda: x)

# Build the augmentation pipeline
data_augmentation = Sequential([Lambda(CustomAugment())])

# import h5py
# # Image preprocessing utils
# @tf.function
# def parse_images(tp):
#     images = h5py.File(f'/content/camelyonpatch_level_2_split_{tp}_x.h5')['x'][:]
#     image = tf.convert_to_tensor(images, dtpe='uint8')
#     image = tf.image.convert_image_dtype(image, tf.float32)
#     image = tf.image.resize(image, size=[256, 256])

#     return image

# Create TensorFlow dataset
BATCH_SIZE = 256

tp='train'
import tensorflow_io as tfio
train_images = tfio.IODataset.from_hdf5('/gpfs/workdir/shared/cpm4c/CAMELYON/prepared_datasets/pre_9_no_norm_patient_split/camelyonpatch_level_2_split_train_x.h5', dataset='/x')
import numpy as np
test_images = tfio.IODataset.from_hdf5('/gpfs/workdir/shared/cpm4c/CAMELYON/prepared_datasets/pre_9_no_norm_patient_split/camelyonpatch_level_2_split_valid_x.h5', dataset='/x')


import sys



def apply_random_crop(sample):
    thr = 0.3
    pred = np.random.uniform(low=0.0, high=1.0, size=1)[0]
    if pred < thr: 
        sample = tf.image.random_crop(sample, size=[98, 98, 3])
        sample =  tf.image.resize(sample, [112, 112])
    return sample

train_ds = train_images
# train_ds = train_ds.map(lambda x: tf.image.resize(x, [224, 224]))
# train_ds = train_ds.map(lambda x: tf.image.central_crop(x, 0.5))
# train_ds = train_ds.map(lambda x: apply_random_crop(x))

train_ds = train_ds.map(lambda x: tf.image.resize(x, [128, 128]))
train_ds = train_ds.map(lambda x: tf.image.random_crop(x, size=[112, 112, 3]))

#train_ds = train_ds.map(lambda x: tf.image.resize(x, [112, 112]))


#train_ds = train_ds.map(lambda x: tf.image.central_crop(x, 0.4375))
train_ds = train_ds.map(lambda x: tf.cast(x,tf.float32)/255.)
train_ds = train_ds.map(lambda x: (x - mean)/std)
train_ds = (
    train_ds
    .shuffle(1024)
    .batch(BATCH_SIZE, drop_remainder=True)
    .prefetch(tf.data.experimental.AUTOTUNE)
    
)

test_dataset = test_images
# test_dataset  = test_dataset.map(lambda x: tf.image.resize(x, [224, 224]))
# test_dataset  = test_dataset.map(lambda x: tf.image.central_crop(x, 0.5))
# test_dataset  = test_dataset.map(lambda x: apply_random_crop(x))

test_dataset = test_dataset.map(lambda x: tf.image.resize(x, [128, 128]))
test_dataset = test_dataset.map(lambda x: tf.image.random_crop(x, size=[112, 112, 3]))

#test_dataset  = test_dataset.map(lambda x: tf.image.central_crop(x, 0.4375))
test_dataset  = test_dataset.map(lambda x: tf.cast(x,tf.float32)/255.)
test_dataset  = test_dataset.map(lambda x: (x - mean)/std)
test_dataset  = (
    test_dataset 
    .shuffle(1024)
    .batch(BATCH_SIZE, drop_remainder=True)
    .prefetch(tf.data.experimental.AUTOTUNE)
    
)

"""## Utilities"""

#from tf2_resnets import models

#from classification_models.tfkeras import Classifiers
from tensorflow import keras

# Architecture utils
def get_resnet_simclr(hidden_1, hidden_2, hidden_3):
    #ResNet18, preprocess_input = Classifiers.get('resnet18')
    base_model = keras.models.load_model('resnet18_112.h5') #ResNet18(include_top=False, weights='imagenet', input_shape=(224, 224, 3))
    base_model.trainable = True
    inputs = Input((112, 112, 3))
    h = base_model(inputs, training=True)
    h = GlobalAveragePooling2D()(h) 

    projection_1 = Dense(hidden_1)(h)
    projection_1 = Activation("relu")(projection_1)
    projection_2 = Dense(hidden_2)(projection_1)
    #projection_2 = Activation("relu")(projection_2)
    #projection_3 = Dense(hidden_3)(projection_2)

    resnet_simclr = Model(inputs, projection_2)

    return resnet_simclr


from losses import _dot_simililarity_dim1 as sim_func_dim1, _dot_simililarity_dim2 as sim_func_dim2
import helpers

# Mask to remove positive examples from the batch of negative samples
negative_mask = helpers.get_negative_mask(BATCH_SIZE)

@tf.function
def train_step(xis, xjs, model, optimizer, criterion, temperature, do_gradient=True):
    with tf.GradientTape() as tape:
        zis = model(xis)
        zjs = model(xjs)

        # normalize projection feature vectors
        zis = tf.math.l2_normalize(zis, axis=1)
        zjs = tf.math.l2_normalize(zjs, axis=1)

        l_pos = sim_func_dim1(zis, zjs)
        l_pos = tf.reshape(l_pos, (BATCH_SIZE, 1))
        l_pos /= temperature

        negatives = tf.concat([zjs, zis], axis=0)

        loss = 0

        for positives in [zis, zjs]:
            l_neg = sim_func_dim2(positives, negatives)

            labels = tf.zeros(BATCH_SIZE, dtype=tf.int32)

            l_neg = tf.boolean_mask(l_neg, negative_mask)
            l_neg = tf.reshape(l_neg, (BATCH_SIZE, -1))
            l_neg /= temperature

            logits = tf.concat([l_pos, l_neg], axis=1) 
            loss += criterion(y_pred=logits, y_true=labels)

        loss = loss / (2 * BATCH_SIZE)
    if do_gradient:
        gradients = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))

    return loss

def train_simclr(model, dataset, optimizer, criterion,
                 temperature=0.1, epochs=100):
    step_wise_loss = []
    epoch_wise_loss = []
    test_step_wise_loss = []
    test_epoch_wise_loss = []
     
    previous_acc = 0
    
    for epoch in tqdm(range(epochs)):
        if epoch < 5:
            model.layers[1].trainable = False
        else:
            model.layers[1].trainable = True
            for i in range(55):
                model.layers[1].layers[i].trainable = False
        for image_batch in dataset:
            #print(image_batch.shape)
            #print(image_batch)
            a = data_augmentation(image_batch)
#             print(a.numpy().max())
            b = data_augmentation(image_batch)

            loss = train_step(a, b, model, optimizer, criterion, temperature)
            step_wise_loss.append(loss)

        epoch_wise_loss.append(np.mean(step_wise_loss))
        # wandb.log({"nt_xentloss": np.mean(step_wise_loss)})
        for image_batch in test_dataset:
            #print(image_batch.shape)
            #print(image_batch)
            a = data_augmentation(image_batch)
            # print(a.shape)
            b = data_augmentation(image_batch)

            loss = train_step(a, b, model, optimizer, criterion, temperature, do_gradient=False)
            test_step_wise_loss.append(loss)

        test_epoch_wise_loss.append(np.mean(step_wise_loss))

        if epoch % 1 == 0 or True:
            print("train epoch: {} loss: {:.3f}".format(epoch + 1, np.mean(step_wise_loss)))
            print("test epoch: {} loss: {:.3f}".format(epoch + 1, np.mean(test_step_wise_loss)))
            model_name = 'final2/with_frz_resize_001'
            print(model_name)
            model.save(model_name+'.h5')
            if np.mean(test_step_wise_loss) < previous_acc:
                model.save(model_name+'_test.h5')
            previous_acc = np.mean(test_step_wise_loss)
    return epoch_wise_loss, model

"""## Training"""

tf.config.experimental_run_functions_eagerly(True)

criterion = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True, 
                                                          reduction=tf.keras.losses.Reduction.SUM)

optimizer = tf.keras.optimizers.SGD(learning_rate=0.01, momentum=0.9, nesterov=True)


resnet_simclr_2 = get_resnet_simclr(256,256,256)
resnet_simclr_2.summary()

from tensorflow import keras
# resnet_simclr_2 = keras.models.load_model('112_pre9_001_001_norm.h5')



epoch_wise_loss, resnet_simclr  = train_simclr(resnet_simclr_2, train_ds, optimizer, criterion,
                 temperature=0.1, epochs=350)

#with plt.xkcd():
#    plt.plot(epoch_wise_loss)
#    plt.title("tau = 0.1, h1 = 256, h2 = 128, h3 = 50")
#    plt.show()

#resnet_simclr_2.summary()


