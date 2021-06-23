import tensorflow as tf


def color_distortion(image, s=1.0):
    # image is a tensor with value range in [0, 1].
    # s is the strength of color distortion.

    def color_jitter(x):
        # one can also shuffle the order of following augmentations
        # each time they are applied.
        x = tf.image.random_brightness(x, max_delta=0.8 * s)
        x = tf.image.random_contrast(x, lower=1 - 0.8 * s, upper=1 + 0.8 * s)
        x = tf.image.random_saturation(x, lower=1 - 0.8 * s, upper=1 + 0.8 * s)
        x = tf.image.random_hue(x, max_delta=0.2 * s)
        x = tf.clip_by_value(x, 0, 1)
        return x

    def color_drop(x):
        x = tf.image.rgb_to_grayscale(x)
        x = tf.tile(x, [1, 1, 3])
        return x

    rand_ = tf.random.uniform(shape=(), minval=0, maxval=1)
    # randomly apply transformation with probability p.
    if rand_ < 0.8:
        image = color_jitter(image)

    rand_ = tf.random.uniform(shape=(), minval=0, maxval=1)
    if rand_ < 0.2:
        image = color_drop(image)
    return image
