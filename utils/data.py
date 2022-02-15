import tensorflow as tf
from tensorflow.keras.datasets import cifar10, cifar100
from imgaug.augmenters import RandAugment

rand_augmentation = RandAugment(n=3, m=7)
tf.random.set_seed(777)
def random_augmentation(images):
    images = tf.cast(images, tf.uint8)
    return rand_augmentation(images=images.numpy())

def data_load(dataset= 'cifar10', batch_size= 16, size= 256):
    AUTO = tf.data.AUTOTUNE
    if dataset=='cifar10':
        (x_train,y_train), (x_test, y_test) = cifar10.load_data()
    elif dataset=='cifar100':
        (x_train,y_train), (x_test, y_test) = cifar100.load_data()
    else:
        raise ValueError("dataset must be cifar10 or cifar100")
    train_ds = (
        tf.data.Dataset.from_tensor_slices((x_train,y_train)).
        batch(batch_size).
        shuffle(batch_size*100, seed=777, reshuffle_each_iteration=True).
        map(lambda x,y: (tf.image.resize(x, (size,size), method='bicubic'), y), num_parallel_calls=AUTO).
        map(lambda x,y: (tf.py_function(random_augmentation, [x], [tf.float32])[0], y),num_parallel_calls=AUTO).
        map(lambda x,y: (x/255.0, y),num_parallel_calls=AUTO)
        ).prefetch(AUTO)

    test_ds = (
        tf.data.Dataset.from_tensor_slices((x_test,y_test)).
        batch(batch_size).
        map(lambda x,y: (tf.image.resize(x, (size,size), method='bicubic'), y), num_parallel_calls=AUTO).
        map(lambda x,y: (x/255.0,y), num_parallel_calls=AUTO)
    ).prefetch(AUTO)

    return train_ds, test_ds