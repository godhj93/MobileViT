from multiprocessing.sharedctypes import Value
import tensorflow as tf
from tensorflow.keras.datasets import cifar10, cifar100
from imgaug.augmenters import RandAugment

rand_augmentation = RandAugment(n=3, m=7)
tf.random.set_seed(777)

class data_loader():
    def __init__(self, dataset= 'cifar10', batch_size= 16, size= 256, DEBUG=False):
        
        self.dataset=dataset
        self.batch_size = batch_size
        self.size = size
        self.DEBUG = DEBUG
        
        if self.dataset == 'cifar10':
            
            self.num_classes = 10
            (x_train, y_train), (x_test, y_test) = cifar10.load_data()
            y_train = tf.keras.utils.to_categorical(y_train, num_classes=self.num_classes)
            y_test = tf.keras.utils.to_categorical(y_test, num_classes=self.num_classes)
            self.train_dataset = (x_train,y_train)
            self.test_dataset = (x_test, y_test)
            
        elif self.dataset == 'cifar100':
            
            self.num_classes = 100
            self.train_dataset, self.test_dataset = cifar100.load_data()
            (x_train, y_train), (x_test, y_test) = cifar10.load_data()
            y_train = tf.keras.utils.to_categorical(y_train, num_classes=self.num_classes)
            y_test = tf.keras.utils.to_categorical(y_test, num_classes=self.num_classes)
            self.train_dataset = (x_train,y_train)
            self.test_dataset = (x_test, y_test)
            
        else:
            raise ValueError("Dataset must be cifar10 or cifar100.")
        
        if self.DEBUG == True:
            print(f"Debug Mode")
            self.train_dataset = self.train_dataset[:1000]
            self.test_dataset = self.test_dataset[:100]
            print(f"length of train and test data: {len(self.train_dataset), len(self.test_dataset)}")

        print(f"dataset: {self.dataset}, num of classes: {self.num_classes}")
        print(f"batch size: {self.batch_size}")
        print(f"image size: {self.size}")
        print(f"debug: {self.DEBUG}")
        
    def random_augmentation(self, images):
        images = tf.cast(images, tf.uint8)
        return rand_augmentation(images=images.numpy())

    def normalize(self, x,y):
        x = x/255.0
        
        return x,y

    def load(self):
        
        AUTO = tf.data.AUTOTUNE
        
        train_ds = (
            tf.data.Dataset.from_tensor_slices(self.train_dataset).
            batch(self.batch_size).
            shuffle(self.batch_size*100, seed=777, reshuffle_each_iteration=True).
            map(lambda x,y: (tf.image.resize(x, (self.size,self.size), method='bicubic'), y), num_parallel_calls=AUTO).
            map(lambda x,y: (tf.py_function(self.random_augmentation, [x], [tf.float32])[0], y),num_parallel_calls=AUTO).
            map(self.normalize)
            
            ).prefetch(AUTO)

        test_ds = (
            tf.data.Dataset.from_tensor_slices(self.test_dataset).
            batch(self.batch_size).
            map(lambda x,y: (tf.image.resize(x, (self.size,self.size), method='bicubic'), y), num_parallel_calls=AUTO).
            map(self.normalize)
        ).prefetch(AUTO)

        return train_ds, test_ds
