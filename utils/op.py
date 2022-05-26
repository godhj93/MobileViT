from logging.config import valid_ident
import tensorflow as tf
import numpy as np
from tqdm import tqdm
from utils.data import data_load
from tensorflow.keras.optimizers import SGD
import copy
from datetime import datetime
import tensorflow_addons as tfa
class Trainer:
    '''
    Train a Neural Network
    Author: H.J Shin
    Date: 2022.05.02
    '''
    def __init__(self, model, dataset, epochs, batch_size, size, name='MODEL', DEBUG=False):
        '''
        model: model for training.
        dataset: cifar10 or cifar100.
        epochs: positive int
        batch_size: positive int
        '''
        super(Trainer, self).__init__()

        self.name = name
        self.batch_size = batch_size
        self._model = copy.deepcopy(model)
        self._epochs = epochs
        self.train_ds, self.test_ds = data_load(dataset=dataset, batch_size=batch_size, size=size, DEBUG=DEBUG)
        self._optimizer = tfa.optimizers.AdamW(learning_rate = self.LR_Scheduler(), weight_decay=0.0001)
        # self._optimizer = SGD(momentum=0.9, nesterov=True, learning_rate = self.LR_Scheduler())#, weight_decay=1e-5)
        self.CrossEntropy = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False)
        self.time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        self.save_path = './models/' + self.time[:10] + '/' + self.name + self.time[10:] + '/'
        self.ckpt_path = './ckpt/' + self.time[:10] + '/' + self.name + self.time[10:] + '/'
        #Tensorboard
        train_log_dir = 'logs/' + self.time[:10] + '/' + self.name + self.time[10:] + '/train'
        test_log_dir = 'logs/' + self.time[:10] + '/' + self.name + self.time[10:] + '/test'
        self.train_summary_writer = tf.summary.create_file_writer(train_log_dir)
        self.test_summary_writer = tf.summary.create_file_writer(test_log_dir)

    def LR_Scheduler(self):
        
        return LearningRateScheduler(initial_learning_rate=0.0002, steps=np.ceil(50000/ self.batch_size))
        
    def progress_bar(self, dataset):
        if dataset == 'train':
            return tqdm(self.train_ds, ncols=0)
        elif dataset == 'test':
            return tqdm(self.test_ds, ncols=0)
        else:
            raise ValueError("dataset must be 'train' or 'test'")

    

    
    def train(self):
        print(f"Initializing...")
        
        self.train_loss = tf.keras.metrics.Mean('train_loss', dtype=tf.float32)
        self.train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy('train_accuracy')

        self.test_loss = tf.keras.metrics.Mean('test_loss', dtype=tf.float32)
        self.test_accuracy = tf.keras.metrics.SparseCategoricalAccuracy('test_accuracy')

        best_acc = 0
        for e in range(self._epochs):
            print(f"\nEPOCH: {e+1}/{self._epochs}")
            
            train_bar = self.progress_bar('train')
            for x,y in train_bar:
                self.train_step(x,y)
                train_bar.set_description(f"Loss: {self.train_loss.result().numpy():.4f}, Acc: {self.train_accuracy.result().numpy():.4f}, Learning Rate: {self._optimizer._decayed_lr('float32').numpy()}")
            with self.train_summary_writer.as_default():
                tf.summary.scalar('loss', self.train_loss.result(), step=e)
                tf.summary.scalar('accuracy', self.train_accuracy.result(), step=e)

            test_bar = self.progress_bar('test')
            for x,y in test_bar:
                self.test_step(x,y)
                test_bar.set_description(f"Loss: {self.test_loss.result().numpy():.4f}, Acc: {self.test_accuracy.result().numpy():.4f}")
            with self.test_summary_writer.as_default():
                tf.summary.scalar('loss', self.test_loss.result(), step=e)
                tf.summary.scalar('accuracy', self.test_accuracy.result(), step=e)

            if best_acc < self.test_accuracy.result().numpy():
                self._model.save_weights(self.ckpt_path)
                print(f"The best accuracy has been updated {self.test_accuracy.result().numpy():.4f}... Save checkpoint...")
                best_acc = self.test_accuracy.result().numpy()
            self.reset_metric()
        

        
        print(f"Training is completed.")
        self.save_model()
    
    def reset_metric(self):

        self.train_loss.reset_states()
        self.test_loss.reset_states()
        self.train_accuracy.reset_states()
        self.test_accuracy.reset_states()

    @tf.function
    def train_step(self, x,y):
              
        with tf.GradientTape() as tape:
            y_hat = self._model(x, training=True)
            loss = self.CrossEntropy(y,y_hat)
        
        grads = tape.gradient(loss, self._model.trainable_variables)
        self._optimizer.apply_gradients(zip(grads, self._model.trainable_variables))
        
        self.train_accuracy.update_state(y, y_hat)
        self.train_loss.update_state(loss)
       
    @tf.function
    def test_step(self, x,y):
              
        y_hat = self._model(x, training=False)
        loss = self.CrossEntropy(y,y_hat)

        self.test_accuracy.update_state(y, y_hat)
        self.test_loss.update_state(loss)

    def save_model(self):

        self._model.save(self.save_path)
        print(f'the model has been saved in {self.save_path}')


class LearningRateScheduler(tf.keras.optimizers.schedules.LearningRateSchedule):

        def __init__(self, initial_learning_rate=0.0002,steps=3000):
            print(f"Total Steps: {steps}")
            self.steps = steps

            self.initial_learning_rate = initial_learning_rate
            
            
            self.cosine_annealing = tf.keras.optimizers.schedules.CosineDecayRestarts(
                initial_learning_rate=0.002,
                first_decay_steps= self.steps,
                t_mul=1.0,
                m_mul=1.0,
                alpha=2e-4,
                name=None
    )

        def __call__(self, step):
            return tf.cond(step<=self.steps, lambda: self.linear_increasing(step) ,lambda: self.cosine_annealing(step) )
            # return self.cosine_annealing(step)
        
        def linear_increasing(self, step):
            return (0.002-0.0002)/(self.steps)*step + self.initial_learning_rate
