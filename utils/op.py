from logging.config import valid_ident
import tensorflow as tf
from tqdm import tqdm
from utils.data import data_load
from tensorflow.keras.optimizers import SGD
import copy
from datetime import datetime

class Trainer:
    '''
    Train a Neural Network
    Author: H.J Shin
    Date: 2022.02.14
    '''
    def __init__(self, model, dataset='cifar10', epochs=50, batch_size= 16, size=256, DEBUG=False):
        '''
        model: model for training.
        dataset: cifar10 or cifar100.
        epochs: positive int
        batch_size: positive int
        '''
        super(Trainer, self).__init__()
        self._model = copy.deepcopy(model)
        self._epochs = epochs
        self.train_ds, self.test_ds = data_load(dataset=dataset, batch_size=batch_size, size=size, DEBUG=DEBUG)
        self._optimizer = SGD(nesterov=True, momentum=0.9, learning_rate = self.LR_Scheduler())
        self.CrossEntropy = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False)
        
        #Tensorboard
        self.time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        train_log_dir = 'logs/gradient_tape/' + self.time + '/train'
        test_log_dir = 'logs/gradient_tape/' + self.time + '/test'
        self.train_summary_writer = tf.summary.create_file_writer(train_log_dir)
        self.test_summary_writer = tf.summary.create_file_writer(test_log_dir)

    def LR_Scheduler(self):
        STEPS = len(self.train_ds) # Stpes in one epoch
        B1 = STEPS*(0.5*self._epochs)
        B2 = STEPS*(0.75*self._epochs)
        boundaries, values = [B1,B2], [1e-3,1e-4,1e-5]
        return tf.keras.optimizers.schedules.PiecewiseConstantDecay(boundaries=boundaries, values=values)
        
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
      
        for e in range(self._epochs):
            print(f"\nEPOCHS: {e+1}/{self._epochs}")
            
            train_bar = self.progress_bar('train')
            for x,y in train_bar:
                self.train_step(x,y)
                train_bar.set_description(f"Loss: {self.train_loss.result().numpy():.4f}, Acc: {self.train_accuracy.result().numpy():.4f}")
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

            self.reset_metric()
        
        print(f"Training is completed.")
        
    
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

    def save_weights(self, name):

        model_path = './models/' + name + '_' + self.time +'.h5'

        self._model.save_weights(model_path)
        print(f'the model weights has been saved in {model_path}')

    def save_model(self, name):

        model_path = './models/' + name + '_' + self.time

        self._model.save(model_path)
        print(f'the model has been saved in {model_path}')


    