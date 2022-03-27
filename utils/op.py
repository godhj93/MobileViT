from logging.config import valid_ident
import tensorflow as tf
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
        self._optimizer = tfa.optimizers.AdamW(learning_rate = self.LR_Scheduler(), weight_decay=0.01)
        self.CrossEntropy = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False)
        
        #Tensorboard
        self.time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        train_log_dir = 'logs/gradient_tape/' + self.time + '/train'
        test_log_dir = 'logs/gradient_tape/' + self.time + '/test'
        self.train_summary_writer = tf.summary.create_file_writer(train_log_dir)
        self.test_summary_writer = tf.summary.create_file_writer(test_log_dir)

    def LR_Scheduler(self):
        
        return LearningRateScheduler()
        
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


class LearningRateScheduler(tf.keras.optimizers.schedules.LearningRateSchedule):

        def __init__(self, initial_learning_rate=0.0002):
            

            self.initial_learning_rate = initial_learning_rate
            
            
            self.cosine_annealing = tf.keras.optimizers.schedules.CosineDecayRestarts(
                initial_learning_rate=0.002,
                first_decay_steps=3000,
                t_mul=1.0,
                m_mul=1.0,
                alpha=1e-5,
                name=None
    )

        def __call__(self, step):
            return tf.cond(step<=3000, lambda: self.linear_increasing(step) ,lambda: self.cosine_annealing(step) )
        
        def linear_increasing(self, step):
            return (0.002-0.0002)/3000*step + self.initial_learning_rate