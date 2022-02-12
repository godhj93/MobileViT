import tensorflow as tf
from utils.op import Trainer
from utils.op import save_model
import argparse
from utils.nets.MobileViT import MobileViT

gpus = tf.config.experimental.list_physical_devices("GPU")
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu,True)

parser = argparse.ArgumentParser('Train MobileViT | Dataset : CIFAR 10')
parser.add_argument("--ep", default=50, type=int,help="Epochs")
parser.add_argument("--bs", default=64, type=int,help="Batch Size")
parser.add_argument("--data", default='cifar10')
args = parser.parse_args()

def main():

    model = MobileViT().model(input_shape=(256,256,3))
    print(model.summary())
    trainer = Trainer(model, epochs=args.ep, batch_size=args.bs)
    trainer.train()
    save_model(model= model, name='MobileViT')
    
if __name__ == '__main__':

    main()
    print("Done.")