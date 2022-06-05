import tensorflow as tf
import argparse
from utils.data import data_loader
from tensorflow.keras.losses import CategoricalCrossentropy as Crossentropy
from tensorflow.keras.metrics import CategoricalAccuracy
gpus = tf.config.experimental.list_physical_devices("GPU")
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu,True)

from tqdm import tqdm
import time
import numpy as np

parser = argparse.ArgumentParser('Evaluate MobileViT | Dataset : CIFAR 10')
parser.add_argument("--arch", default='S', type=str,help="Architecture: [S, XS, XSS]")
parser.add_argument("--model")
parser.add_argument("--size", default=256)
parser.add_argument("--data", default='cifar10')
parser.add_argument("--device", default='gpu')
args = parser.parse_args()

def test():

    # model = MobileViT(arch=args.arch,classes=10).model(input_shape=(256,256,3))

    model = tf.keras.models.load_model(args.model)
    print(model.summary())

    
    test_data_loader = data_loader(dataset=args.data, batch_size=1, size=args.size, DEBUG=False)
    _, test_ds = test_data_loader.load()
    # model.compile(
    #     loss = Crossentropy(from_logits=False,label_smoothing=0.1),
    #     metrics = CategoricalAccuracy()
    # )

    # model.evaluate(test_ds)


    
    device = args.device
    
    
    print(f"Device: {device}")
    with tf.device(device):
            
        pbar = tqdm(test_ds)
        acc_fn = tf.keras.metrics.CategoricalAccuracy()
        latency_list = []
        for x,y in pbar:
            t0 = time.time()
            y_hat = model.predict(x)
            t1 = time.time()
            latency_list.append(t1-t0)
            acc = acc_fn(y, y_hat)

            pbar.set_description(f"acc: {acc:.4f}, latency: {np.mean(latency_list)*1000:.4f}ms")
            

    
if __name__ == '__main__':

    test()
    print("Done.")
