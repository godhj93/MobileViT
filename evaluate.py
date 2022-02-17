import tensorflow as tf
import argparse
from utils.data import data_load
from tensorflow.keras.losses import SparseCategoricalCrossentropy as Crossentropy
from tensorflow.keras.metrics import SparseCategoricalAccuracy as CategoricalAccuracy
gpus = tf.config.experimental.list_physical_devices("GPU")
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu,True)

parser = argparse.ArgumentParser('Evaluate MobileViT | Dataset : CIFAR 10')
parser.add_argument("--arch", default='S', type=str,help="Architecture: [S, XS, XSS]")
parser.add_argument("--model")
args = parser.parse_args()

def test():

    # model = MobileViT(arch=args.arch,classes=10).model(input_shape=(256,256,3))

    model = tf.keras.models.load_model(args.model)
    print(model.summary())

    _, test_ds = data_load(dataset='cifar10', batch_size=32, size=256)

    model.compile(
        loss = Crossentropy(),
        metrics = CategoricalAccuracy()
    )

    model.evaluate(test_ds)
    
    
if __name__ == '__main__':

    test()
    print("Done.")