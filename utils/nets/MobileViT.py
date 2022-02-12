from pickletools import string1
from turtle import st
import tensorflow as tf
from tensorflow.keras import layers


class MobileViT(tf.keras.Model):
    '''
    MobileViT
    https://arxiv.org/abs/2110.02178.pdf
    Author: H.J. Shin
    Date: 2022.02.12
    '''
    def __init__(self, classes=1000):
        '''
        classes: number of logits, default=1000(Imagenet)
        '''
        super(MobileViT, self).__init__()
    
        self.conv3x3 = layers.Conv2D(kernel_size= 3, filters= 16, strides= 2, padding= 'same')
        self.MV1_1 = InvertedResidual(strides= 1, filters= 32)

        self.MV2_1 = InvertedResidual(strides= 2, filters= 64)
        self.MV2_2 = InvertedResidual(strides= 1, filters= 64)
        self.MV2_3 = InvertedResidual(strides= 1, filters= 64)

        self.MV3_1 = InvertedResidual(strides= 2, filters= 96)
        self.MViT_block_1 = MViT_block(dim=144, n=3, L=2)

        self.MV4_1 = InvertedResidual(strides=2, filters=128)
        self.MViT_block_2 = MViT_block(dim=192, n=3, L=4)

        self.MV5_1 = InvertedResidual(strides=2, filters=160)
        self.MViT_block_3 = MViT_block(dim=240, n=3, L=3)
        self.point_conv1 = layers.Conv2D(filters=640, kernel_size=1, strides=1, activation=tf.nn.swish)
        
        self.global_pool = layers.GlobalAveragePooling2D()
        self.logits = layers.Dense(classes, activation = tf.nn.softmax)

    def call(self, x):
        
        y = self.conv3x3(x)

        y = self.MV1_1(y)

        y = self.MV2_1(y)

        y = self.MV2_2(y)

        y = self.MV2_3(y)        

        y = self.MV3_1(y)

        y = self.MViT_block_1(y)

        y = self.MV4_1(y)

        y = self.MViT_block_2(y)

        y = self.MV5_1(y)

        y = self.MViT_block_3(y)

        y = self.point_conv1(y)

        y = self.global_pool(y)

        return self.logits(y)
    
    def model(self, input_shape):
        '''
        This method makes the command "model.summary()" work.
        input_shape: (H,W,C), do not specify batch B
        '''
        x = layers.Input(shape=input_shape)
        return tf.keras.Model(inputs=[x], outputs=self.call(x))

class InvertedResidual(tf.keras.layers.Layer):
    '''
    Inverted Residual Block
    https://arxiv.org/pdf/1801.04381.pdf
    Author: H.J. Shin
    Date: 2022.02.12
    '''
    def __init__(self, strides, filters):
        super(InvertedResidual, self).__init__()
        self.strides = strides
        self.filters = filters
        self.bn1 = layers.BatchNormalization()
        self.bn2 = layers.BatchNormalization()
        self.bn3 = layers.BatchNormalization()
        self.add = layers.Add()
    
        if self.strides not in [1,2]:
            raise ValueError('strides must be 1 or 2')

        self.conv1 = layers.DepthwiseConv2D(kernel_size=3, strides=self.strides, padding='same', use_bias=False)
        self.swish = tf.nn.swish

    def build(self, input_shape):

        B,H,W,C = input_shape
        self.point_conv1 = layers.Conv2D(filters=C, kernel_size=1, strides=1)
        self.point_conv2 = layers.Conv2D(filters=self.filters, kernel_size=1, strides=1)

    def call(self, x):

        y = self.swish( self.bn1( self.point_conv1(x)))

        y = self.swish( self.bn2( self.conv1(y)))
        
        y = self.bn3 ( self.point_conv2(y))
        
        if self.strides == 1 and (x.shape == y.shape):
            return self.add([x, y])
        return y

class MViT_block(tf.keras.layers.Layer):
    '''
    MobileViT Block
    https://arxiv.org/abs/2110.02178.pdf
    Author: H.J. Shin
    Date: 2022.02.12
    '''
    def __init__(self, dim, n=3, L=1):
        '''
        dim: Dimension to be projected
        n: convolution filter size, must be equal or bigger than the patch dimension (w,h)
        L: L stacked Encoder
        '''
        super(MViT_block, self).__init__()

        self.p_dim = dim
        self.L = L
        self.n = n
        self.w, self.h = 2, 2 #Patch dimension w,h: w,h <= n

    def build(self, input_shape):
        
        B, H, W, C = input_shape
        P = self.w * self.h
        N = H*W//P
        
        self.local_rep_conv1 = layers.Conv2D(filters=self.p_dim, kernel_size=3, padding='same', use_bias=False, activation=tf.nn.swish)
        self.local_rep_conv2 = layers.Conv2D(filters=self.p_dim, kernel_size=1, use_bias=False, activation=tf.nn.swish)
        #output : H W self.p_dim

        self.reshape = layers.Reshape((N,P,self.p_dim))

        self.flatten = layers.Flatten()
        
        self.encoders = []
        for i in range(self.L):
            encoder_name = 'Transformer_encoder_%d'%i
            self.encoders.append(T_encoder(num_heads=2, project_dim=self.p_dim, name=encoder_name))

        self.concat = layers.Concatenate()
        
        
        self.reshape2 = layers.Reshape((H,W,self.p_dim))

        self.point_conv = layers.Conv2D(filters= C, kernel_size= 1, strides= 1, use_bias= False, activation= tf.nn.swish)
        self.conv = layers.Conv2D(filters= C, kernel_size= self.n, strides= 1, use_bias= False, padding='same', activation= tf.nn.swish)

    def call(self, x):
        
        #Local representations
        y = self.local_rep_conv1(x)
        y = self.local_rep_conv2(y)
        #####
        # Transformers as Convolutions(global representations)
        #   Unfold
        y = self.reshape(y)
        #   Transformer Encoder
        for i in range(self.L): 
            y = self.encoders[i](y)
        #   Fold
        y = self.reshape2(y)
        #####

        #Fusions
        y = self.point_conv(y)
        y = self.concat([y,x])
        
        return self.conv(y)
        

class T_encoder(tf.keras.layers.Layer):
    '''
    Transformer Encoder
    https://arxiv.org/pdf/2010.11929.pdf
    Author: H.J. Shin
    Date: 2022.02.12
    '''
    def __init__(self, num_heads, project_dim, name):
        super(T_encoder, self).__init__(name=name)
        self.p_dim = project_dim
        self.num_heads = num_heads
        self.norm1 = layers.LayerNormalization()
        self.norm2 = layers.LayerNormalization()
        
        self.add = layers.Add()

    def build(self, input_shape):
        
        B,H,W,C = input_shape
        self.MHA = layers.MultiHeadAttention(num_heads= self.num_heads, key_dim= self.p_dim, value_dim=None, use_bias=False)
        self.MLP = layers.Dense(C, activation=tf.nn.swish, use_bias=False)

    def call(self, x):
        
        y = self.norm1(x)
    
        y = self.MHA(y,y)
    
        residual = self.add([x,y])
    
        y = self.norm2(residual)
    
        
        y = self.MLP(y)
    

        return self.add([residual, y])


        

