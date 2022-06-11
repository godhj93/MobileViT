import tensorflow as tf
from tensorflow.keras import layers

tf.random.set_seed(777)
class MobileViT(tf.keras.Model):
    '''
    MobileViT
    https://arxiv.org/abs/2110.02178.pdf
    Author: H.J. Shin
    Date: 2022.02.12
    '''
    def __init__(self, classes=1000, arch='S'):
        '''
        classes: number of logits, default=1000(Imagenet)
        '''
        super(MobileViT, self).__init__()
        ViTArch ={
            'S':[16, 32, 64, 64, 64, 96, 144, 128, 192, 160, 240, 640],
            'XS':[16, 32, 48, 48, 48, 64, 96, 80, 120, 96, 144, 384],
            'XXS':[16, 16, 24, 24, 24, 48, 64, 64, 80, 80, 96, 320]
             }
        if arch not in ['S', 'XS', 'XXS']:
            raise ValueError("arch must be 'S', 'XS', 'XXS'")

        arch = ViTArch[arch]
        self.conv3x3 = layers.Conv2D(kernel_size= 3, filters= arch[0], strides= 2, padding= 'same')
        self.MV1_1 = InvertedResidual(strides= 1, filters= arch[1])

        self.MV2_1 = InvertedResidual(strides= 2, filters= arch[2])
        self.MV2_2 = InvertedResidual(strides= 1, filters= arch[3])
        self.MV2_3 = InvertedResidual(strides= 1, filters= arch[4])

        self.MV3_1 = InvertedResidual(strides= 2, filters= arch[5])
        self.MViT_block_1 = MViT_block(dim=arch[6], n=3, L=2)

        self.MV4_1 = InvertedResidual(strides=2, filters=arch[7])
        self.MViT_block_2 = MViT_block(dim=arch[8], n=3, L=4)

        self.MV5_1 = InvertedResidual(strides=2, filters=arch[9])
        self.MViT_block_3 = MViT_block(dim=arch[10], n=3, L=3)
        self.point_conv1 = layers.Conv2D(filters=arch[11], kernel_size=1, strides=1, activation=tf.nn.swish)
        
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
    Date: 2022.05.28
    '''
    def __init__(self, strides, filters):
        super(InvertedResidual, self).__init__()
        self.strides = strides
        self.filters = filters
    
        if self.strides not in [1,2]:
            raise ValueError('strides must be 1 or 2')

    def build(self, input_shape):

        B,H,W,C = input_shape
        expansion_factor = 4
        self.bn1 = layers.BatchNormalization()
        self.bn2 = layers.BatchNormalization()
        self.bn3 = layers.BatchNormalization()
        self.add = layers.Add()
        
        self.conv1 = layers.DepthwiseConv2D(kernel_size=3, strides=self.strides, padding='same', use_bias=False, kernel_initializer='he_normal', kernel_regularizer=tf.keras.regularizers.l2(0.001))
        self.swish = tf.nn.swish

        self.point_conv1 = layers.Conv2D(filters=C*expansion_factor, kernel_size=1, strides=1, use_bias=False, kernel_initializer='he_normal', kernel_regularizer=tf.keras.regularizers.l2(0.001))
        self.point_conv2 = layers.Conv2D(filters=self.filters, kernel_size=1, strides=1, use_bias=False, kernel_initializer='he_normal', kernel_regularizer=tf.keras.regularizers.l2(0.001))

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
    Date: 2022.06.11
    '''
    def __init__(self, dim, n=3, L=1):
        '''
        dim: Dimension to be projected
        n: convolution filter size, must be equal or bigger than the patch dimension (w,h)
        L: L stacked Encoder
        '''
        super(MViT_block, self).__init__()

        self.dim = dim
        self.L = L
        self.n = n
        self.w, self.h = 2, 2 #Patch dimension w,h: w,h <= n

    def build(self, input_shape):
        
        B, H, W, C = input_shape
        
        P = self.w * self.h
        N = H*W//P
        
        self.local_rep_conv1 = layers.Conv2D(filters=C, kernel_size=3, padding='same', use_bias=False, kernel_initializer='he_normal', kernel_regularizer=tf.keras.regularizers.l2(0.001), activation=tf.nn.swish)
        self.local_rep_conv2 = layers.Conv2D(filters=self.dim, kernel_size=1, use_bias=False, kernel_initializer='he_normal', kernel_regularizer=tf.keras.regularizers.l2(0.001), activation=tf.nn.swish)
        
        # self.reshape1 = layers.Reshape([N, P*self.dim])
        self.reshape1 = layers.Reshape([N, P, self.dim])
        
        self.get_patches = extract_patches()
        self.reconstuct = patches_to_image(H,C)
        
        self.encoders = []
        for _ in range(self.L):
            
            self.encoders.append(EncoderLayer(d_model= self.dim , num_heads=1, dff= self.dim*2 ))

        self.concat = layers.Concatenate()
        
        self.fusion_conv1 = layers.Conv2D(filters= C, kernel_size= 1, strides= 1, use_bias=False, kernel_initializer='he_normal', kernel_regularizer=tf.keras.regularizers.l2(0.001), activation= tf.nn.swish)
        self.fusion_conv2 = layers.Conv2D(filters= C, kernel_size= self.n, strides= 1, use_bias=False, kernel_initializer='he_normal', kernel_regularizer=tf.keras.regularizers.l2(0.001), padding='same', activation= tf.nn.swish)
    
    def call(self, x):
        
        B,H,W,C = x.shape
      
        #Local representations
        y = self.local_rep_conv1(x)
        y = self.local_rep_conv2(y)
        
        #Unfold
        patches = self.get_patches(y) # patches shape -> (H/self.h, W/self.w, self.dim*4)
       
        _, num_patches, patch_h , patch_w, dim_features = patches.shape
        
        y = self.reshape1(patches)
        
        for encoder in self.encoders:
            y = encoder(y)
       
        y = tf.reshape(y, (-1,num_patches, patch_h, patch_w, dim_features))
        
        y = self.reconstuct(y)
        
        y = self.fusion_conv1(y)
        y = self.concat([y,x])
        y = self.fusion_conv2(y)
        
        return y
  
'''
Ref: https://stackoverflow.com/questions/44047753/reconstructing-an-image-after-using-extract-image-patches
'''

class extract_patches(tf.keras.layers.Layer):
 
    def __init__(self):
        super(extract_patches, self).__init__()
        self.p = 2
        self.pad = [[0,0],[0,0]]
        
    def build(self, input_shape):
        
        self.reshape = layers.Reshape([(input_shape[1]//2)**2, 2,2, input_shape[-1]])
        self.h, self.c = input_shape[1], input_shape[-1]
        
    def call(self,x):
        
        self.h, self.c = x.shape[1], x.shape[-1]
        patches = tf.space_to_batch_nd(x,[self.p,self.p],self.pad)
        patches = tf.split(patches,self.p*self.p,0)
        patches = tf.stack(patches,3)
        
        patches = self.reshape(patches)
        
        return patches


class patches_to_image(tf.keras.layers.Layer):
 
    def __init__(self, H, C):
        super(patches_to_image, self).__init__()
        
        self.pad = [[0,0],[0,0]]
        self.p = 2
        self.h = H
        self.channel = C

    def build(self, input_shape):
        
        B, Num_patches, Patch_H, Patch_W, C = input_shape

        self.c = C
        self.reshape1 = layers.Reshape([self.h//self.p, self.h//self.p, self.p*self.p, self.c])
        self.reshape2 = layers.Reshape([self.p*self.p,self.h//self.p,self.h//self.p,self.c])
        self.reshape3 = layers.Reshape([self.h, self.h, self.c])
        
    def call(self, patches):

       
        patches_proc = self.reshape1(patches)
       
        patches_proc = tf.split(patches_proc,self.p*self.p,3)
        
        patches_proc = tf.stack(patches_proc,axis=1)
       
        patches_proc = self.reshape2(patches_proc)
        
        reconstructed = self.reshape3(patches_proc)
       
        return reconstructed


'''
Ref: https://www.tensorflow.org/text/tutorials/transformer#multi-head_attention
'''
class EncoderLayer(tf.keras.layers.Layer):
    
    def __init__(self,*, d_model, num_heads, dff, rate=0.1):
        super(EncoderLayer, self).__init__()

        self.mha = layers.MultiHeadAttention(key_dim=d_model, num_heads=num_heads)
        self.ffn = self.point_wise_feed_forward_network(d_model, dff)

        self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)

        self.dropout1 = tf.keras.layers.Dropout(rate)
        self.dropout2 = tf.keras.layers.Dropout(rate)

    
    def point_wise_feed_forward_network(self, d_model, dff):
        return tf.keras.Sequential([
                tf.keras.layers.Dense(dff, activation='relu'),  # (batch_size, seq_len, dff)
                tf.keras.layers.Dense(d_model)  # (batch_size, seq_len, d_model)
            ])
    def call(self, x, training):
        
        attn_output = self.mha(x, x, x)  # (batch_size, input_seq_len, d_model)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(x + attn_output)  # (batch_size, input_seq_len, d_model)

        ffn_output = self.ffn(out1)  # (batch_size, input_seq_len, d_model)
        ffn_output = self.dropout2(ffn_output, training=training)
        out2 = self.layernorm2(out1 + ffn_output)  # (batch_size, input_seq_len, d_model)

        return out2
