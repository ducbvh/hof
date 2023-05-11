import tensorflow as tf
from tensorflow import keras

from keras.layers import Add, LeakyReLU
from keras.layers.convolutional import Conv2D,MaxPooling2D
from keras import Model

class spatial_attention(tf.keras.layers.Layer):
    """ spatial attention module 
        
    Contains the implementation of Convolutional Block Attention Module(CBAM) block.
    As described in https://arxiv.org/abs/1807.06521.
    """
    def __init__(self, kernel_size=7, trainable=True ,**kwargs):
        self.kernel_size = kernel_size
        super(spatial_attention, self).__init__(**kwargs)

    def build(self, input_shape):
        self.conv3d = tf.keras.layers.Conv2D(filters=1, 
                                             kernel_size=self.kernel_size,
                                             strides=1, 
                                             padding='same', 
                                             activation='sigmoid',
                                             kernel_initializer='he_normal', 
                                             use_bias=False)
        super(spatial_attention, self).build(input_shape)

    def compute_output_shape(self, input_shape):
        return input_shape

    def call(self, inputs):
        avg_pool = tf.keras.layers.Lambda(lambda x: tf.keras.backend.mean(x, axis=-1, keepdims=True))(inputs)
        max_pool = tf.keras.layers.Lambda(lambda x: tf.keras.backend.max(x, axis=-1, keepdims=True))(inputs)
        concat = tf.keras.layers.Concatenate(axis=-1)([avg_pool, max_pool])
        feature = self.conv3d(concat)	
        multiplied = tf.keras.layers.Multiply()([inputs, feature])
        # shape_out = multiplied.shape   
        # return tf.keras.layers.multiply()([inputs, feature])
        return multiplied
    

class Attention_map(tf.keras.layers.Layer):
    """ 
    Attention module
    As described in: https://arxiv.org/pdf/2112.01098.pdf
    """ 
    def __init__(self,num_filters, trainable=True, **kwargs):
        self.num_filters = num_filters
        self.initializer = tf.random_normal_initializer(0., 0.02)

        self.conv4a = tf.keras.layers.Conv2D(4*self.num_filters, 
                                            kernel_size=3,
                                            padding="same", activation=LeakyReLU())
        
        self.conv8 = tf.keras.layers.Conv2D(8*self.num_filters, 
                                            kernel_size=3,
                                            padding="same", activation=LeakyReLU())
        
        self.conv4b = tf.keras.layers.Conv2D(4*self.num_filters, 
                                            kernel_size=3,
                                            padding="same", activation=LeakyReLU())
        
        self.conv2 = tf.keras.layers.Conv2D(2*self.num_filters, 
                                            kernel_size=3,
                                            padding="same", activation=LeakyReLU()) 
        
        super(Attention_map,self).__init__(**kwargs)
    
    def call(self, inputs, training, **kwargs):
        x = self.conv4a(inputs)
        x = self.conv8(x)
        x = self.conv4b(x)
        x = self.conv2(x)
        return x
    

def compute_fused(fenc, fdec, num_filter, name_layer, trainable=True):
    """
    Compute fused module \n
    f_fused = fenc*Attention_map[0] + fdec*Attention_map[1]
    """
    fconcat =  tf.concat([fenc, fdec], 0)
    output_attentionmap = Attention_map(num_filters=num_filter,trainable=trainable, name=name_layer)(fconcat)
    f_fused = fenc*output_attentionmap[0] + fdec*output_attentionmap[1]
    return f_fused

class Residual_Block(keras.layers.Layer):
    """ 
    Residual Block from ResNet:
    Input: (2H,2W,C) => Ouput: (H,W,C')
    """
    def __init__(self, num_filter, kernel_size=3, trainable=True, **kwargs):
        super(Residual_Block, self).__init__(**kwargs)
        self.filters= num_filter
        self.kernel_size = kernel_size
        self.trainable = trainable

    def build(self, input_shape):
        self.x_skip = tf.keras.layers.Conv2D(self.filters, 
                                             kernel_size=1, 
                                             padding="same")

        self.conv2a = tf.keras.layers.Conv2D(self.filters, 
                                             kernel_size=self.kernel_size, 
                                             padding='same')
        self.bn2a = tf.keras.layers.BatchNormalization()

        self.conv2b = tf.keras.layers.Conv2D(filters=self.filters, 
                                             kernel_size=self.kernel_size,
                                             padding='same')
        self.bn2b = tf.keras.layers.BatchNormalization()
        super(Residual_Block, self).build(input_shape)

    def compute_output_shape(self, input_shape):
        return input_shape
        
    def call(self, inputs):
        x_skip = self.conv2a(inputs)

        x = self.conv2a(inputs)
        x = self.bn2a(x,self.trainable)
        out1 = tf.nn.leaky_relu(x, alpha=0.2)

        out2 = self.conv2b(out1)
        out2 = self.bn2b(out2,self.trainable)
#         out2 = tf.nn.relu(out2)
        out2 = tf.nn.leaky_relu(out2, alpha=0.2)



        add = tf.keras.layers.Add()([x_skip, out2])
#         out = tf.nn.relu(add)
        out = tf.nn.leaky_relu(add, alpha=0.2)
        return out
    

class Decoder_Block(tf.keras.layers.Layer):
    """ 
    Decoder Block: Upsampling and Residual Block 
    Input: (H,W, 2C) => Ouput: (2H,2W, C)
    """
    def __init__(self,num_filter, kernel_size=3, trainable=True, **kwargs):
        super(Decoder_Block, self).__init__(**kwargs)
        self.num_filter = num_filter
        self.kernel_size = kernel_size
        self.residual = Residual_Block(num_filter=self.num_filter, 
                                       kernel_size=self.kernel_size)
        self.upsampling = tf.keras.layers.Conv2DTranspose(filters=self.num_filter,
                                                          kernel_size=2,
                                                          strides=2)
    def call(self, inputs, **kwargs):
        x = self.upsampling(inputs)
        x = self.residual(x)
        return x
  
    
class Ouput_layer(tf.keras.layers.Layer):
    """
    Output layer of model with shape (H,W,3)
    """
    def __init__(self, trainable=True, **kwargs):
        super(Ouput_layer, self).__init__(**kwargs)
        self.conv2d = tf.keras.layers.Conv2D(filters=3,
                                             kernel_size=3,
                                             padding="same")
        self.bn2d = tf.keras.layers.BatchNormalization()
    
    def call(self, inputs, training, **kwargs):
        output = self.conv2d(inputs)
        output = self.bn2d(output)
        output = tf.nn.tanh(output)
        return output
    

def Generator_model(shape=(256,256,3), train_attention=True):
    x_input = keras.Input(shape, name="Input_layer")

    encoder0 = Residual_Block(num_filter=32, kernel_size=3, name="Encoder_0")(x_input)
    encoder0 = MaxPooling2D((2,2), name="MaxPooling2D_0")(encoder0)

    encoder1 = Residual_Block(num_filter=64, kernel_size=3, name="Encoder_1")(encoder0)
    encoder1 = MaxPooling2D((2,2), name="MaxPooling2D_1")(encoder1)
    
    encoder2 = Residual_Block(num_filter=128, kernel_size=3, name="Encoder_2")(encoder1)

    #FENC
    f_enc = spatial_attention(trainable=True, name="F_enc")(encoder2)
    encoder2 = MaxPooling2D((2,2), name="MaxPooling2D_2")(encoder2)

    encoder3 = Residual_Block(num_filter=256, kernel_size=3, name="Encoder_3")(encoder2)
    encoder3 = MaxPooling2D((2,2), name="MaxPooling2D_3")(encoder3)

    encoder4 = Residual_Block(num_filter=512, kernel_size=3, name="Encoder_4")(encoder3)
    encoder4 = MaxPooling2D((2,2), name="MaxPooling2D_4")(encoder4)

    bottleneck = Conv2D(1024, 3, padding="same", name="bottleneck")(encoder4)
    
    decoder0 = Decoder_Block(num_filter=512, kernel_size=3, name="Decoder_0")(bottleneck)

    #DEC2
    decoder1 = Decoder_Block(num_filter=256, kernel_size=3, name="Decoder_1")(decoder0)
    decoder2 = Decoder_Block(num_filter=128, kernel_size=3, name="Decoder_2")(decoder1)

    #DEC 
    f_dec = spatial_attention(trainable=train_attention, name="F_dec")(decoder2)
    f_fused = compute_fused(f_enc,f_dec, num_filter=64,trainable=True, name_layer="Attention_map")

    if(train_attention==True):
        input_decoder3 = Add(name="Add_f_fused")([decoder2, f_fused])
    elif(train_attention==False):
         input_decoder3=decoder2

#     input_decoder3 = Add(name="Add_f_fused")([decoder2, f_fused])

    decoder3 = Decoder_Block(num_filter=64, kernel_size=3, name="Decoder_3")(input_decoder3)
    decoder4 = Decoder_Block(num_filter=32, kernel_size=3, name="Decoder_4")(decoder3)

    imageout = Ouput_layer(name="Ouput_layer")(decoder4)

    mymodel = Model(inputs = x_input, outputs = imageout, name = "Generator")
    return mymodel 