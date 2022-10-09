import tensorflow as tf
import numpy as np
from tensorflow.keras import layers


def lrelu(x):
    return tf.maximum(x*0.2,x)

def identity_initializer():
    def _initializer(shape, dtype=tf.float32, partition_info=None):
        array = np.zeros(shape, dtype=float)
        cx, cy = shape[0]//2, shape[1]//2
        for i in range(np.minimum(shape[2],shape[3])):
            array[cx, cy, i, i] = 1
        return tf.constant(array, dtype=dtype)
    return _initializer

class Linear_BN(layers.Layer):
  def __init__(self):
    super(Linear_BN, self).__init__()
    w0_init = tf.random_normal_initializer()
    self.w0 = tf.Variable(initial_value=w0_init(shape=(1,),
                                              dtype='float32'),
                         trainable=True)
    w1_init = tf.zeros_initializer()
    self.w1 = tf.Variable(initial_value=w1_init(shape=(1,),
                                              dtype='float32'),
                         trainable=True)

  def call(self, x):
    return self.w0 * x[0] + self.w1*x[1]

def slim_conv2D(filters, kernel_size,rate,activation_fn,weights_initializer):
  initializer = tf.random_normal_initializer(0., 0.02)
  result = tf.keras.Sequential()
  result.add(
    tf.keras.layers.Conv2D(filters, kernel_size, strides=1,
                                    padding='same',
                                    dilation_rate=rate,
                                    activation='relu',
                                    kernel_initializer=initializer,
                                    use_bias=False))
  return result

def Full_Block(input,filters, kernel_size,rate,activation_fn,weights_initializer):
    net1=slim_conv2D(filters,kernel_size,rate,activation_fn,weights_initializer)(input)
    net1_BN=tf.keras.layers.BatchNormalization()(net1)
    final=Linear_BN()([net1,net1_BN])
    return final


def build_uCAN():
    input = tf.keras.layers.Input(shape=[None,None,3],dtype=tf.float32)
    net1=Full_Block(input,24,[3,3],rate=1,activation_fn=lrelu,weights_initializer=identity_initializer())#2
    net=Full_Block(net1,24,[3,3],rate=2,activation_fn=lrelu,weights_initializer=identity_initializer())#2
    net=Full_Block(net,24,[3,3],rate=4,activation_fn=lrelu,weights_initializer=identity_initializer())#2
    net=Full_Block(net,24,[3,3],rate=8,activation_fn=lrelu,weights_initializer=identity_initializer())#2
    net=Full_Block(net,24,[3,3],rate=16,activation_fn=lrelu,weights_initializer=identity_initializer())#2
    net=Full_Block(net,24,[3,3],rate=32,activation_fn=lrelu,weights_initializer=identity_initializer())#2
    net=Full_Block(net,24,[3,3],rate=64,activation_fn=lrelu,weights_initializer=identity_initializer())#2
    #net=Full_Block(net,24,[3,3],rate=128,activation_fn=lrelu,weights_initializer=identity_initializer())#2
    net_con = tf.keras.layers.Concatenate()([net,net1])
    net=Full_Block(net_con,24,[3,3],rate=1,activation_fn=lrelu,weights_initializer=identity_initializer())#2
    outnet=tf.keras.layers.Conv2D(1, 1, 
                                activation=None,
                                padding='same',
                                dilation_rate=1)(net)#18
    return tf.keras.Model(inputs=input, outputs=outnet)


def build_CAN():
    inp = tf.keras.layers.Input(shape=[None, None, 3],dtype=tf.float32, name='input_image')
    net=Full_Block(inp,24,[3,3],rate=1,activation_fn=lrelu,weights_initializer=identity_initializer())#2
    net=Full_Block(net,24,[3,3],rate=2,activation_fn=lrelu,weights_initializer=identity_initializer())#2
    net=Full_Block(net,24,[3,3],rate=4,activation_fn=lrelu,weights_initializer=identity_initializer())#2
    net=Full_Block(net,24,[3,3],rate=8,activation_fn=lrelu,weights_initializer=identity_initializer())#2
    net=Full_Block(net,24,[3,3],rate=16,activation_fn=lrelu,weights_initializer=identity_initializer())#2
    net=Full_Block(net,24,[3,3],rate=32,activation_fn=lrelu,weights_initializer=identity_initializer())#2
    net=Full_Block(net,24,[3,3],rate=64,activation_fn=lrelu,weights_initializer=identity_initializer())#2
    net=Full_Block(net,24,[3,3],rate=1,activation_fn=lrelu,weights_initializer=identity_initializer())#2
    outnet=tf.keras.layers.Conv2D(1, 1, 
                                activation=None,
                                padding='same',
                                dilation_rate=1)(net)#18
    return tf.keras.Model(inputs=inp, outputs=outnet)

def downsample(filters, size, apply_batchnorm=True):
  initializer = tf.random_normal_initializer(0., 0.02)

  result = tf.keras.Sequential()
  result.add(
      tf.keras.layers.Conv2D(filters, size, strides=2, padding='same',
                             kernel_initializer=initializer, use_bias=False))

  if apply_batchnorm:
    result.add(tf.keras.layers.BatchNormalization())

  result.add(tf.keras.layers.LeakyReLU())

  return result

def Discriminator():
  initializer = tf.random_normal_initializer(0., 0.02)

  inp = tf.keras.layers.Input(shape=[None, None, 3], name='input_image')
  tar = tf.keras.layers.Input(shape=[None, None, 1], name='target_image')

  x = tf.keras.layers.concatenate([inp, tar]) # (bs, 256, 256, channels*2)

  down1 = downsample(64, 4, False)(x) # (bs, 128, 128, 64)
  down2 = downsample(128, 4)(down1) # (bs, 64, 64, 128)
  down3 = downsample(256, 4)(down2) # (bs, 32, 32, 256)

  zero_pad1 = tf.keras.layers.ZeroPadding2D()(down3) # (bs, 34, 34, 256)
  conv = tf.keras.layers.Conv2D(512, 4, strides=1,
                                kernel_initializer=initializer,
                                use_bias=False)(zero_pad1) # (bs, 31, 31, 512)

  batchnorm1 = tf.keras.layers.BatchNormalization()(conv)

  leaky_relu = tf.keras.layers.LeakyReLU()(batchnorm1)

  zero_pad2 = tf.keras.layers.ZeroPadding2D()(leaky_relu) # (bs, 33, 33, 512)

  last = tf.keras.layers.Conv2D(1, 4, strides=1,
                                kernel_initializer=initializer)(zero_pad2) # (bs, 30, 30, 1)

  return tf.keras.Model(inputs=[inp, tar], outputs=last)