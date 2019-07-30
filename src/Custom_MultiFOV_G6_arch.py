import tensorflow as tf
import tensorflow.keras as tk
import tensorflow.keras.layers as layers
from tensorflow.keras import Model



def build_segmentor(mri_shape, seg_channels=1):
    
    def S_enc_block(_input, f, k=4, strides=2, batchnorm=True, name='S_enc', activation=True, bias=False):
        initializer = tk.initializers.GlorotNormal()
        out = layers.Conv2D(f, k, strides=(strides, strides), use_bias=bias, kernel_initializer=initializer, padding='same', name=name+'_conv')(_input) 
        if batchnorm:
            out = layers.BatchNormalization(epsilon=1e-5, momentum=0.1, name=name+'_bn')(out)
        if activation:
            out = layers.LeakyReLU(name=name+'_out')(out)
        return out
    
    def S_dec_block(_input, f, k=3, strides=2, batchnorm=True, name='S_dec', activation=True, bias=False):
        initializer = tk.initializers.GlorotNormal()
        out = layers.Conv2DTranspose(f, k, strides=(strides, strides), use_bias=bias, kernel_initializer=initializer, padding='same', name=name+'_conv')(_input) 
        if batchnorm:
            out = layers.BatchNormalization(epsilon=1e-5, momentum=0.1, name=name+'_bn')(out)
        if activation:
            out = layers.ReLU(name=name+'_out')(out)
        return out
    
    
    s_mri_in = layers.Input(mri_shape)
    
    enc1_f1 = S_enc_block(s_mri_in, f=64, k=1, name='S_enc_1f1', batchnorm=False, bias=True)
    enc1_f3 = S_enc_block(s_mri_in, f=64, k=3, name='S_enc_1f3', batchnorm=False, bias=True)
    enc1_f5 = S_enc_block(s_mri_in, f=64, k=5, name='S_enc_1f5', batchnorm=False, bias=True)
    enc1 = layers.Add(name='S_enc_1')([enc1_f1, enc1_f3, enc1_f5])
    
    enc2_f1 = S_enc_block(enc1, f=128, k=1, name='S_enc_2f1')
    enc2_f3 = S_enc_block(enc1, f=128, k=3, name='S_enc_2f3')
    enc2_f5 = S_enc_block(enc1, f=128, k=5, name='S_enc_2f5')
    enc2 = layers.Add(name='S_enc_2')([enc2_f1, enc2_f3, enc2_f5])
    
    enc3_f1 = S_enc_block(enc2, f=256, k=1, name='S_enc_3f1')
    enc3_f3 = S_enc_block(enc2, f=256, k=3, name='S_enc_3f3')
    enc3_f5 = S_enc_block(enc2, f=256, k=5, name='S_enc_3f5')
    enc3 = layers.Add(name='S_enc_3')([enc3_f1, enc3_f3, enc3_f5])
    
    enc4_f1 = S_enc_block(enc3, f=512, k=1, name='S_enc_4f1')
    enc4_f3 = S_enc_block(enc3, f=512, k=3, name='S_enc_4f3')
    enc4_f5 = S_enc_block(enc3, f=512, k=5, name='S_enc_4f5')
    enc4 = layers.Add(name='S_enc_4')([enc4_f1, enc4_f3, enc4_f5])
    
    enc5_f1 = S_enc_block(enc4, f=1024, k=1, name='S_enc_5f1')
    enc5_f3 = S_enc_block(enc4, f=1024, k=3, name='S_enc_5f3')
    enc5_f5 = S_enc_block(enc4, f=1024, k=5, name='S_enc_5f5')
    enc5 = layers.Add(name='S_enc_5')([enc5_f1, enc5_f3, enc5_f5])
    
    
    dec4_f1 = S_dec_block(enc5, f=512, k=1, name='S_dec_4f1')
    dec4_f3 = S_dec_block(enc5, f=512, k=3, name='S_dec_4f3')
    dec4_f5 = S_dec_block(enc5, f=512, k=5, name='S_dec_4f5')
    dec4 = layers.Add(name='S_dec_4')([dec4_f1, dec4_f3, dec4_f5])
    skip4 = layers.Add(name='S_skip4')([dec4, enc4])
    
    
    
    dec3_f1 = S_dec_block(skip4, f=256, k=1, name='S_dec_3f1')
    dec3_f3 = S_dec_block(skip4, f=256, k=3, name='S_dec_3f3')
    dec3_f5 = S_dec_block(skip4, f=256, k=5, name='S_dec_3f5')
    dec3 = layers.Add(name='S_dec_3')([dec3_f1, dec3_f3, dec3_f5])
    
    skip3 = layers.Add(name='S_skip3')([dec3, enc3])
    
    
    dec2_f1 = S_dec_block(skip3, f=128, k=1, name='S_dec_2f1')
    dec2_f3 = S_dec_block(skip3, f=128, k=3, name='S_dec_2f3')
    dec2_f5 = S_dec_block(skip3, f=128, k=5, name='S_dec_2f5')
    dec2 = layers.Add(name='S_dec_2')([dec2_f1, dec2_f3, dec2_f5])
    skip2 = layers.Add(name='S_skip2')([dec2, enc2])
    
    dec1_f1 = S_dec_block(skip2, f=64, k=1, name='S_dec_1f1')
    dec1_f3 = S_dec_block(skip2, f=64, k=3, name='S_dec_1f3')
    dec1_f5 = S_dec_block(skip2, f=64, k=5, name='S_dec_1f5')
    dec1 = layers.Add(name='S_dec_1')([dec1_f1, dec1_f3, dec1_f5])
    skip1 = layers.Add(name='S_skip1')([dec1, enc1])
    
    out_unb = S_dec_block(skip1, f=seg_channels, name='out_unbound', batchnorm=False, activation=False, bias=False)
    out = layers.Activation('sigmoid')(out_unb)

    return Model([s_mri_in], out, name='S')

def build_critic(mri_shape, seg_shape):
    def C_enc_block(_input, f, k=4, strides=2, batchnorm=True, name='C_enc', activation=True, bias=False):
        initializer = tk.initializers.GlorotNormal()
        out = layers.Conv2D(f, k, strides=(strides, strides), use_bias=bias, kernel_initializer=initializer, padding='same', name=name+'_conv')(_input) 
        if batchnorm:
            out = layers.BatchNormalization(epsilon=1e-5, momentum=0.1, name=name+'_bn')(out)
        if activation:
            out = layers.LeakyReLU(name=name+'_out')(out)
        return out
    
    c_mri_in = layers.Input(mri_shape)
    c_seg_in = layers.Input(seg_shape)
    mri_masked = layers.Concatenate()([c_mri_in, c_seg_in])
       
    enc1_f1 = C_enc_block(mri_masked, k=1, f=64, name='C_enc_1_f1', batchnorm=False, bias=True)
    enc1_f3 = C_enc_block(mri_masked, k=3, f=64, name='C_enc_1_f3', batchnorm=False, bias=True)
    enc1_f5 = C_enc_block(mri_masked, k=5, f=64, name='C_enc_1_f5', batchnorm=False, bias=True)
    enc1 = layers.Add(name='C_enc_1')([enc1_f1, enc1_f3, enc1_f5]) 
    
    enc2_f1 = C_enc_block(enc1, f=128, k=1, name='C_enc_2_f1')
    enc2_f3 = C_enc_block(enc1, f=128, k=3, name='C_enc_2_f3')
    enc2_f5 = C_enc_block(enc1, f=128, k=5, name='C_enc_2_f5')
    enc2 = layers.Add(name='C_enc_2')([enc2_f1, enc2_f3, enc2_f5])
    
    enc3_f1 = C_enc_block(enc2, f=256, k=1, name='C_enc_3_f1')
    enc3_f3 = C_enc_block(enc2, f=256, k=3, name='C_enc_3_f3')
    enc3_f5 = C_enc_block(enc2, f=256, k=5, name='C_enc_3_f5')
    enc3 = layers.Add(name='C_enc_3')([enc3_f1, enc3_f3, enc3_f5])    
    
    enc4_f1 = C_enc_block(enc3, f=512, k=1, name='C_enc_4_f1')
    enc4_f3 = C_enc_block(enc3, f=512, k=3, name='C_enc_4_f3')
    enc4_f5 = C_enc_block(enc3, f=512, k=5, name='C_enc_4_f5')
    enc4 = layers.Add(name='C_enc_4')([enc4_f1, enc4_f3, enc4_f5])
    
    
    enc0fl = layers.Flatten()(mri_masked)
    enc1fl = layers.Flatten()(enc1)
    enc2fl = layers.Flatten()(enc2)
    enc3fl = layers.Flatten()(enc3)
    enc4fl = layers.Flatten()(enc4)
    out = layers.Concatenate()([enc0fl, enc1fl, enc2fl, enc3fl, enc4fl])
    #out = tf.stack([enc1fl, tf.tile(enc3fl, [1,4])], axis=-1)
    return Model([c_mri_in, c_seg_in], out, name='C')
    
@tf.function
def loss_d(d_real, d_fake):
    return -tf.reduce_mean(tf.abs(d_real-d_fake))

@tf.function
def smooth_dice_loss(x, y):
    ''' Dice loss continuous approximation used in SegAN paper. Converted from 
    https://github.com/YuanXue1993/SegAN/blob/master/train.py'''
    eps = 1e-6
    num = x*y
    num = tf.reduce_sum(num, axis=1)
    num = tf.reduce_sum(num, axis=1)
    den1 = x*x
    den1 = tf.reduce_sum(den1, axis=1)
    den1 = tf.reduce_sum(den1, axis=1)
    den2 = y*y
    den2 = tf.reduce_sum(den2, axis=1)
    den2 = tf.reduce_sum(den2, axis=1)
    dice=2*((num+eps)/(den1+den2+2*eps))
    dice_total=1-1*tf.reduce_sum(dice)/dice.shape[0]
    return dice_total

@tf.function
def loss_g(d_real, d_fake, y_fake, y_true):
    return tf.reduce_mean(tf.abs(d_real-d_fake)) + smooth_dice_loss(y_true, y_fake)
    