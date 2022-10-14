import tensorflow as tf

from utils.srgan_tools import normalize_01, normalize_m11, denormalize_m11, get_activation, pixel_shuffle



# ---------------------------------------
#  Generator
# ---------------------------------------

def res_block(x_in, num_filters, momentum=0.8, activation='relu', batch_norm=False, return_features=True):
    
    x = tf.keras.layers.Conv2D(num_filters, kernel_size=3, padding='same')(x_in)
    
    if batch_norm:
        x = tf.keras.layers.BatchNormalization(momentum=momentum)(x)
        feats = x 
    elif return_features:
        x = AccumulateNorm()(x)
        feats = x  
        
    x = get_activation(activation)(x)
    
    x = tf.keras.layers.Conv2D(num_filters, kernel_size=3, padding='same', activation=None)(x)
    
    if batch_norm:
        x = tf.keras.layers.BatchNormalization(momentum=momentum)(x)
        
    x = tf.keras.layers.Add()([x_in, x])
    
    if return_features:
        return x, feats
    else:
        return x


def upsample(x_in, num_filters, upsample='TransposeConv', scale=2, activation='tanh'):
        
    x = x_in 
    
    if upsample == 'TransposeConv':
        x = tf.keras.layers.Conv2DTranspose(num_filters, kernel_size=scale, strides=scale, 
                                            padding='valid', output_padding=0)(x)
    elif upsample == 'Bilinear':
        x = tf.keras.layers.UpSampling2D(size=scale, interpolation='bilinear')(x)
        x = tf.keras.layers.Conv2D(num_filters, kernel_size=3, padding='same')(x)
        
    return get_activation(activation)(x)


def upsample_pix(x_in, num_filters):
    x = tf.keras.layers.Conv2D(num_filters, kernel_size=3, padding='same')(x_in)
    x = tf.keras.layers.Lambda(pixel_shuffle(scale=2))(x)
    return tf.keras.layers.PReLU(shared_axes=[1, 2])(x)


def generator(scale=4, num_filters=64, num_res_blocks=16, shape=(None,None,3), batch_size=None,
              activation='relu', batch_norm=False, upsampling='TransposeConv', return_features=True, head_only=False):
    
    if head_only:
        x_in = tf.keras.layers.Input(shape=(shape[0],shape[1],num_filters), batch_size=batch_size)
        x = upsample(x_in, num_filters=num_filters, upsample=upsampling, scale=scale//2, activation=activation)
        x = upsample(x, num_filters=shape[-1], upsample=upsampling, scale=2, activation='tanh')
    
        x = tf.keras.layers.Lambda(denormalize_m11)(x)
    
        return tf.keras.models.Model(x_in, x)
    
    features = []
    x_in = tf.keras.layers.Input(shape=shape, batch_size=batch_size)
    x = tf.keras.layers.Lambda(normalize_01)(x_in)

    x = tf.keras.layers.Conv2D(num_filters, kernel_size=9, padding='same')(x)

    if return_features:
        x = AccumulateNorm()(x)
        features.append(x)
        
    x = x_1 = get_activation(activation)(x)
        
    for _ in range(num_res_blocks):
        x, feats = res_block(x, num_filters, activation=activation, batch_norm=batch_norm, return_features=return_features)
        features.append(feats)

    x = tf.keras.layers.Conv2D(num_filters, kernel_size=3, padding='same')(x)
    
    if batch_norm:
        x = tf.keras.layers.BatchNormalization()(x)
        
    x = tf.keras.layers.Add()([x_1, x])
    
    if upsampling == 'PixelShuffle':
        x = upsample_pix(x, num_filters * 4)
        x = upsample_pix(x, num_filters * 4)
        x = tf.keras.layers.Conv2D(3, kernel_size=9, padding='same', activation='tanh')(x)
        
    else:
        x = upsample(x, num_filters=num_filters, upsample=upsampling, scale=scale//2, activation=activation)
        x = upsample(x, num_filters=shape[-1], upsample=upsampling, scale=2, activation='tanh')
    
    x = tf.keras.layers.Lambda(denormalize_m11)(x)
    
    if return_features:
        return tf.keras.models.Model(x_in, [x, tf.stack(features)])
    else:
        return tf.keras.models.Model(x_in, x)


    
# ---------------------------------------
#  Discriminator
# ---------------------------------------

def discriminator_block(x_in, num_filters, strides=1, batch_norm=True, momentum=0.8, return_features=False):
    
    x = tf.keras.layers.Conv2D(num_filters, kernel_size=(3,3), strides=strides, padding='same')(x_in)
    
    if batch_norm:
        x = tf.keras.layers.BatchNormalization(momentum=momentum)(x)
    elif return_features:
        x = AccumulateNorm()(x)
    if return_features:
        feats = x    
        
    if return_features:
        return tf.keras.layers.LeakyReLU(alpha=0.2)(x), feats
    else:
        return tf.keras.layers.LeakyReLU(alpha=0.2)(x)


def discriminator(num_filters=64, shape=(None,None,3), bottleneck='Flatten', head=512, activation='linear', big=True):
    
    x_in = tf.keras.layers.Input(shape)
    x = tf.keras.layers.Lambda(normalize_m11)(x_in)
    
    if big:
        x = discriminator_block(x, num_filters, batch_norm=False)
    x = discriminator_block(x, num_filters, strides=2, batch_norm=big)
    
    if big:
        x = discriminator_block(x, num_filters * 2)
    x = discriminator_block(x, num_filters * 2, strides=2)
    
    if big:
        x = discriminator_block(x, num_filters * 4)
    x = discriminator_block(x, num_filters * 4, strides=2)
    
    if big:
        x = discriminator_block(x, num_filters * 8)
    x = discriminator_block(x, num_filters * 8, strides=2)

    if bottleneck == 'Flatten':
        x = tf.keras.layers.Flatten()(x)
    else:
        x = tf.keras.layers.GlobalAveragePooling2D()(x)
    
    x = tf.keras.layers.Dense(head)(x)
    x = tf.keras.layers.LeakyReLU(alpha=0.2)(x)
    x = tf.keras.layers.Dense(1, activation=get_activation(activation))(x)
    
    return tf.keras.models.Model(x_in, x)


# ---------------------------------------
#  Perceptual Model (VGG)
# ---------------------------------------

def _vgg(output_layer):
    vgg = tf.keras.applications.vgg19.VGG19(input_shape=(None, None, 3), include_top=False)
    return tf.keras.models.Model(vgg.input, vgg.layers[output_layer].output)


def vgg_22():
    return _vgg(5)


def vgg_54():
    return _vgg(20)


   
# ---------------------------------------
#  Custom Layers
# ---------------------------------------

class AccumulateNorm(tf.keras.layers.Layer):
    
    def __init__(self, *args, **kwargs):
        
        super().__init__(*args, **kwargs)
        self.bn = tf.keras.layers.BatchNormalization(axis=-1,center=False, scale=False)
    
    def call(self, inputs):
        
        _ = self.bn(inputs)
        return inputs