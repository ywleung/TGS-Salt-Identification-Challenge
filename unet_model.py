from keras.layers import Input, Dense, Activation, BatchNormalization, Flatten, Dropout, Conv2D, MaxPooling2D, Add, concatenate, Conv2DTranspose

# define customized block
def conv_block(x, filters, size, strides=(1, 1), padding='same', activation=True):
    x = Conv2D(filters, size, strides=strides, padding=padding)(x)
    if activation:
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
    
    return x

def residual_block(block_input, filters=16, batch_activate=False):
    x = BatchNormalization()(block_input)
    x = Activation('relu')(x)
    x = conv_block(x, filters, (3, 3), activation=True)
    x = conv_block(x, filters, (3, 3), activation=False)
    x = Add()([x, block_input])
    if batch_activate:
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        
    return x
    
    
# define U-Net model
def build_model(input_layer, start_neurons):
    # 101 -> 50
    conv1 = Conv2D(start_neurons*1, (3, 3), activation=None, padding='same')(input_layer)
    conv1 = residual_block(conv1, start_neurons*1)
    conv1 = residual_block(conv1, start_neurons*1, True)
    pool1 = MaxPooling2D((2, 2))(conv1)
    pool1 = Dropout(0.25)(pool1)
    
    # 50 -> 25
    conv2 = Conv2D(start_neurons*2, (3, 3), activation=None, padding='same')(pool1)
    conv2 = residual_block(conv2, start_neurons*2)
    conv2 = residual_block(conv2, start_neurons*2, True)
    pool2 = MaxPooling2D((2, 2))(conv2)
    pool2 = Dropout(0.5)(pool2)
    
    # 25 -> 12
    conv3 = Conv2D(start_neurons*4, (3, 3), activation=None, padding='same')(pool2)
    conv3 = residual_block(conv3, start_neurons*4)
    conv3 = residual_block(conv3, start_neurons*4, True)
    pool3 = MaxPooling2D((2, 2))(conv3)
    pool3 = Dropout(0.5)(pool3)
    
    # 12 -> 6
    conv4 = Conv2D(start_neurons*8, (3, 3), activation=None, padding='same')(pool3)
    conv4 = residual_block(conv4, start_neurons*8)
    conv4 = residual_block(conv4, start_neurons*8, True)
    pool4 = MaxPooling2D((2, 2))(conv4)
    pool4 = Dropout(0.5)(pool4)
    
    # Middle
    convm = Conv2D(start_neurons*16, (3, 3), activation=None, padding='same')(pool4)
    convm = residual_block(convm, start_neurons*16)
    convm = residual_block(convm, start_neurons*16, True)
    
    # 6 -> 12
    deconv4 = Conv2DTranspose(start_neurons*8, (3,3), strides=(2, 2), padding='same')(convm)
    uconv4 = concatenate([deconv4, conv4])
    uconv4 = Dropout(0.5)(uconv4)
    
    uconv4 = Conv2D(start_neurons*8, (3, 3), activation=None, padding='same')(uconv4)
    uconv4 = residual_block(uconv4, start_neurons*8)
    uconv4 = residual_block(uconv4, start_neurons*8, True)
    
    # 12 -> 25
    deconv3 = Conv2DTranspose(start_neurons*4, (3, 3), strides=(2, 2), padding='valid')(uconv4)
    uconv3 = concatenate([deconv3, conv3])
    uconv3 = Dropout(0.5)(uconv3)
    
    uconv3 = Conv2D(start_neurons*4, (3, 3), activation=None, padding='same')(uconv3)
    uconv3 = residual_block(uconv3, start_neurons*4)
    uconv3 = residual_block(uconv3, start_neurons*4, True)
    
    # 25 -> 50
    deconv2 = Conv2DTranspose(start_neurons*2, (3, 3), strides=(2, 2), padding='same')(uconv3)
    uconv2 = concatenate([deconv2, conv2])
    uconv2 = Dropout(0.5)(uconv2)
    
    uconv2 = Conv2D(start_neurons*2, (3, 3), activation=None, padding='same')(uconv2)
    uconv2 = residual_block(uconv2, start_neurons*2)
    uconv2 = residual_block(uconv2, start_neurons*2, True)
    
    # 50 -> 101
    deconv1 = Conv2DTranspose(start_neurons*1, (3, 3), strides=(2, 2), padding='valid')(uconv2)
    uconv1 = concatenate([deconv1, conv1])
    uconv1 = Dropout(0.5)(uconv1)
    
    uconv1 = Conv2D(start_neurons*1, (3, 3), activation=None, padding='same')(uconv1)
    uconv1 = residual_block(uconv1, start_neurons*1)
    uconv1 = residual_block(uconv1, start_neurons*1, True)
    
    output_layer_noActi = Conv2D(1, (1, 1), padding='same', activation=None)(uconv1)
    output_layer = Activation('sigmoid')(output_layer_noActi)
    
    return output_layer