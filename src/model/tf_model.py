import tensorflow as tf

def convolution(l, filters):
    ksize=3
    
    l=tf.keras.layers.Conv2D(filters, kernel_size=ksize, padding='same', activation='relu')(l)
    l=tf.keras.layers.BatchNormalization()(l)
    l=tf.keras.layers.Conv2D(filters, kernel_size=ksize, padding='same', activation='relu')(l)
    l=tf.keras.layers.BatchNormalization()(l)

    return l


def unet_model(inp=(256, 256, 1), classes=1):
    inputs = tf.keras.Input(shape=inp)

    l1=convolution(inputs, 32)
    m1=tf.keras.layers.MaxPool2d()(l1) #maxpool at each step

    l2=convolution(m1, 64)
    m2=tf.keras.layers.MaxPool2d()(l2)

    l3=convolution(m2, 128)
    m3=tf.keras.layers.MaxPool2d()(l3)

    l4=convolution(m3, 256)
    m4=tf.keras.layers.MaxPool2d()(l4)

    bneck=convolution(m4, 512)      #bottleneck

    u4=tf.keras.layers.UpSampling2D()(bneck)
    u4=tf.keras.layers.Concatenate()([u4, l4])
    l5=convolution(u4, 256)     #u-path

    u3=tf.keras.layers.UpSampling2D()(l5)
    u3=tf.keras.layers.Concatenate()([u3, l3])
    l6=convolution(u3, 128)

    u2=tf.keras.layers.UpSampling2D()(l6)
    u2=tf.keras.layers.Concatenate()([u2, l2])
    l7=convolution(u2, 64)

    u1=tf.keras.layers.UpSampling2D()(l7)
    u1=tf.keras.layers.Concatenate()([u1, l1])
    l8=convolution(u3, 32)

    if classes==1:
        outputs = tf.keras.layers.Conv2D(1, 1, activation='sigmoid')(l8)
        loss = 'binary_crossentropy'
        metrics=['accuracy', tf.keras.metrics.MeanIoU(num_classes=2)]

    else:
        outputs = tf.keras.layers.Conv2D(classes, 1, activation='softmax')(l8)
        loss = 'sparse_categorical_crossentropy'
        metrics=['accuracy']

    model = tf.keras.Model(inputs, outputs)
    model.compile(optimizer=tf.keras.optimizers.Adam(1e-3), loss=loss, metrics=metrics)
    return model