
import tensorflow as tf


# modeling(functional API)
input_shape = (224, 224, 3)  # 논문에서 제시된 shape
x = Input(shape = input_shape, name='INPUT')

# CONV
conv1 = tf.keras.layers.Conv2D(filters=96, kernel_size=11, activation='relu', strides=4, name='CONV_1')(x)
pool1 = tf.keras.llayers.MaxPooling2D((3,3), strides=2, name='POOL_1')(conv1)  # overlapped pooling
# lrn1 = local_response_normalization(conv1,depth_radius=5, bias=2, alpha=0.0001, beta=0.75) 
lrn1 = tf.keras.llayers.BatchNormalization(name='LRN_1')(pool1)

conv2 = tf.keras.layers.Conv2D(filters=256, kernel_size=5, activation='relu', strides=1, padding='same', name='CONV_2')(lrn1)
pool2 = tf.keras.llayers.MaxPooling2D((3,3), strides=2, name='POOL_2')(conv2)
# lrn2 = local_response_normalization(conv2,depth_radius=5, bias=2,  alpha=0.0001, beta=0.75)
lrn2 = tf.keras.llayers.BatchNormalization(name='LRN_2')(pool2)

conv3 = tf.keras.layers.Conv2D(filters=384, kernel_size=3, activation='relu', strides=1, padding='same', name='CONV_3')(lrn2)
conv4 = tf.keras.layers.Conv2D(filters=384, kernel_size=3, activation='relu', strides=1, padding='same', name='CONV_4')(conv3)
conv5 = tf.keras.layers.Conv2D(filters=256, kernel_size=3, activation='relu', strides=1, padding='same', name='CONV_5')(conv4)
pool3 = tf.keras.llayers.MaxPooling2D((3,3), strides=2, name='POOL_3')(conv5)

# FC
f = tf.keras.llayers.Flatten()(pool3)
f = tf.keras.layers.Dense(4096, activation='relu', name='FC_1')(f)
f = tf.keras.llayers.Dropout(0.5)(f)  # 논문 parameter 0.5 이용
f = tf.keras.layers.Dense(4096, activation='relu', name='FC_2')(f)
f = tf.keras.llayers.Dropout(0.5)(f)
out = tf.keras.layers.Dense(1000, activation='softmax', name='OUTPUT')(f)

model = Model(inputs=x, outputs=out)
model.summary()
