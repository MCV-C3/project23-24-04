import tensorflow as tf
from tensorflow.keras.applications import InceptionResNetV2
from tensorflow.keras import layers, models, regularizers
from tensorflow.keras.utils import plot_model
from utils.label_smoothing import label_smooth

class CustomInceptionResNetV2:
    def __init__(self, num_classes, img_size, num_units, dropout, reg_coeff, label_smooth_epsilon):
        self.num_classes = num_classes
        self.img_size = img_size
        self.num_units = num_units
        self.dropout = dropout
        self.regularization_coefficient = reg_coeff
        self.label_smooth_epsilon = label_smooth_epsilon
        self.build_model()

    def build_model(self):
        # Load the Inception-ResNet V2 base model (pre-trained on ImageNet)
        self.base_model = InceptionResNetV2(input_shape=(self.img_size, self.img_size, 3), include_top=False, weights='imagenet')

        # Freeze the layers of the base model
        self.base_model.trainable = False

        # Create a new model with custom top layers
        self.model = models.Sequential([
            self.base_model,
            layers.GlobalAveragePooling2D(),
            layers.Dense(self.num_units, activation='gelu', kernel_regularizer=regularizers.l2(self.regularization_coefficient)),
            layers.Dropout(self.dropout),
            layers.Dense(self.num_classes, activation='softmax')
        ])

        if self.label_smooth_epsilon > 0:
            loss=lambda y_true, y_pred: tf.keras.losses.categorical_crossentropy(label_smooth(y_true, self.label_smooth_epsilon), y_pred)
        else: 
            loss = 'categorical_crossentropy'
        
        self.model.compile(optimizer='adam',
                           loss=loss,
                           metrics=['accuracy'])
        
        self.model.summary()

    def train(self, train_generator, validation_generator, epochs=10, callbacks=None):
        # Train the model
        history = self.model.fit(
            train_generator,
            epochs=epochs,
            validation_data=validation_generator,
            callbacks=callbacks,
            verbose=1
        )
        return history

    def save_model(self, filepath='inception_resnet_v2_model.h5'):
        # Save the trained model
        self.model.save(filepath)
