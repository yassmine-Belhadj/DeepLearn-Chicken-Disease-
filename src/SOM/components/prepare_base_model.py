import os
from pathlib import Path
import tensorflow as tf
from cnnClassifier.entity.config_entity import PrepareBaseModelConfig

class PrepareBaseModelSOM:
    def __init__(self, config: PrepareBaseModelConfig):
        self.config = config

    def get_base_model(self):
        input_shape = self.config.params_image_size

        # SOMs aren't directly supported in Keras — mock for structure
        inputs = tf.keras.Input(shape=input_shape)
        x = tf.keras.layers.Flatten()(inputs)
        x = tf.keras.layers.Dense(64, activation="relu")(x)
        outputs = tf.keras.layers.Dense(self.config.params_classes, activation="softmax")(x)

        self.model = tf.keras.Model(inputs=inputs, outputs=outputs)
        self.model.compile(
            optimizer='adam',
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )

        self.model.summary()
        self.save_model(path=self.config.base_model_path, model=self.model)

    def update_base_model(self):
        self.save_model(path=self.config.updated_base_model_path, model=self.model)

    @staticmethod
    def save_model(path: Path, model: tf.keras.Model):
        model.save(path)