import os
from pathlib import Path
import tensorflow as tf
from cnnClassifier.entity.config_entity import PrepareBaseModelConfig

class PrepareBaseModelMLP:
    def __init__(self, config: PrepareBaseModelConfig):
        self.config = config

    def get_base_model(self):
        input_shape = self.config.params_image_size
        self.model = tf.keras.models.Sequential([
            tf.keras.layers.Flatten(input_shape=input_shape),
            tf.keras.layers.Dense(256, activation='relu'),
            tf.keras.layers.Dropout(0.3),
            tf.keras.layers.Dense(128, activation='relu'),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Dense(self.config.params_classes, activation='softmax')
        ])

        self.model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=self.config.params_learning_rate),
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )

        self.model.summary()
        self.save_model(path=self.config.base_model_path, model=self.model)

    def update_base_model(self):
        # No freezing for MLP
        self.save_model(path=self.config.updated_base_model_path, model=self.model)

    @staticmethod
    def save_model(path: Path, model: tf.keras.Model):
        model.save(path)