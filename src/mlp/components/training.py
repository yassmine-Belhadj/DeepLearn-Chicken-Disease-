import tensorflow as tf
from pathlib import Path
from cnnClassifier.entity.config_entity import TrainingConfig

class TrainingMLP:
    def __init__(self, config: TrainingConfig):
        self.config = config

    def get_base_model(self):
        self.model = tf.keras.models.load_model(self.config.updated_base_model_path)

    def train_valid_generator(self):
        datagenerator_kwargs = dict(
            rescale=1. / 255,
            validation_split=0.20
        )

        dataflow_kwargs = dict(
            target_size=self.config.params_image_size[:-1],
            batch_size=self.config.params_batch_size,
            interpolation="nearest"
        )

        datagen = tf.keras.preprocessing.image.ImageDataGenerator(**datagenerator_kwargs)

        self.train_generator = datagen.flow_from_directory(
            directory=self.config.training_data,
            subset="training",
            shuffle=True,
            **dataflow_kwargs
        )

        self.valid_generator = datagen.flow_from_directory(
            directory=self.config.training_data,
            subset="validation",
            shuffle=False,
            **dataflow_kwargs
        )

    def train(self, callback_list: list):
        self.steps_per_epoch = self.train_generator.samples // self.train_generator.batch_size
        self.validation_steps = self.valid_generator.samples // self.valid_generator.batch_size

        self.model.fit(
            self.train_generator,
            validation_data=self.valid_generator,
            epochs=self.config.params_epochs,
            steps_per_epoch=self.steps_per_epoch,
            validation_steps=self.validation_steps,
            callbacks=callback_list
        )

        self.save_model(self.config.trained_model_path, self.model)

    @staticmethod
    def save_model(path: Path, model: tf.keras.Model):
        model.save(path)