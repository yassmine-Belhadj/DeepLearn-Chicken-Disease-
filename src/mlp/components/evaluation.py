import tensorflow as tf
from pathlib import Path
from cnnClassifier.entity.config_entity import EvaluationConfig
from cnnClassifier.utils.common import save_json

class EvaluationMLP:
    def __init__(self, config: EvaluationConfig):
        self.config = config

    def _valid_generator(self):
        datagen = tf.keras.preprocessing.image.ImageDataGenerator(
            rescale=1. / 255,
            validation_split=0.30
        )

        self.valid_generator = datagen.flow_from_directory(
            directory=self.config.training_data,
            subset="validation",
            shuffle=False,
            target_size=self.config.params_image_size[:-1],
            batch_size=self.config.params_batch_size
        )

    @staticmethod
    def load_model(path: Path):
        return tf.keras.models.load_model(path)

    def evaluation(self):
        model = self.load_model(self.config.path_of_model)
        self._valid_generator()
        self.score = model.evaluate(self.valid_generator)

    def save_score(self):
        scores = {"loss": self.score[0], "accuracy": self.score[1]}
        save_json(Path("scores_mlp.json"), scores)