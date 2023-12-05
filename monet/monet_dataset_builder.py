"""monet dataset."""

import tensorflow_datasets as tfds
from pathlib import Path
import tensorflow as tf
from PIL import Image
import numpy as np


class Builder(tfds.core.GeneratorBasedBuilder):
  """DatasetBuilder for monet dataset."""

  VERSION = tfds.core.Version('1.0.0')
  RELEASE_NOTES = {
      '1.0.0': 'Initial release.',
  }

  def _info(self) -> tfds.core.DatasetInfo:
    """Returns the dataset metadata."""
    return self.dataset_info_from_configs(
        features=tfds.features.FeaturesDict({
            'image': tfds.features.Image(shape=(256, 256, 3), dtype=np.uint8),
        }),
       supervised_keys=None, 
        homepage='https://www.kaggle.com/competitions/gan-getting-started',
    )

  def _split_generators(self, dl_manager: tfds.download.DownloadManager):
    """Returns SplitGenerators."""

    return {
        'photo': self._generate_examples(Path('../photo_jpg')),
        'monet': self._generate_examples(Path('../monet_jpg'))
    }

  def _generate_examples(self, path):
    """Yields examples."""
    for f in path.glob('*.jpg'):
      yield f.name, {
          'image': np.asarray(Image.open(f), dtype=np.uint8),
      }
