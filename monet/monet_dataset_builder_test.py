"""monet dataset."""

import tensorflow_datasets as tfds
import monet_dataset_builder


class MonetTest(tfds.testing.DatasetBuilderTestCase):
  """Tests for monet dataset."""

  DATASET_CLASS = monet_dataset_builder.Builder
  SPLITS = {
      'monet': 300,
      'photo': 7038, 
  }

  # If you are calling `download/download_and_extract` with a dict, like:
  #   dl_manager.download({'some_key': 'http://a.org/out.txt', ...})
  # then the tests needs to provide the fake output paths relative to the
  # fake data directory
  # DL_EXTRACT_RESULT = {'some_key': 'output_file1.txt', ...}


if __name__ == '__main__':
  tfds.testing.test_main()
