import pandas as pd
import os
from keras.preprocessing.image import ImageDataGenerator

class DataGenerator:
    """
    This class creates dataframes containing ID and the label, if it exists
    for the train, validation, and test sets. The IDs for the tets set need to
    be found by looking at the names in the test folder. It then makes tensorflow
    image generators by flowing from the dataframes that can be used to fit a model.
    """

    def __init__(self, train_path, train_labels_path, test_path, unique_identifier,
                        image_size, n_channels, target_class_column, validation_frac = 0.2,
                        train_batch_size=64, train_sample_frac=1):
        self.train_path = train_path
        self.train_labels_path = train_labels_path
        self.test_path = test_path

        self.train_data_gen = ImageDataGenerator(validation_split = validation_frac, #0.2, # Fraction of images reserved for validation
                                                rescale = 1/255,  # Normalize
                                                horizontal_flip = True, # Randomly flip the orientations for training
                                                vertical_flip = True)
        self.test_data_gen = ImageDataGenerator(rescale = 1./255,
                                                horizontal_flip = True,
                                                vertical_flip = True)

        self.image_size = image_size # (96, 96)
        self.n_channels = n_channels # 3
        self.unique_identifier = unique_identifier # 'id'
        self.target_class_column = target_class_column # 'label'
        self.train_batch_size = train_batch_size

        self.train_df = self._create_train_df(self.train_labels_path)

        # Choose random subset of the training data for faster testing of utils etc.
        if train_sample_frac < 1
            self.train_df = self.train_df.sample(train_sample_frac, random_state=1)

        self.test_df = self._create_test_df(self.test_path)
        self._create_data_generators()

    def _create_train_df(self, train_labels_path):
        train_df = self._load_target_labels(train_labels_path)
        train_df['id'] = train_df['id'].apply(lambda x: x+".tif")
        return train_df.astype({'label': 'str'})

    def _create_test_df(self, test_path):
        filenames =[]
        for dirname, _, filename in os.walk(test_path):
            filenames.extend(filename)
        return pd.DataFrame({"id":filenames})

    def _create_data_generators(self):
        pars = {'dataframe': self.train_df,
                'directory': self.train_path,
                'x_col': self.unique_identifier, # filenames of images
                'y_col': self.target_class_column, # class
                'target_size': self.image_size,
                'class_mode':'binary',
                'batch_size': self.train_batch_size}

        self.train_generator = self.train_data_gen.flow_from_dataframe(**pars, subset = 'training')
        self.validation_generator = self.train_data_gen.flow_from_dataframe(**pars,
                                                                            subset = 'validation',
                                                                            shuffle = False)
                     # Don't shuffle the validation set, metrics will be evaluated
                     # on the whole set at each epoch, and we want it in a set order
                     # for predictions after training.

        self.test_generator = self.test_data_gen.flow_from_dataframe(dataframe = self.test_df,
                                                    directory = self.test_path,
                                                    x_col = self.unique_identifier, # filename
                                                    class_mode = None,
                                                    target_size = self.image_size,
                                                    batch_size = 1,
                                                    shuffle = False) # Don't want to shuffle the test data
    def _load_target_labels(self, filename):
        return pd.read_csv(filename)
