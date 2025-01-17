from tensorflow import keras
import os
import matplotlib as plt
import pickle

class ModelContainer:
    """
    This class holds various models and may be used to train them and make
    predictions on a data set. It will select the best model based on the AUC of
    predictions on the validation set.
    """

    def __init__(self, models=[]):
        self.models = {}
        [self.add_model(model) for model in models]

        self.models_path = 'models'
        os.makedirs(self.models_path, exist_ok = True)

        self.best_model_name = None
        self.predictions = None
        self.roc_auc = {}
        self.val_roc_auc = {} # Validation set
        self.history = {}

    def add_model(self, model):
        # Adds a compiled model to the container
        self.models[model.name] = model

    def _get_auc(self, history, model_name):
        # For some reason the names of these keys change for multiple runs of the
        # training procedure:
        auc_label = list(history.keys())[1]
        val_auc_label = list(history.keys())[3]
        self.roc_auc[model_name] = history[auc_label][-1] # 'auc'
        self.val_roc_auc[model_name] = history[val_auc_label][-1] # 'val_auc'

    def load_model(self, model_name):
        model_path = os.path.join(self.models_path, model_name)

        # Loads a model from file and adds it to the container
        model = keras.models.load_model(os.path.join(model_path, model_name + ".h5"))
        self.add_model(model)

        # Load the training history
        with open(os.path.join(model_path, 'history.pickle'), "rb") as input_file:
             self.history[model.name] = pickle.load(input_file)

        # Load the metrics from the history
        self._get_auc(self.history[model.name], model.name)

    def save_model(self, model_name):
        # Save to model to file:
        model_dir = os.path.join(self.models_path, model_name)
        os.makedirs(model_dir, exist_ok = True)
        path_to_model = os.path.join(model_dir, model_name + ".h5")

        self.models[model_name].save(path_to_model)

        # Save the history as well as the model for plotting later if needed:
        with open(os.path.join(model_dir, 'history.pickle'), 'wb') as file_pi:
            pickle.dump(self.history[model_name], file_pi)

    def train_model(self, data, model_name, num_epochs):

        model = self.models[model_name]

        checkpoint_cb = keras.callbacks.ModelCheckpoint(os.path.join(self.models_path, model_name + ".h5"))
        early_stopping_cb = keras.callbacks.EarlyStopping(patience = 10, restore_best_weights = True, verbose = 1)

        # Reduce learning rate on plateaus:
        reduce_LR_cb = keras.callbacks.ReduceLROnPlateau(monitor='val_loss', patience = 10, verbose = 1, factor = 0.1)

        history = model.fit(data.train_generator,
                            epochs = num_epochs, #15
                            validation_data = (data.validation_generator),
                            #use_multiprocessing=True,
                            callbacks = [checkpoint_cb, reduce_LR_cb, early_stopping_cb])

        self.history[model_name] = history.history
        self._get_auc(self.history[model_name], model_name)

    def select_best_model(self):
        # Selects and saves the best model based based on AUC on validation set:
        self.best_model_name = max(self.val_roc_auc, key = self.val_roc_auc.get)

    def make_predictions(self, model_name, data_gen):
        """ Score a data set using the specified model:
         Don't save to this container because each set of predictions could be
         for a different model and a different data set."""
        return self.models[model_name].predict(data_gen, verbose = 1)

    def print_summary(self):
        print('\nModel Summaries:\n')
        for model_name in self.models.keys():
            print('\n', model_name, '- ROC AUC:', self.roc_auc[model_name])
            print('\n', model_name, '- Validation ROC AUC:', self.val_roc_auc[model_name])

        print('\nBest Model:\n', self.best_model_name)
        print('\nROC AUC of Best Model\n', self.roc_auc[self.best_model_name])

    def plot_learning_curve(self, model_name):
        history = self.history[model_name]

        fig, axs = plt.subplots(1, 2, figsize=(15, 7))

        auc_label = list(history.keys())[1]
        val_auc_label = list(history.keys())[3]

        axs[0].plot(history[auc_label])
        axs[0].plot(history[val_auc_label])
        axs[0].set_title('Model ROC AUC', fontsize = 18)
        axs[0].set_ylabel('ROC AUC', fontsize = 18)
        axs[0].set_xlabel('Epoch', fontsize = 18)
        axs[0].legend(['Train', 'Validation'], loc='upper left', fontsize = 18)

        axs[1].plot(history['loss'])
        axs[1].plot(history['val_loss'])
        axs[1].set_title('Model Loss', fontsize = 18)
        axs[1].set_ylabel('Loss', fontsize = 18)
        axs[1].set_xlabel('Epoch', fontsize = 18)
        axs[1].legend(['Train', 'Validation'], loc='upper left', fontsize = 18)
