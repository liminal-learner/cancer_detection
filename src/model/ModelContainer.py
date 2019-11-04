from tensorflow import keras
import os

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

    def add_model(self, model):
        # Adds a compiled model to the container
        self.models[model.name] = model

    def load_model(self, model_path):
        # Loads a model from file and adds it to the container
        model = keras.models.load_model(model_path)
        self.add_model(model)

    def train_model(self, data, model_name):

        model = self.models[model_name]

        checkpoint_cb = keras.callbacks.ModelCheckpoint(os.path.join(self.models_path, model_name + ".h5"))
        early_stopping_cb = keras.callbacks.EarlyStopping(patience = 10, restore_best_weights = True, verbose = 1)

        # Reduce learning rate on plateaus:
        reduce_LR_cb = keras.callbacks.ReduceLROnPlateau(monitor='val_loss', patience = 10, verbose = 1, factor = 0.1)

        history = model.fit(data.train_generator,
                            epochs = 15,
                            validation_data = (data.validation_generator),
                            #use_multiprocessing=True,
                            callbacks = [checkpoint_cb, reduce_LR_cb, early_stopping_cb])

        # For some reason the names of these keys change for multiple runs of the
        # training procedure:
        self.roc_auc[model.name] = history.history[list(history.history.keys())[1]][-1] # 'auc'
        self.val_roc_auc[model.name] = history.history[list(history.history.keys())[3]][-1] # 'val_auc'

    def save_model(self, model_name):
        # Save to model to file:
        path_to_model = os.path.join(self.models_path, model_name + ".h5")
        self.models[model_name].save(path_to_model)

    def select_best_model(self):
        # Selects and saves the best model based based on AUC on validation set:
        self.best_model_name = max(self.val_roc_auc, key = self.val_roc_auc.get)

    def best_model_predict(self, data_gen):
        self.predictions = self.models[self.best_model_name].predict(data_gen, verbose = 1)

    def print_summary(self):
        print('\nModel Summaries:\n')
        for model_name in self.models.keys():
            print('\n', model_name, '- ROC AUC:', models.roc_auc[model_name])
            print('\n', model_name, '- Validation ROC AUC:', models.val_roc_auc[model_name])

        print('\nBest Model:\n', self.best_model_name)
        print('\nROC AUC of Best Model\n', models.roc_auc[self.best_model_name])
