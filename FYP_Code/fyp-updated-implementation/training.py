from argparse import ArgumentParser
from collections import namedtuple
from datetime import datetime
from math import ceil
from pathlib import Path
from sklearn.metrics import confusion_matrix
import numpy as np
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split, StratifiedKFold
from tensorflow import keras
from dataset_tools import preprocess_raw_eeg, ACTIONS, load_all_raw_data
from neural_nets import EEGNet
import os
import itertools

classes = ["chew","clench","eye_blink","feet","head","jerk"]

def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')


def fit_model(train_X: np.ndarray, train_y: np.ndarray, validation_X: np.ndarray, validation_y: np.ndarray,
              network_hyperparameters_dict: dict, training_name=""):
    F1 = network_hyperparameters_dict['F1']
    D = network_hyperparameters_dict['D']
    F2 = network_hyperparameters_dict['F2']
    learning_rate = network_hyperparameters_dict['learning_rate']
    batch_size = network_hyperparameters_dict['batch_size']
    model_function = network_hyperparameters_dict['network_to_train']
    epochs = network_hyperparameters_dict['epochs']
    metric_to_monitor = network_hyperparameters_dict['metric_to_monitor']

    training_name = f"F1:{F1}_D:{D}_F2:{F2}_lr:{learning_rate}{training_name}"
    model = model_function(nb_classes=len(ACTIONS), F1=F1, D=D, F2=F2)
    model.summary()
    model.compile(loss='sparse_categorical_crossentropy',
                  optimizer=keras.optimizers.Nadam(learning_rate=learning_rate),
                  metrics=['accuracy'])

    model_name = model.name
    training_start = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    model_path: Path = Path(f"../{model_name}_{training_start}_{training_name}")
    Path.mkdir(model_path, exist_ok=True)

    history = model.fit(
        x=train_X,
        y=train_y,
        steps_per_epoch=ceil(train_X.shape[0] / batch_size),
        validation_data=(validation_X, validation_y),
        epochs=epochs
    )

    plot_model_accuracy_and_loss(history, model_path)



def plot_model_accuracy_and_loss(history, model_path):
    plt.clf()
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    epochs = range(1, len(loss) + 1)
    plt.plot(epochs, loss, 'g', label='Training loss')
    plt.plot(epochs, val_loss, 'b', label='Validation loss')
    plt.title('Training and validation loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig(f'{model_path}/training-validation-loss')

    plt.clf()
    accuracy = history.history['accuracy']
    val_accuracy = history.history['val_accuracy']
    plt.plot(epochs, accuracy, 'g', label='Training accuracy')
    plt.plot(epochs, val_accuracy, 'b', label='Validation accuracy')
    plt.title('Training and validation accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('acc')
    plt.legend()
    plt.savefig(f'{model_path}/training-validation-accuracy')



def kfold_cross_val(data_X: np.ndarray, data_y: np.ndarray, network_hyperparameters_dict: dict):
    F1 = network_hyperparameters_dict['F1']
    D = network_hyperparameters_dict['D']
    F2 = network_hyperparameters_dict['F2']
    learning_rate = network_hyperparameters_dict['learning_rate']
    batch_size = network_hyperparameters_dict['batch_size']
    model_function = network_hyperparameters_dict['network_to_train']
    epochs = network_hyperparameters_dict['epochs']
    training_name = f"F1:{F1}_D:{D}_F2:{F2}_lr:{learning_rate}"
    random_state = network_hyperparameters_dict['RANDOM_STATE']
    metric_to_monitor = network_hyperparameters_dict['metric_to_monitor']
    model_name = model_function.__name__
    training_start = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    num_folds = network_hyperparameters_dict['num_folds'] = 5

    models_path: Path = Path(
        f"./{model_name}")
    Path.mkdir(models_path, exist_ok=True, parents=True)

    acc_per_fold = []
    loss_per_fold = []

    strat_kfold = StratifiedKFold(n_splits=num_folds, shuffle=True,
                                  random_state=random_state)  # For mantaining class balance
    fold_no = 1

    for train_indexes, test_indexes in strat_kfold.split(data_X, data_y):
        model = model_function(nb_classes=len(ACTIONS), F1=F1, D=D, F2=F2)
        model.compile(loss='sparse_categorical_crossentropy',
                      optimizer=keras.optimizers.Nadam(learning_rate=learning_rate),
                      metrics=['accuracy'])
        curerent_fold_model_path = Path(models_path / f"fold_n{fold_no}")
        Path.mkdir(curerent_fold_model_path, exist_ok=True)

        train_X, val_X, train_y, val_y = train_test_split(data_X[train_indexes], data_y[train_indexes], test_size=2 / 9,
                                                          random_state=random_state,
                                                          stratify=data_y[train_indexes])
        test_X = data_X[test_indexes]
        test_y = data_y[test_indexes]

        print('------------------------------------------------------------------------')
        print(f'Training for fold {fold_no} ...')

        history = model.fit(train_X,
                            train_y,
                            batch_size=batch_size,
                            steps_per_epoch=ceil(train_X.shape[0] / batch_size),
                            epochs=epochs,
                            verbose=1,
                            validation_data=(val_X, val_y))
        scores = model.predict(test_X)
        predictions = np.argmax(scores,axis = 1)
        print(test_y,predictions)
        np.random.seed(0)
        cm = confusion_matrix(test_y, predictions)
        plt.figure()
        plot_confusion_matrix(cm, classes=classes, title='Confusion matrix')

        # Save the plot as an image
        plt.savefig(f'{curerent_fold_model_path}/{fold_no}_confusion_matrix.jpg', dpi=300)
        

        print(f'Score for fold {fold_no}: {model.metrics_names[0]} of {scores[0]};'
              f' {model.metrics_names[1]} of {scores[1] * 100}%')
        acc_per_fold.append(scores[1] * 100)
        loss_per_fold.append(scores[0])
        model.save(f'{curerent_fold_model_path}/{fold_no}_model.h5')
        fold_no += 1
        print('------------------------------------------------------------------------')
        print(acc_per_fold)
        plot_model_accuracy_and_loss(history, curerent_fold_model_path)

class Hyperparameters:
    def __init__(self, random_state, label_mapping):
        self.random_state = random_state
        self.label_mapping = label_mapping

    def set_default_hyperparameters(self):
        network_hyperparameters_dict = {}

        # NETWORKS PARAMETERS
        network_hyperparameters_dict['network_to_train'] = EEGNet
        network_hyperparameters_dict['epochs'] = 10000
        network_hyperparameters_dict['learning_rate'] = 5e-5
        network_hyperparameters_dict['F1'] = 12
        network_hyperparameters_dict['D'] = 2
        network_hyperparameters_dict['F2'] = 24
        network_hyperparameters_dict['RANDOM_STATE'] = self.random_state
        network_hyperparameters_dict['batch_size'] = 32
        network_hyperparameters_dict['metric_to_monitor'] = 'val_loss'
        network_hyperparameters_dict['label_mapping'] = self.label_mapping
        network_hyperparameters_dict['num_folds'] = 5
        return network_hyperparameters_dict

def main():
    parser = ArgumentParser()
    parser.add_argument("--random-state", type=int, default=42)
    args = parser.parse_args()
    RANDOM_STATE = args.random_state

    STARTING_DIR = Path("./dataset_6D_actions")
    #GAN_PATH = "../WGAN_2021-04-16 00:58:22/models"  # Insert GAN model path here or comment such line
    SPLITTING_PERCENTAGE = namedtuple('SPLITTING_PERCENTAGE', ['train', 'val', 'test'])
    SPLITTING_PERCENTAGE.train, SPLITTING_PERCENTAGE.val, SPLITTING_PERCENTAGE.test = (70, 20, 10)

    raw_data_X, data_y, label_mapping = load_all_raw_data(starting_dir=STARTING_DIR)

    hyperparameters = Hyperparameters(RANDOM_STATE, label_mapping)

    network_hyperparameters_dict = hyperparameters.set_default_hyperparameters()
    print(len(raw_data_X), len(data_y))
    data_X, fft_data_X = preprocess_raw_eeg(raw_data_X, lowcut=8, highcut=45, coi3order=0)

    tmp_train_X, test_X, tmp_train_y, test_y = train_test_split(data_X, data_y,
                                                                test_size=SPLITTING_PERCENTAGE.test / 100,
                                                                random_state=network_hyperparameters_dict[
                                                                    'RANDOM_STATE'], stratify=data_y)
    actual_valid_split_fraction = SPLITTING_PERCENTAGE.val / (100 - SPLITTING_PERCENTAGE.test)
    train_X, val_X, train_y, val_y = train_test_split(tmp_train_X, tmp_train_y, test_size=actual_valid_split_fraction,
                                                      random_state=network_hyperparameters_dict['RANDOM_STATE'],
                                                      stratify=tmp_train_y)

    kfold_cross_val(data_X, data_y, network_hyperparameters_dict)


if __name__ == '__main__':
    main()
