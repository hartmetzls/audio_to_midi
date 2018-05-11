from keras.models import load_model
from models import pickle_if_not_pickled, root_mse, reshape_for_conv2d, r2_coeff_determination
from sklearn.model_selection import KFold, train_test_split
# no GPU support for sklearn's cross_val_score
from create_dataset import done_beep
import matplotlib.pyplot as plt
from models import create_model

def k_fold_cv():
    filepath = "model_and_visualizations.1363/weights-improvement-38-0.1363.hdf5"
    cqt_segments, midi_segments = pickle_if_not_pickled()
    cqt_segments_reshaped, midi_segments_reshaped = reshape_for_conv2d(cqt_segments, midi_segments)
    # shuffles before splitting by default
    cqt_train_and_valid, cqt_test, midi_train_and_valid, midi_test = train_test_split(
        cqt_segments_reshaped, midi_segments_reshaped, test_size=0.2, random_state=21)
    # check model robustness with kfold cross val
    n_folds = 5
    data, labels = cqt_train_and_valid, midi_train_and_valid
    k_fold = KFold(n_splits=n_folds)  # Provides train/test indices to split data in train/test sets.
    for train_indices, valid_indices in k_fold.split(cqt_train_and_valid):
        print('Train: %s | valid: %s' % (train_indices, valid_indices))
    example_cqt_segment = cqt_train_and_valid[0]
    input_height, input_width, input_depth = example_cqt_segment.shape
    example_midi_segment = midi_train_and_valid[0]
    one_d_array_len = len(example_midi_segment)

    for i, (train, test) in enumerate(k_fold.split(cqt_train_and_valid)):
        print("Running Fold", i + 1, "/", n_folds)
        # model = None #Clearing the NN # https://github.com/keras-team/keras/issues/1711 KeironO
        # model = load_model(filepath,
        #                    custom_objects={'root_mse': root_mse, 'r2_coeff_determination': r2_coeff_determination})

        model = create_model(input_height, input_width, one_d_array_len)
        # saving time (best models have reached best val_score before epoch 40)
        epochs = 50

        # train the models using these subsets
        # (Normally I would not use the same data for validation_data and evaluate,
        # but I believe in this case it's appropriate in order to visualize signs of over/under -fitting)
        history_for_plotting = model.fit(data[train], labels[train],
                                         epochs=epochs, verbose=2, validation_data=(data[test], labels[test]))

        valid_score = model.evaluate(data[test], labels[test], verbose=0)
        print("valid score:")
        print("[loss (rmse), root_mse, mae, r2_coeff_determination]")
        print(valid_score)

        done_beep()

        # summarize history for loss
        # https://machinelearningmastery.com/display-deep-learning-model-training-history-in-keras/
        plt.plot(history_for_plotting.history['loss'])
        plt.plot(history_for_plotting.history['val_loss'])
        plt.title('model loss')
        plt.ylabel('loss')
        plt.xlabel('epoch')
        plt.legend(['train rmse', 'validation rmse'], loc='upper right')
        plt.show()

def main():
    k_fold_cv()

if __name__ == '__main__':
    main()