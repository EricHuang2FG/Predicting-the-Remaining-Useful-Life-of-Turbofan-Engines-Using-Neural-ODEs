import torch
import numpy as np
from sklearn.preprocessing import StandardScaler

from src.constants import NUM_SETTINGS_AND_SENSOR_READINGS


# reference paper for this processing:
# https://link.springer.com/article/10.1007/s44196-024-00639-w#data-availability
def preprocess_training_data(
    file_path: str, window_size: int = 40, max_rul: int = 100
) -> tuple[torch.FloatTensor, torch.FloatTensor, StandardScaler]:
    # data has the size (num_lines, 26)
    # column 1: engine number
    # column 2: cycle count
    # column 3 to 5: operational settings
    # column 6 to end: sensor measurements
    data: np.ndarray = np.loadtxt(file_path)

    sliding_window_sequences: list = []
    rul_labels: list = []

    # engine_num = [1, 2, 3, 4, ..., ]
    for engine_num in np.unique(data[:, 0]):
        # get data s.t. its first column is the engine_num
        engine_data: np.ndarray = data[data[:, 0] == engine_num]
        max_life: int = len(engine_data)

        # perform RUL segmentation
        # element-wise subtraction of max_num_cycles - engine_data array
        # then cap every element at max_rul (ref the paper)
        rul: np.ndarray = max_life - engine_data[:, 1]
        rul_segmented: np.ndarray = np.minimum(max_rul, rul)

        # do sliding window processing
        settings_and_sensor_readings: np.ndarray = engine_data[:, 2:26]
        for i in range(len(settings_and_sensor_readings) - window_size + 1):
            sliding_window_sequences.append(
                settings_and_sensor_readings[i : i + window_size]
            )
            rul_labels.append(
                rul_segmented[i + window_size - 1]
            )  # rul at the end of the window

    # shape: (num_sequences, window_size, 24)
    input_data: np.ndarray = np.array(sliding_window_sequences)
    expected_output: np.ndarray = np.array(rul_labels)

    # standardize using Z = (x_i - u) / sigma
    # documentation for this one: https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.StandardScaler.html
    # it does Z = (x_i - u) / sigma for you
    scaler: StandardScaler = StandardScaler()
    orig_shape: tuple = input_data.shape
    input_data_flattened: np.ndarray = input_data.reshape(
        -1, NUM_SETTINGS_AND_SENSOR_READINGS
    )
    input_data_flattened = scaler.fit_transform(input_data_flattened)
    input_data = input_data_flattened.reshape(orig_shape)

    return torch.FloatTensor(input_data), torch.FloatTensor(expected_output), scaler


def preprocess_test_data(
    test_data_path: str, rul_path: str, scaler: StandardScaler, window_size: int = 40
) -> tuple[torch.FloatTensor, torch.FloatTensor]:
    # similar to preprocess_trainng_data() but runs on test data and real RUL
    # also only takes the last window of each engine test
    test_data: np.ndarray = np.loadtxt(test_data_path)
    rul_data: np.ndarray = np.loadtxt(rul_path)

    test_sequence: list = []

    for engine_num in np.unique(test_data[:, 0]):
        engine_data: np.ndarray = test_data[test_data[:, 0] == engine_num]

        engine_sequence: np.ndarray = engine_data[-window_size:, 2:26]
        engine_sequence_size: int = engine_sequence.shape[0]
        if engine_sequence_size < window_size:
            filler: np.ndarray = np.zeros(
                (window_size - engine_sequence_size, NUM_SETTINGS_AND_SENSOR_READINGS)
            )
            engine_sequence = np.concatenate((filler, engine_sequence), axis=0)

        test_sequence.append(engine_sequence)

    input_data: np.ndarray = np.array(test_sequence)

    # standardize using scaler from TRAIN DATA
    orig_shape: tuple = input_data.shape
    input_data_flattened: np.ndarray = input_data.reshape(
        -1, NUM_SETTINGS_AND_SENSOR_READINGS
    )
    input_data_flattened = scaler.transform(input_data_flattened)
    input_data = input_data_flattened.reshape(orig_shape)

    return torch.FloatTensor(input_data), torch.FloatTensor(rul_data)


if __name__ == "__main__":
    pass
