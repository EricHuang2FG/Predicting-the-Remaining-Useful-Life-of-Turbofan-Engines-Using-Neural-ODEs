DATASET_ID_FD001 = "FD001"
DATASET_ID_FD002 = "FD002"
DATASET_ID_FD003 = "FD003"
DATASET_ID_FD004 = "FD004"

DEFAULT_FIGURE_SIZE: tuple = (12, 9)
LINE_WIDTH: int = 3

NUM_SETTINGS_AND_SENSOR_READINGS: int = 24

MODEL_TYPE_NODE: str = "node"
MODEL_TYPE_CNN_NODE: str = "cnn_node"

DIMENSION_TYPE_HIDDEN: str = "hidden_dimension"
DIMENSION_TYPE_ENCODER: str = "encoder_dimension"
DIMENSION_TYPE_REGRESSOR: str = "regressor_dimension"

DEFAULT_WINDOW_SIZE: int = 40
DEFAULT_NETWORK_SETTINGS: dict = {
    "batch_size": 128,
    "epochs": 25,
    "lr": 0.001,
    "hidden_dimension": 64,
    "encoder_dimension": 128,
    "regressor_dimension": 32,
    "dropout": 0.2,
}

OPTIMIZED_NODE_SETTINGS: dict = {
    "batch_size": 128,
    "epochs": 25,
    "lr": 0.001,
    "hidden_dimension": 128,
    "encoder_dimension": 128,
    "regressor_dimension": 128,
    "dropout": 0.0,
}
OPTIMIZED_CNN_NODE_SETTINGS: dict = {
    "batch_size": 128,
    "epochs": 25,
    "lr": 0.001,
    "cnn_num_kernals": 12,
    "cnn_kernal_size": 3,
    "cnn_stride": 1,
    "cnn_padding": 1,
    "hidden_dimension": 128,
    "encoder_dimension": 128,
    "regressor_dimension": 128,
    "dropout": 0.0,
}
