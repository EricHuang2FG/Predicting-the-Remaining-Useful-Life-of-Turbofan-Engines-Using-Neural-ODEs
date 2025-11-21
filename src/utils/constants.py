NUM_SETTINGS_AND_SENSOR_READINGS: int = 24

MODEL_TYPE_NODE = "node"
MODEL_TYPE_CNN_NODE = "cnn_node"

DEFAULT_WINDOW_SIZE = 40
DEFAULT_NETWORK_SETTINGS: dict = {
    "batch_size": 128,
    "epochs": 25,
    "lr": 0.001,
    "hidden_dimension": 64,
    "encoder_dimension": 128,
    "regressor_dimension": 32,
    "dropout": 0.2
}