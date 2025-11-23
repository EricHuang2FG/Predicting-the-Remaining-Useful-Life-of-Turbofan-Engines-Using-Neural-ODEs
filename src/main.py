from src.models.utils.model_utils import evaluate_model, train_model
from src.utils.data_processing import (
    preprocess_training_data,
    preprocess_test_data,
    split_tensors_by_ratio,
)
from src.utils.constants import (
    MODEL_TYPE_NODE,
    MODEL_TYPE_CNN_NODE,
    OPTIMIZED_NODE_SETTINGS,
    OPTIMIZED_CNN_NODE_SETTINGS,
)


def main() -> None:
    x, y, scaler = preprocess_training_data("CMAPSS/train_FD002.txt")

    # code for training a model
    # (x_train, x_validation), (y_train, y_validation) = split_tensors_by_ratio(
    #     x, y, ratio=0.9
    # )

    # train_model(
    #     MODEL_TYPE_CNN_NODE,
    #     "models/cnn_ode.FD002.v1.model",
    #     x_train,
    #     y_train,
    #     settings=OPTIMIZED_CNN_NODE_SETTINGS,
    #     input_validation_data=x_validation,
    #     expected_validation_output=y_validation,
    # )

    # train_model(
    #     MODEL_TYPE_NODE,
    #     "models/ode.FD003.v2.model",
    #     x_train,
    #     y_train,
    #     settings=OPTIMIZED_NODE_SETTINGS,
    #     input_validation_data=x_validation,
    #     expected_validation_output=y_validation,
    # )

    # code for testing a model
    x_test, y_test = preprocess_test_data(
        "CMAPSS/test_FD002.txt", "CMAPSS/RUL_FD002.txt", scaler
    )

    evaluate_model(
        x_test,
        y_test,
        MODEL_TYPE_CNN_NODE,
        None,
        "models/cnn_ode.FD002.v1.model",
        settings=OPTIMIZED_CNN_NODE_SETTINGS,
        figure_dest="figures/plot_cnn_node_FD002.pdf",
    )

    # evaluate_model(
    #     x_test,
    #     y_test,
    #     MODEL_TYPE_NODE,
    #     None,
    #     "models/ode.FD003.v2.model",
    #     settings=OPTIMIZED_NODE_SETTINGS,
    #     figure_dest="figures/plot_node_FD003.pdf",
    # )


if __name__ == "__main__":
    main()
