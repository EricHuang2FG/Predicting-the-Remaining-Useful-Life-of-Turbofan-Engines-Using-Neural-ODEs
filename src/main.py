import torch
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
    # reproducing figures in the paper

    # process test data
    datasets: list[str] = ["001", "002", "003", "004"]
    test_data: list[tuple[torch.FloatTensor, torch.FloatTensor]] = []
    for dataset in datasets:
        _, _, scaler = preprocess_training_data(f"CMAPSS/train_FD{dataset}.txt")
        x_test, y_test = preprocess_test_data(
            f"CMAPSS/test_FD{dataset}.txt", f"CMAPSS/RUL_FD{dataset}.txt", scaler
        )
        test_data.append((x_test, y_test))

    # produce graphs for CNN-NODE models
    cnn_node_models: list[str] = [
        "cnn_ode.FD001.v2.model",
        "cnn_ode.FD002.v2.model",
        "cnn_ode.FD003.v2.model",
        "cnn_ode.FD004.v2.model",
    ]
    for index, cnn_node_model in enumerate(cnn_node_models):
        x_test, y_test = test_data[index]
        evaluate_model(
            x_test,
            y_test,
            MODEL_TYPE_CNN_NODE,
            None,
            f"models/{cnn_node_model}",
            settings=OPTIMIZED_CNN_NODE_SETTINGS,
            figure_dest=f"figures/plot_cnn_node_FD{datasets[index]}.pdf",
        )

    # produce graphs for NODE models
    node_models: list[str] = [
        "ode.FD001.v5.model",
        "ode.FD002.v3.model",
        "ode.FD003.v2.model",
        "ode.FD004.v2.model",
    ]
    for index, node_model in enumerate(node_models):
        x_test, y_test = test_data[index]
        evaluate_model(
            x_test,
            y_test,
            MODEL_TYPE_NODE,
            None,
            f"models/{node_model}",
            settings=OPTIMIZED_NODE_SETTINGS,
            figure_dest=f"figures/plot_node_FD{datasets[index]}.pdf",
        )

    # example code for training a model
    # (x_train, x_validation), (y_train, y_validation) = split_tensors_by_ratio(
    #     x, y, ratio=0.9
    # )

    # train_model(
    #     MODEL_TYPE_CNN_NODE,  # model type, either MODEL_TYPE_CNN_NODE or MODEL_TYPE_NODE
    #     "models/cnn_ode.FD003.v2.model",  # file path of where you want to save the model
    #     x_train,
    #     y_train,
    #     settings=OPTIMIZED_CNN_NODE_SETTINGS,  # OPTIMIZED_CNN_NODE_SETTINGS for CNN-NODE, OPTIMIZED_NODE_SETTINGS for NODE
    #     input_validation_data=x_validation,
    #     expected_validation_output=y_validation,
    # )


if __name__ == "__main__":
    main()
