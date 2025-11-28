import src.node as node
import matplotlib.pyplot as plt
import torch
import numpy as np
import random

# general order of tuning: learning rate, hidden dimensions, dropout rate
# important to hold seed constant as then can compare between different neural networks with variations in their parameters
seed = 42
torch.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)

from src.data_processing import preprocess_training_data, preprocess_test_data


def main() -> None:
    x, y, scaler = preprocess_training_data("CMAPSS/train_FD002.txt")
    # right now we just use an epoch of 25
    # before training of model
    # model = node.train_model("models/ode.FD001.v3.model", x, y)
    
    x_test, y_test = preprocess_test_data(
        "CMAPSS/test_FD002.txt", "CMAPSS/RUL_FD002.txt", scaler
    )
    node.evaluate_model(x_test, y_test, "models/ode.FD002.v1.model")

def learning_rate_sweep():
    x, y, scaler = preprocess_training_data("CMAPSS/train_FD001.txt")
    x_test, y_test = preprocess_test_data(
        "CMAPSS/test_FD001.txt", "CMAPSS/RUL_FD001.txt", scaler
    )

    candidate_lrs = [0.1, 0.01, 0.005, 0.001, 0.0003]
    losses = []
    for lr in candidate_lrs:
        model = node.train_model_tunelr(lr, x, y)
        losses.append(node.evaluate_model_tunelr(x_test, y_test, model))

        
    plt.figure(figsize=(12,9))
    plt.plot(candidate_lrs, losses, marker='x')
    plt.xscale('log')
    plt.xlabel("learning rates")
    plt.ylabel("validation loss RMSE")
    plt.grid(True)
    plt.show()

def hidden_dimensions_sweep():
    x, y, scaler = preprocess_training_data("CMAPSS/train_FD001.txt")
    x_test, y_test = preprocess_test_data(
        "CMAPSS/test_FD001.txt  ", "CMAPSS/RUL_FD001.txt", scaler
    )
    candidate_hds = [32, 48, 64, 96, 128]
    losses = []
    for hd in candidate_hds:
        model = node.train_model_tunehd(hd, x, y)
        losses.append(node.evaluate_model_tunelr(x_test, y_test, model))

        
    plt.figure(figsize=(12,9))
    plt.plot(candidate_hds, losses, marker='x')
    plt.xlabel("number of hidden dimensions for ODE")
    plt.ylabel("validation loss RMSE")
    plt.grid(True)
    plt.show()

def dropout_rate_sweep():
    x, y, scaler = preprocess_training_data("CMAPSS/train_FD002.txt")
    x_test, y_test = preprocess_test_data(
        "CMAPSS/test_FD002.txt", "CMAPSS/RUL_FD002.txt", scaler
    )
    # tuning dropout rate to see if overfit to training data
    # and need to remove some neurons so others dont become too generalized and overall more generalizable
    candidate_dor = [0, 0.05, 0.1, 0.2, 0.3]
    losses = [] # error/difference between the predicted value and the correct RUL values
    total_loss = [] # error/difference between the predicted value and the corresponding training data value
    for dor in candidate_dor:
        trained_model, trained_loss = node.train_model_tunedor(dor, 32, x, y)
        rmse = node.evaluate_model_tunelr(x_test, y_test, trained_model)
        losses.append(rmse)
        total_loss.append(trained_loss)
        print(rmse) #priniting rmse just to see what the values are, can get rid of

    plt.figure(figsize=(12,9))
    plt.plot(candidate_dor, losses, color="red", marker='x')
    plt.plot(candidate_dor, total_loss, color="blue", marker='o')
    plt.xlabel("dropout rate")
    plt.ylabel("validation loss RMSE")
    plt.grid(True)
    plt.show()
    
    # want to see how the validation loss (test data) compares with total loss (train data)
    # general rule of thumb: if training loss << validation loss, increase dropout because overfitting, vice versa
    print(f"total losses from training set after 25 epochs {[round(l, 6) for l in total_loss]}")    

def visualize():
    #just to test out what the final thing will look like
    x, y, scaler = preprocess_training_data("CMAPSS/train_FD001.txt")
    x_test, y_test = preprocess_test_data(
        "CMAPSS/test_FD001.txt", "CMAPSS/RUL_FD001.txt", scaler
    )
    # can replace with any train_model_XXX function, need to remove trained_loss for other train_model_XXX functions
    trained_model, trained_loss = node.train_model_tunedor(0.1, 96, x, y)
    node.evaluate_model_tune_visualize(x_test, y_test, trained_model)


if __name__ == "__main__":
    main()
    #learning_rate_sweep()
