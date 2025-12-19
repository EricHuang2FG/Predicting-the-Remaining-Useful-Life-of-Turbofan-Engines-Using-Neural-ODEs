# Predicting the Remaining Useful Life of Turbofan Engines Using CNN-Neural ODEs

### Getting the Results
To generate the 8 graphs, run [main.py](src/main.py):
The first 4 graphs show the CNN-NODE graphs in the order of FD001, FD002, FD003, FD004.
The 4 graphs afterwards show the NODE graphs in the order of FD001, FD002, FD003, FD004.


## Introduction
The aim of this project is to determine the efficacy of using neural networks to predict the Remaining Useful Life (RUL) of jet turbofan engines using the [CMAPSS Jet Engine Simulated Dataset](https://data.nasa.gov/dataset/cmapss-jet-engine-simulated-data). This is an important field of research as the consequences of unexpected jet engine failure are so devastating that it is important for engineers to know remaining service life estimates in order to enact proper preventative measures. Although there currently exists deep learning approaches–primarily LSTM based models–to predict RUL, they are founded upon a time-discrete architecture. This is ill-suited to engine sensor readings which are inherently a time continuous dataset. Forcing a fixed time scale is unrealistic as oftentimes, real dynamic systems operate at different time scales due to the fact that for different data intervals, the the data dynamics vary in frequency. This can lead to the loss of key temporal features of the data and introduce instability during back propagation including the generation vanishing or exploding gradients. Thus, the motivation behind the use of a Neural ODE (NODE) stems from the fact that it is a time-continuous neural network that replaces the discrete hidden layers with a vector field governed by an ODE. The state of each "hidden layer" is then computed by integrating the ODE from $t_0$ to $t_1$ using numerical methods. As such, we hypothesize that the underlying changes in sensor data can be modeled by an ODE of the form $$\frac{d \mathbf{h}(t)}{dt} = f(\mathbf{h}(t), t, \theta)$$ and a neural network can model the function $$f(\mathbf{h}(t), t, \theta)$$ which govern the instantaneous derivatives. The decision to implement a CNN-NODE was driven by the motivation that the CNN will be able to pre-extract key patterns in the data that then lead to better RUL predictions once fed through the NODE. The CNN extracts key elements by convolving the weights of the kernel or convolution filter with the input data. By varying the weights of the kernel, the repeated convolution will be able to isolate key patterns within the sensor readings.

The project was inspired from the [Prediction of Remaining Useful  Life of Aero-engines Based on CNN-LSTM-Attention](https://link.springer.com/article/10.1007/s44196-024-00639-w) paper in which the authors achieved the highest RUL prediction accuracy using a CNN-LSTM-Attention model when compared against CNN and LSTM-Attention models. Thus, the results of the Neural ODE is benchmarked against the RUL predictions of the CNN-LSTM_Attention model to see if there is an improvement in RUL prediction accuracy as a result of using a different underlying neural network structure. To evaluate the accuracy, the authors used two regression metrics commonly used in RUL prediction papers as the criteria used for comaprison against the paper. The authors were also interested in how feeding the data through a CNN prior to the ODE-to preemptively isolate patterns in the data-would affect the RUL predictions. The initialization of the CNN-NODE and NODE can be seen in [cnn_node.py](src/models/cnn_node.py) and [node.py](src/models/node.py) respectively. Furthermore, the codebase is organized into files that govern specific functionalities that are further separated by network architecture.

### Outputting the Graphs and Final Results
Only the main.py file needs to be compiled and run to generate the 8 graphs: CNN_NODE and NODE models, each evaluating the 4 engine data sets (FD001, FD002, FD003, FD004). The models weights are stored in `.model` files under the [models](models/) directory. The [main.py](src/main.py) file loads the pre-trained model weigths into the specific type of model (CNN_NODE or NODE), evaluates RMSE and MAPE, and plots a graph comparing the predicted and ground truth RUL values.

### CMAPSS Jet Engine Simulated Data
The CMAPSS dataset is stored under the [CMAPSS](CMAPSS/) directory. Each FD00# file corresponds to the data collected from the # tested jet turbofan engine in simulation. Each engine has many sensors with different readings.

Files of the type "CMAPSS/RUL_FD00#.txt" consist of the ground truth measured RULs to validate the final trained models during testing.

Files of the type "CMAPSS/train_FD00#.txt" and "CMAPSS/test_FD00#.txt" hold sensor measurements for the 21 sensors at each time point and for the training data specifically, also contains the correct RUL associated with each sensor measurement:  
Data is of size [number of time cycles, 26]  
- Column 1: condition number  
- Column 2: time measured in number of cycles  
- Column 3 to 5: operational settings  
- Column 6 to end: sensor measurements from the 21 sensors
- Columns 3-26 were used for training the models.


### Setting Up Virtual Environment and Installing the Required Dependencies
1. In the command terminal, create a virtual environment through the following command: `python3 -m venv .venv`
2. Activate the virtual environment: `.venv\Scripts\activate` (for Windows) `source .venv/bin/activate` (for macOS/Linux)
3. Install the required libraries from requirements.txt: `pip install -r requirements.txt`
4. Select the python interpreter to be the one inside .venv: `Ctrl + Shift + P`, choose `Select Interpreter` and choose the recommended option `.\.venv\Scripts\python.exe`


## Summary of Codebase Functionalities

### Data Processing
The processing of the CMAPPS data follows the method outlined in the [aforementioned paper](https://link.springer.com/article/10.1007/s44196-024-00639-w) and is located in `src/utils/data_processing.py`. It consists of three main components, Z-score standardization, RUL clipping, and sliding time window processing.

1. Z-Score Standardization
Applying a Z-score standardization initially on the data enables meaningful comparisons between data values and reduces the disproportionate skew effects of large data values. Furthermore, by setting the mean to 0 and standard deviation to 1, the data becomes evenly distributed around the the origin with approximately 67% of the data within the range of `[-1, 1]`. It is important to normalize the input training data as neural networks work more efficiently with centered and scaled data [1]. This is because during gradient descent, the gradient is calculated based on the magnitude of the data values, which means that the neural network weights are disproportionately altered and there are instances of vanishing and exploding gradients. This also makes it difficult to optimize hyperparameters such as learning rate.
2. RUL Clipping
All RUL values were clipped to a maximum of 100 cycles as it is a big enough number to adequately convey the idea that the current remaining lifespan of the engine is significantly long. Furthermore, this specific value was shown to work in the paper.
3. Sliding Time Window Processing
Implementing a sliding window for the data set allows a better extraction of local temporal features and gives the model more data to make more robust predictions. A window size of `$k=40$` was chosen for benchmarking purposes as the paper also used this value.


### Neural ODE
The code for creating a Neural ODE can be found in `src/models/node.py`.
Inheiriting from the nn.Module base class, the `NeuralODE` class creates the foundational outline of a neural network by defining the derivative funtion $\frac{d \mathbf{h}(t)}{dt} = f(\mathbf{h}(t), t, \theta)$ that follows a general feed foreward architecture to generate the derivative for each state $h(t)$ over a continuous time interval. The `ODE` class implemenets the numerical integral solver that computes for the predicted values at each state by calling the `NeuralODE` class and implementing its calculated derivates. The Dormand-Prince-15 (DOPRI-15) numerical method is a Runge Kutta method of order 8 that is widely used for integrating ODEs where high precision is needed and adaptive time steps are used.


### CNN-Neural ODE
The code for creating the CNN-Neural ODE can be found by instancing the `CNN_ODE` class in `src/models/cnn_node.py` which also inherits from the nn.Module base class. Due to the high RMSE values obtained from training the FD002 and FD004 datasets using the NODE architecture, the authors decided to pass the data through a CNN before feeding it into the NODE. This is so that the complex data can be first broken down into some key features by the CNN, making it more manageable for the NODE ultimately leading to better RUL predictions.
The final optimized `CNN_ODE` class consists of 36 kernels each 3 ints long that slides along the data one at a time such that no data value is skipped over. The final result from the CNN is then transformed into one long vector that is then used by the Neural ODE as the initial state. Subsequent predictions are then made similar to the `ODE` class in which `Neural_ODE` is called to get the state derivatives which are then numberically integrated using the DOPRI-15 method.


### Model Utilities
The code for the different functionalities of each model can be found in `src/models/utils/model_utils.py` which includes functions to initialize, train, and evaluate the models.

1. `def initialize_model` initializes the general architecture of the specific neural network (NODE or CNN-NODE) without the specification of any weights.

2. `def train_model` trains the model by implementing the Adam Algorithm to optimize and specify the weights of the model in order to minimize the loss function. Iterating through the epochs one at a time, and traversing through each epoch in batches, the function calls the current model (with its current likely unoptimized weights) to get its predicted values. The loss is then computed by comparing the predicted with the ground truth RUL values. The line `loss.backward()` is the back propagation step that determines how much each weight contributes to the total loss and calculates the gradient of the loss function with respect to each weight to determine the magnitude of weight adjustment.

3. `def evaluate_model` evaluates the model by comparing the model's predicted RUL with the ground truth RUL and graphs both in addition to calculating the RMSE and MAPE values. Note, the data is sorted for ease of visual validation of the accuracy of the predicted RUL values and there is no relationship to be inferred from the resulting curves.


### Tuning Hyperparameters
In `src/models/utils/tune_models.py`, models are trained based on the first 70% of the training data for each engine and is then evaluated on the rest of the 30% of the training data. It contains functions `learning_rate_sweep`, `hidden_dimensions_sweep`, and `dropout_rate_sweep` that sweep through different values of those respective hyperparameters to find the most optimized values that result in the lowest RMSE values. For example, in choosing the best learning rate, the `lr` entry of the settings/hyperparameter dictionary for the model is swapped with all five candidate learning rate values in the array in a for loop: `candidate_lrs = [0.1, 0.01, 0.005, 0.001, 0.0003]`

For the hidden dimensions, if the specific model being trained is CNN_NODE, the array is `[2, 3, 4]`; otherwise, if it is NODE, the array becomes `[32, 64, 128]`.

Furthermore, it is important to note that the seed is held constant throughout the process of training the models with different values of hyperparameters to make the results reproducible. Only then can lower calculated RMSE and MAPE values be attributed to changes in hyperparameter values and not due to randomly initialized values that happen to lead to favourable results.


### Constants
The `src/utils/constants.py` file contains all the constants used thorughout all the files. Examples include two dictionaries `OPTIMIZED_CNN_NODE_SETTINGS` and `OPTIMIZED_NODE_SETTINGS` which correspond to CNN_NODE models and NODE models respectively. They each describe the values of the hyperparameters to be used for their respective model. `OPTIMIZED_CNN_NODE_SETTINGS` also specifies the kernel size.


[1] https://medium.com/@weidagang/demystifying-machine-learning-normalization-0cdb8b281234
