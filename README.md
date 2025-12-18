# Predicting the Remaining Useful Life of Turbofan Engines Using CNN-Neural ODEs

### Results:
To generate the 8 graphs, run [main.py](src/main.py):
The first 4 graphs show the CNN-NODE graphs in the order of FD001, FD002, FD003, FD004.
The second 4 graphs show the NODE graphs in the order of FD001, FD002, FD003, FD004.


## Introduction
The aim of this project is to determine the efficacy of using neural networks to predict the Remaining Useful Life (RUL) of jet turbofan engines using the [CMAPSS Jet Engine Simulated Dataset](https://data.nasa.gov/dataset/cmapss-jet-engine-simulated-data). This is an important field of research as the consequences of unexpected jet engine failure are so devastating that it is important for engineers to know remaining service life estimates in order to enact proper preventative measures. Although there currently exists deep learning approaches–primarily LSTM based models–to predict RUL, they are founded upon a time-discrete architecture. This is ill-suited to engine sensor readings which are inherently a time continuous dataset. Forcing a fixed time scale is unrealistic as oftentimes, real dynamic systems operate at different time scales due to the fact that for different data intervals, the the data dynamics vary in frequency. This can lead to the loss of key temporal features of the data and introduce instability during back propagation including the generation vanishing or exploding gradients. Thus, the motivation behind the use of a Neural ODE (NODE) stems from the fact that it is a time-continuous neural network that replaces the discrete hidden layers with a vector field governed by an ODE. The state of each "hidden layer" is then computed by integrating the ODE from $t_0$ to $t_1$ using numerical methods. As such, we hypothesize that the underlying changes in sensor data can be modeled by an ODE of the form $\frac{d \mathbf{h}(t)}{dt} = f(\mathbf{h}(t), t, \theta)$ and a neural network can model the function $f(\mathbf{h}(t), t, \theta)$ which govern the instantaneous derivatives.

The project was inspired from the [Prediction of Remaining Useful  Life of Aero-engines Based on CNN-LSTM-Attention](https://link.springer.com/article/10.1007/s44196-024-00639-w) paper in which the authors achieved the highest RUL prediction accuracy using a CNN-LSTM-Attention model when compared against CNN and LSTM-Attention models. Thus, the results of the Neural ODE is benchmarked against the RUL predictions of the CNN-LSTM_Attention model to see if there is an improvement in RUL prediction accuracy as a result of using a different underlying neural network structure. To evaluate the accuracy, the authors used two regression metrics commonly used in RUL prediction papers as the criteria used for comaprison against the paper. The authors were also interested in how feeding the data through a CNN prior to the ODE-to preemptively isolate patterns in the data-would affect the RUL predictions. The initialization of the CNN-NODE and NODE can be seen in [cnn_node.py](src/models/cnn_node.py) and [node.py](src/models/node.py) respectively. Furthermore, the codebase is organized into files that govern specific functionalities that are further separated by network architecture.

### Outputting the Graphs and Final Results:
Only the main.py file needs to be compiled and run to generate the 8 graphs: CNN_NODE and NODE models, each evaluating the 4 engine data sets (FD001, FD002, FD003, FD004). The models weights are stored in .model files under the [models](models/) directory. The [main.py](src/main.py) file loads the pre-trained model weigths into the specific type of model (CNN_NODE or NODE), evaluates RMSE and MAPE, and plots a graph comparing the predicted and ground truth RUL values.

### CMAPSS Jet Engine Simulated Data
The CMAPSS dataset is stored udner the [CMAPSS](CMAPSS/) directory. Each FD00# file corresponds to the data collected from the # tested jet turbofan engine in simulation. Each engine has many sensors with different readings.

Files of the type "CMAPSS/RUL_FD00#.txt" consist of the ground truth measured RULs to validate the final trained models during testing.

Files of the type "CMAPSS/train_FD00#.txt" and "CMAPSS/test_FD00#.txt" hold sensor measurements for the 21 sensors at each time point and for the training data specifically, also contains the correct RUL associated with each sensor measurement:  
Data is of size [number of time cycles, 26]  
Column 1: condition number  
Column 2: time measured in number of cycles  
Column 3 to 5: operational settings  
Column 6 to end: sensor measurements from the 21 sensors
Columns 3-26 were used for training the models.


### Setting Up Virtual Environment and Installing the Required Dependencies
1. In the command terminal, create a virtual environment through the following command: `python3 -m venv .venv`
2. Activate the virtual environment: `.venv\Scripts\activate` (for Windows) `source .venv/bin/activate` (for macOS/Linux)
3. Install the required libraries from requirements.txt: `pip install -r requirements.txt`
4. Select the python interpreter to be the one inside .venv: `Ctrl + Shift + P`, choose `Select Interpreter` and choose the recommended option `.\.venv\Scripts\python.exe`


## Summary of Codebase Functionalities:

### Data Processing
src/utils/data_processing.py
The data processing for the CMAPSS dataset closely follows the method followed in the aforementioned paper. EXPLAIN MORE

### src/utils/constants.py
This file contains all the constants used thorughout all the files. Examples include two dictionaries `OPTIMIZED_CNN_NODE_SETTINGS` and `OPTIMIZED_NODE_SETTINGS`, one corresponding to CNN_NODE models and the other to NODE models respectively, that describe the values of the hyperparameters to be used for the model. It also specifies the kernel size.

### src/models/node.py


### src/models/cnn_node.py




### src/models/utils/model_utils.py


### src/models/utils/tune_models.py
In this file, models are trained based on the first 70% of the training data for each engine and is then evaluated on the rest of the 30% of the training data. It contains functions `learning_rate_sweep`, `hidden_dimensions_sweep`, and `dropout_rate_sweep` that sweep through different values of those respective hyperparameters to find the best value for each hyperparameter that results in the lowest RMSE values. For example, in choosing the best learning rate, the `lr` entry of the settings/hyperparameter dictionary for the model is swapped with all five candidate learning rate values in the array in a for loop: `candidate_lrs = [0.1, 0.01, 0.005, 0.001, 0.0003]`

For the hidden dimensions, if the specific model being trained is CNN_NODE, the array is `[2, 3, 4]` oterwise if it is NODE, the array is `[32, 64, 128]`.

Furthermore, it is important to note that the seed is held constant throughout the process of training the models with different values of hyperparameters to make the results reproducibl. Only then can differences noted in RMSE values be attributed to changes in hyperparameter values and not due to favourable randomly initialized values.




***Predicting the Remaining Useful Life of Turbofan Engines Using Neural ODEs***

Please do not push directly to the main branch!!!!