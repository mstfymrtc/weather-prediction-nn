import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.metrics import explained_variance_score, \
    mean_absolute_error, \
    median_absolute_error
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt


def show_graph(training_steps, loss_values):
    # print(evaluations[0])
    plt.rcParams['figure.figsize'] = [14, 10]

    plt.scatter(x=training_steps, y=loss_values)
    plt.xlabel('Training steps (Epochs = steps / 2)')
    plt.ylabel('Loss (SSE)')
    plt.show()


# read dataset
df = pd.read_csv('weather-forecast.csv').set_index('date')

# print(df.describe().T)
# print(df.info())
df = df.drop(['mintempm', 'maxtempm'], axis=1)

# X will be a pandas dataframe of all columns except meantempm
X = df[[col for col in df.columns if col != 'meantempm']]

# y will be a pandas series of the meantempm
y = df['meantempm']

# split data into training set and a temporary set using sklearn.model_selection.traing_test_split
X_train, X_tmp, y_train, y_tmp = train_test_split(X, y, test_size=0.2, random_state=23)

# take the remaining 20% of data in X_tmp, y_tmp and split them evenly
X_test, X_val, y_test, y_val = train_test_split(X_tmp, y_tmp, test_size=0.5, random_state=23)

print(f"Training instances   {X_train.shape[0]}, Training features   {X_train.shape[1]}")
print(f"Validation instances {X_val.shape[0]}, Validation features {X_val.shape[1]}")
print(f"Testing instances    {X_test.shape[0]}, Testing features    {X_test.shape[1]}")

# we have give feature column names to our neural network. so we extract these column names
feature_cols = [tf.feature_column.numeric_column(col) for col in X.columns]

# specify that our regressor gonna include 2 layers with 50 nodes width
# relu is used by default as as activation function

# feature_columns: A list-like structure containing a definition of the name and data types for the features being fed into the model
# hidden_units: A list-like structure containing a definition of the number width and depth of the neural network
# optimizer: An instance of tf.Optimizer subclass, which optimizes the model's weights during training; its default is the AdaGrad optimizer.
# activation_fn: An activation function used to introduce non-linearity into the network at each layer; the default is ReLU
# model_dir: A directory to be created that will contain metadata and other checkpoint saves for the model

regressor = tf.estimator.DNNEstimator(head=tf.estimator.RegressionHead(1), feature_columns=feature_cols,
                                      hidden_units=[50, 50], model_dir='tf_wx_model')


# X: The input features to be fed into one of the three DNNRegressor interface methods (train, evaluate, and predict)
# y: The target values of X, which are optional and will not be supplied to the predict call
# num_epochs: An optional parameter. An epoch occurs when the algorithm executes over the entire dataset one time.
# shuffle: An optional parameter, specifies whether to randomly select a batch (subset) of the dataset each time the algorithm executes
# batch_size: The number of samples to include each time the algorithm executes

def wx_input_fn(X, y=None, num_epochs=None, shuffle=True, batch_size=400):
    return tf.compat.v1.estimator.inputs.pandas_input_fn(x=X,
                                                         y=y,
                                                         num_epochs=num_epochs,
                                                         shuffle=shuffle,
                                                         batch_size=batch_size)


evaluations = []
STEPS = 400
for i in range(100):
    regressor.train(input_fn=wx_input_fn(X_train, y=y_train), steps=STEPS)
    evaluations.append(regressor.evaluate(input_fn=wx_input_fn(X_val,
                                                               y_val,
                                                               num_epochs=1,
                                                               shuffle=False)))

loss_values = [ev['loss'] for ev in evaluations]
training_steps = [ev['global_step'] for ev in evaluations]

# show loss-train graph
show_graph(training_steps, loss_values)

# prediction
pred = regressor.predict(input_fn=wx_input_fn(X_test,
                                              num_epochs=1,
                                              shuffle=False))
predictions = np.array([p['predictions'][0] for p in pred])

print("The Explained Variance: %.2f" % explained_variance_score(
    y_test, predictions))
print("The Mean Absolute Error: %.2f degrees Celcius" % mean_absolute_error(
    y_test, predictions))
print("The Median Absolute Error: %.2f degrees Celcius" % median_absolute_error(
    y_test, predictions))
