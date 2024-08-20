import pandas as pd
import numpy as np
import time
import matplotlib.pyplot as plt
from tree.base import DecisionTree
from metrics import *

np.random.seed(42)
num_average_time = 100  # Number of times to run each experiment to calculate the average values


# Function to create fake data (take inspiration from usage.py)
# ...
# Function to calculate average time (and std) taken by fit() and predict() for different N and P for 4 different cases of DTs
# ...
# Function to plot the results
# ...
# Other functions
# ...
# Run the functions, Learn the DTs and Show the results/plots

def generate_data(N, M, discrete=True, real_output=False):
    if discrete:
        X = np.random.randint(0, 2, size=(N, M))
    else:
        X = np.random.rand(N, M)
    
    if real_output:
        y = np.random.rand(N)
    else:
        y = np.random.randint(0, 2, size=N)
    
    return pd.DataFrame(X), pd.Series(y)


def run_experiment(N, M, discrete_input, real_output, iterations=10):
    learning_times = []
    prediction_times = []
    X, y = generate_data(N, M, discrete_input, real_output)
    
    for _ in range(iterations):
        if real_output:
            model = DecisionTree(criterion='information_gain')
        else:
            model = DecisionTree(criterion='information_gain')
        
        start_time = time.time()
        model.fit(X, y)
        learning_times.append(time.time() - start_time)
        
        start_time = time.time()
        model.predict(X)
        prediction_times.append(time.time() - start_time)
    
    learning_time_avg = np.mean(learning_times)
    learning_time_var = np.var(learning_times)
    
    prediction_time_avg = np.mean(prediction_times)
    prediction_time_var = np.var(prediction_times)
    
    return learning_time_avg, prediction_time_avg

N_values = [100, 1000]
M_values = [10, 50]
iterations = 5

results = []

for N in N_values:
    for M in M_values:
        for discrete_input in [True, False]:
            for real_output in [True, False]:
                learning_time_avg, prediction_time_avg = run_experiment(N, M, discrete_input, real_output, iterations)
                results.append((N, M, discrete_input, real_output, learning_time_avg, prediction_time_avg))

# Convert results to a DataFrame for easier analysis
results_df = pd.DataFrame(results, columns=['N', 'M', 'Discrete Input', 'Real Output', 'Learning Time Avg', 'Prediction Time Avg'])

print(results_df)
