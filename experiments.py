import pandas as pd
import numpy as np
import time
import matplotlib.pyplot as plt
from tree.base import DecisionTree
from metrics import *

np.random.seed(42)
num_average_time = 5 # Number of times to run each experiment to calculate the average values


# Function to create fake data (take inspiration from usage.py)
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

# Function to calculate average time (and std) taken by fit() and predict() for different N and P for 4 different cases of DTs
def run_experiment(N, M, discrete_input, real_output, num_average_time=10):
    learning_times = []
    prediction_times = []
    X, y = generate_data(N, M, discrete_input, real_output)
    
    for _ in range(num_average_time):
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
    print(learning_time_var)
    prediction_time_avg = np.mean(prediction_times)
    prediction_time_var = np.var(prediction_times)
    print(prediction_time_avg)
    return learning_time_avg, prediction_time_avg



# Run the functions, Learn the DTs and Show the results/plots
N_values = [10, 20, 30, 50]
M_values = [5, 10]

results = []

for N in range(len(N_values)):
    for M in range(len(M_values)):
        for i in [True, False]:
            for o in [True, False]:
                learning_time_avg, prediction_time_avg = run_experiment(N_values[N], M_values[M], i, o, num_average_time)
                results.append((N_values[N], M_values[M], i, o, learning_time_avg, prediction_time_avg))

results_df = pd.DataFrame(results, columns=['N', 'M', 'Discrete Input', 'Real Output', 'Learning Time Avg', 'Prediction Time Avg'])
print(results_df)


# Function to plot the results
fig, axs = plt.subplots(2, 2, figsize=(15, 12))

# Define markers and colors for different M values
markers = {5: 'o', 10: 's'}  # Marker for M=5 and M=10
colors = {5: 'blue', 10: 'green'}  # Colors for M=5 and M=10

# Mapping for subplot titles
titles = {
    (True, True): 'Discrete Input, Real Output',
    (True, False): 'Discrete Input, Discrete Output',
    (False, True): 'Continuous Input, Real Output',
    (False, False): 'Continuous Input, Discrete Output'
}

# Plot Learning Time for each combination of discrete/continuous input and real/discrete output
for i, (discrete_input, real_output) in enumerate([(True, True), (True, False), (False, True), (False, False)]):
    ax = axs[i//2, i%2]
    for M in [5, 10]:
        subset = results_df[(results_df['Discrete Input'] == discrete_input) & 
                            (results_df['Real Output'] == real_output) & 
                            (results_df['M'] == M)]
        ax.plot(subset['N'], subset['Learning Time Avg'], marker=markers[M], color=colors[M], label=f'M={M}')
    
    ax.set_title(titles[(discrete_input, real_output)])
    ax.set_xlabel('N (Number of Samples)')
    ax.set_ylabel('Average Learning Time (s)')
    ax.legend()
    ax.grid(True)

plt.tight_layout()
plt.show()


# Function to plot the results
fig, axs = plt.subplots(2, 2, figsize=(15, 12))

# Define markers and colors for different M values
markers = {5: 'o', 10: 's'}  # Marker for M=5 and M=10
colors = {5: 'blue', 10: 'green'}  # Colors for M=5 and M=10

# Mapping for subplot titles
titles = {
    (True, True): 'Discrete Input, Real Output',
    (True, False): 'Discrete Input, Discrete Output',
    (False, True): 'Continuous Input, Real Output',
    (False, False): 'Continuous Input, Discrete Output'
}

# Plot Prediction Time for each combination of discrete/continuous input and real/discrete output
for i, (discrete_input, real_output) in enumerate([(True, True), (True, False), (False, True), (False, False)]):
    ax = axs[i//2, i%2]
    for M in [5, 10]:
        subset = results_df[(results_df['Discrete Input'] == discrete_input) & 
                            (results_df['Real Output'] == real_output) & 
                            (results_df['M'] == M)]
        ax.plot(subset['N'], subset['Prediction Time Avg'], marker=markers[M], color=colors[M], label=f'M={M}')
    
    ax.set_title(titles[(discrete_input, real_output)])
    ax.set_xlabel('N (Number of Samples)')
    ax.set_ylabel('Average Prediction Time (s)')
    ax.legend()
    ax.grid(True)

plt.tight_layout()
plt.show()
