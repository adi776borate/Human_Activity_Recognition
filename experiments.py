import pandas as pd
import numpy as np
import time
import matplotlib.pyplot as plt
from tree.base import DecisionTree
from metrics import *

np.random.seed(42)
num_average_time = 100  # Number of times to run each experiment to calculate the average values


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
    print(learning_time_var)
    prediction_time_avg = np.mean(prediction_times)
    prediction_time_var = np.var(prediction_times)
    print(prediction_time_avg)
    return learning_time_avg, prediction_time_avg



# Run the functions, Learn the DTs and Show the results/plots
N_values = [20, 50, 100]
M_values = [5, 10, 30]
iterations = 3

results = []

for N in range(len(N_values)):
    for i in [True, False]:
        for o in [True, False]:
            learning_time_avg, prediction_time_avg = run_experiment(N_values[N], M_values[N], i, o, iterations)
            results.append((N_values[N], M_values[N], i, o, learning_time_avg, prediction_time_avg))

results_df = pd.DataFrame(results, columns=['N', 'M', 'Discrete Input', 'Real Output', 'Learning Time Avg', 'Prediction Time Avg'])
print(results_df)


# Function to plot the results
fig, axs = plt.subplots(1, 2, figsize=(15, 6))

# Define markers and colors for different input/output types
colors = { 
    (True, True): 'blue',   # Discrete Input, Real Output
    (True, False): 'green', # Discrete Input, Discrete Output
    (False, True): 'red',   # Continuous Input, Real Output
    (False, False): 'orange' # Continuous Input, Discrete Output
}

labels = {
    (True, True): 'Discrete Input, Real Output',
    (True, False): 'Discrete Input, Discrete Output',
    (False, True): 'Continuous Input, Real Output',
    (False, False): 'Continuous Input, Discrete Output'
}

# Plot Learning Time
for key, color in colors.items():
    subset = results_df[(results_df['Discrete Input'] == key[0]) & (results_df['Real Output'] == key[1])]
    axs[0].plot(subset['N'], subset['Learning Time Avg'], color=color, label=labels[key])

axs[0].set_xlabel('N (Number of Samples)')
axs[0].set_ylabel('Average Learning Time (s)')
axs[0].set_title('Learning Time vs N for Different M')
axs[0].legend()

# Plot Prediction Time
for key, color in colors.items():
    subset = results_df[(results_df['Discrete Input'] == key[0]) & (results_df['Real Output'] == key[1])]
    axs[1].plot(subset['N'], subset['Prediction Time Avg'], color=color, label=labels[key])

axs[1].set_xlabel('N (Number of Samples)')
axs[1].set_ylabel('Average Prediction Time (s)')
axs[1].set_title('Prediction Time vs N for Different M')
axs[1].legend()

plt.show()