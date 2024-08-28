# Human Activity Recognition (HAR) Project

## Introduction

This project focuses on Human Activity Recognition (HAR) using accelerometer data. The goal is to accurately classify activities such as walking, sitting, standing, etc., based on accelerometer data. The project is divided into several tasks, each addressing a specific aspect of the recognition process.

## Project Structure

The project is organized into the following notebooks:

### Task Notebooks (located inside the `HAR/` folder)
- **[Task 1](HAR/Task1.ipynb)**:  Processed the dataset and plotted waveforms for one sample from each activity class to observe differences and similarities.
Evaluated the need for a machine learning model to distinguish between static and dynamic activities using linear acceleration. Visualized the data using PCA after applying various feature extraction methods.
- **[Task 2](HAR/Task2.ipynb)**: Trained a decision tree model using the `scikit-learn` library on three different data versions and compared their accuracy, precision, recall, and confusion matrix.
Additionally, trained decision trees with varying depths (2-8) and plotted the model accuracy on test data versus tree depth.
- **[Task 3](HAR/Task3.ipynb)**: Implemented Zero-Shot Learning (ZSL) and Few-Shot Learning (FSL) for classifying human activities based on featurized accelerometer data.
Quantitatively compared the accuracy of Few-Shot Learning with Decision Trees and identified the limitations of ZSL and FSL in this context. Also tested the model with random data. 
- **[Task 4](HAR/Task4.ipynb)**: Collected data at 50Hz with 3 samples of 10 seconds per activity using the `Physics Toolbox Suite` smartphone app.
Used the decision tree models trained on three different methods from Task 2 to predict activities. Also applied Few-Shot Learning for activity prediction and analyzed the performance.

### Decision Tree Notebook (located in the root folder)
- **[DecisionTree.ipynb](Decision_Tree.ipynb)**: Implemented and analyzed a decision tree model from scratch,
covering all four cases: i) discrete features, discrete output; ii) discrete features, real output; iii) real features, discrete output; iv) real features, real output.

## How to Use

1. Clone the repository.
2. Navigate to the respective folder to access the `.ipynb` files.
3. Run the notebooks in the provided order to replicate the results or modify them for further experimentation.

## Requirements

Ensure that you have the required Python packages installed.
