# Temporal Relation Classification in Hebrew

## Project Overview
This project focuses on the development of a baseline model for Temporal Relation Classification (TRC) in Hebrew. It involves preprocessing datasets and training a model capable of understanding and classifying temporal relations in Hebrew text.

## Project Structure
The project is structured into three main components:

### 1. Datasets
The dataset component is divided into two subfolders: `raw-data` and `training-data`.

- `raw-data`: Contains the initial dataset in JSON format.
  - `TRC_data.json`: The raw dataset file in JSON format.

- `training-data`: Includes the processed dataset in CSV format and further split for training.
  - `TRC_dataset.csv`: The processed dataset derived from `TRC_data.json`.
  - `split`: This folder contains the dataset split for training and testing purposes.
    - `train.csv`: Dataset used for training the model.
    - `test.csv`: Dataset used for testing the model.

### 2. Model (`trc_model`)
This directory contains the Python modules for the TRC model.
- `__init__.py`: Initialization file for the Python package.
- `temporal_relation_classification.py`: Core module for temporal relation classification.
- `temporal_relation_classification_config.py`: Configuration settings for the TRC model.
- `temporal_relation_classification_pipeline.py`: Pipeline module for processing and classifying data.

### 3. Model Training (`trc_model_training.ipynb`)
This Jupyter Notebook contains the code and procedures for training the TRC model. It demonstrates the entire process from data loading, preprocessing, model configuration, training, and evaluation.

## Dataset Explanation
- `TRC_data.json`: This is the primary dataset file in JSON format. It contains annotated data necessary for understanding and classifying temporal relations in Hebrew text.
- `TRC_dataset.csv`: A CSV file created from `TRC_data.json`. It's a structured format of the dataset, making it easier for the model to process and learn from.
- `train.csv` and `test.csv`: These files are parts of the `TRC_dataset.csv` split for the purpose of training and testing the model, ensuring a robust and accurate learning process.

## Model Training
The model training process is detailed in the `trc_model_training.ipynb` Jupyter Notebook. It includes:
- Data Loading: Reading the dataset and understanding its structure.
- Data Preprocessing: Preparing the data for training, including any necessary transformation or normalization.
- Model Configuration: Setting up the model with appropriate parameters and configurations for the task.
- Training Process: The step-by-step process of feeding the data into the model, including the training algorithm and optimization techniques.
- Evaluation: Assessing the performance of the model on the test dataset to ensure accuracy and reliability.
