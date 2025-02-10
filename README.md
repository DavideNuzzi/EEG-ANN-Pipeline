# EEG-ANN-Pipeline

## Overview

The **EEG-ANN-Pipeline** is a Python-based framework designed to facilitate the analysis of electroencephalography (EEG) data using artificial neural networks (ANNs). 
This pipeline streamlines the process from data preprocessing to model evaluation, making it accessible for researchers aiming to integrate machine learning techniques into EEG studies.

## Features

- **Data Handling**: Supports importing popular EEG data formats and automates formatting for machine learning model compatibility.
- **Feature Selection**: Identifies informative time windows and relevant EEG channels to enhance predictive modeling.
- **Modeling**: Offers customizable deep learning models, including convolutional neural networks (CNNs) and long short-term memory networks (LSTMs), for both decoding (classification) and encoding (representation learning) tasks.
- **Analysis Tools**: Provides tools for representational similarity analysis (RSA) to compare neural activity patterns and sequence analysis to study reactivations in spontaneous activity.

## Repository Structure
EEG-ANN-Pipeline/
│── data/               # Sample EEG datasets and related resources
│── dataset_bci/        # Scripts and data specific to brain-computer interface (BCI) datasets
│── examples/           # Example scripts demonstrating how to use the pipeline
│── helpers/            # Utility functions for data preprocessing and other tasks
│── metrics/            # Implements evaluation metrics for model performance
│── models/             # Neural network architectures and related components
│── tdlm/               # Scripts for Temporally Delayed Linear Modeling (TDLM)
│── requirements.txt    # Dependencies for running the project
│── README.md           # Project documentation





