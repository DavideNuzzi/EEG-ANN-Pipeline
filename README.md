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
- data/: Contains sample EEG datasets and related resources.
- dataset_bci/: Includes scripts and data specific to brain-computer interface (BCI) datasets.
- examples/: Provides example scripts demonstrating how to use the pipeline for various analyses.
- helpers/: Utility functions to support data preprocessing and other auxiliary tasks.
- metrics/: Implements evaluation metrics for model performance assessment.
- models/: Defines neural network architectures and related model components.
- tdlm/: Contains scripts related to temporally delayed linear modeling (TDLM) for sequence analysis.

## Getting Started
1) Clone the Repository:
```bash
git clone https://github.com/DavideNuzzi/EEG-ANN-Pipeline.git
cd EEG-ANN-Pipeline
```
2) Install Dependencies
```bash
pip install -r requirements.txt
```
3) Run Examples:
Navigate to the examples/ directory and choose one of the example notebooks.

## Usage
The pipeline is designed to be modular and flexible. Users can:
- Customize data preprocessing steps.
- Select or define neural network architectures.
- Choose specific analysis tools based on their research needs.
Detailed documentation and tutorials are available in the examples/ directory to guide users through various use cases.

## Contributing
Contributions are welcome! If you have suggestions for improvements or new features, feel free to open an issue or submit a pull request.

## License
Shield: [![CC BY 4.0][cc-by-shield]][cc-by]

This work is licensed under a
[Creative Commons Attribution 4.0 International License][cc-by].

[![CC BY 4.0][cc-by-image]][cc-by]

[cc-by]: http://creativecommons.org/licenses/by/4.0/
[cc-by-image]: https://i.creativecommons.org/l/by/4.0/88x31.png
[cc-by-shield]: https://img.shields.io/badge/License-CC%20BY%204.0-lightgrey.svg
