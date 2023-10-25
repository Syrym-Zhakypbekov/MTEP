# MTEP Multilingual_Translation_Emotion_Project

## Table of Contents

1. [Introduction](#introduction)
2. [Features](#features)
3. [System Requirements](#system-requirements)
4. [Installation](#installation)
5. [Configuration](#configuration)
6. [Usage](#usage)
7. [Benchmarks](#benchmarks)
8. [Methods](#methods)
9. [Models](#models)
10. [Data](#data)
11. [Logs](#logs)
12. [Labels](#labels)
13. [Scripts](#scripts)
14. [Docs](#docs)
15. [Metadata](#metadata)
16. [Tests](#tests)
17. [Security](#security)
18. [Tools](#tools)
19. [Frameworks and Toolkits(#Frameworks and Toolkits)]
20. [Contributing](#contributing)
21. [License](#license)
22. [Changelog](#changelog)
23. [Domain and Task](#domain-and-task)
24. [Parameters/Hyperparameters](#parametershyperparameters)

## Introduction

Development of a Multilingual Method for Automatic Translation and Emotional Synthesis of Speech in Audio-Visual Media to Kazakh language.

## Features

- Audio Extraction
- Segmenting Video into Frames
- Speech-to-Text Conversion
- Text Tokenization
- Sentiment Analysis
- Text Translation
- Emotion Classification
- Text-to-Speech Conversion
- Transferring Voice Styles
- Feature Extraction
- Learning Joint Representations

## System Requirements

Refer to [System_Specs](System_Specs/) for system requirements.

- **Processor**: 13th Gen Intel(R) Core(TM) i9-13980HX (32 CPUs), ~2.2GHz
- **Memory**: Installed RAM: 32 GB. Available RAM: 32.386 GB.
- **GPUs**: NVIDIA GeForce RTX 4090 Laptop GPU with 16 GB dedicated memory.
- **Operating System**: Windows 11 Pro 64-bit.

## Installation

Instructions to install dependencies and set up the project. Refer to [Dockerfile](Dockerfile) and [docker-compose.yml](docker-compose.yml) for Docker-based setup.

## Project Execution Steps

### Step 1: Environment Setup
- Clone the repository.
- Install the required packages via `pip install -r requirements.txt`.

### Step 2: Data Preparation
- Run `Preprocessing_EN.py` and `Preprocessing_KZ.py` from the `Scripts/Data_Preparation/` directory.

### Step 3: Feature Extraction
- Run `Audio_Feature_Extraction.py` and `Video_Feature_Extraction.py` from the `Scripts/Feature_Extraction/` directory.

### Step 4: Model Training
- Execute all the training scripts in `Scripts/Model_Training/`.

### Step 5: Model Evaluation
- Execute all evaluation scripts in `Scripts/Model_Evaluation/`.

### Step 6: Benchmarking
- View performance metrics in `Benchmarks/performance_metrics.csv`.

### Step 7: Model Deployment
- Use `Model_Deployment.py` from the `Scripts/Utilities/` directory.

### Step 8: Tests
- Run `test_data_preparation.py` and `test_feature_extraction.py` from the `Tests/unit_tests/` directory.

### Step 9: Documentation
- Consult `Docs/` for project and model documentation.

### Step 10: Notebook Exploration
- Open `Data_Exploration.ipynb` and `Model_Prototyping.ipynb` from the `Notebooks/` directory for further data and model exploration.


## Configuration

Configuration details are stored in [.env](.env). Modify as needed.

## Usage

How to use the project. Include any [Notebooks](Notebooks/) for demo purposes.

## Benchmarks

Performance benchmarks are located in [Benchmarks](Benchmarks/).

## Methods

Describe the algorithms and methods used in [Methods](Methods/).

- Signal Processing
- Computer Vision Algorithms
- Deep Learning
- Machine Learning
- Neural Networks
- Supervised Learning

## Models

Details about the models used are in [Models](Models/).

### Speech Recognition
- **DeepSpeech**: End-to-end deep learning-based speech recognition.
- **Wav2Vec 2.0**: Self-supervised speech recognition.

### Text-to-Speech
- **Tacotron 2**: Sequence-to-sequence model for human-like speech.
- **WaveGlow**: Converts mel-spectrograms to audio.

### Translation and NLP
- **Transformer**: For machine translation and NLP.
- **BERT**: Text analysis and understanding.
- **GPT-3**: Text generation and translation.

### Emotion Recognition
- **EmoReact**: Emotion recognition from speech.
- **AffectNet**: Emotion recognition from facial expressions.

### Video Processing
- **Optical Flow Models**: Motion detection and video segmentation.

### Multimodal Models
- **MMBT**: Text and image data tasks.
- **ViLBERT**: Visual and textual data tasks.

### Data Fusion
- **Fusion Models**: Combining audio, video, and text features.

### Audio Signal Processing
- **CRNN**: Audio classification and feature extraction.

### Others
- **StarGAN-VC**: Voice conversion and style transfer.
- **WaveGlow**: Generating speech from mel-spectrograms.


## Data

Information about the dataset in [Data](Data/).

- Instance trained on audio-visual dataset
- Instance trained on video dataset
... (list other instances)

## Logs

Logs are stored in [Logs](Logs/).

## Labels

Label information can be found in [Labels](Labels/).

## Scripts

Utility scripts are in [Scripts](Scripts/).

## Docs

Additional documentation is in [Docs](Docs/).

## Metadata

Metadata related to the project is in [Metadata](Metadata/).

## Tests

Unit tests are in [Tests](Tests/).

## Security

Security guidelines are in [SECURITY.md](SECURITY.md).

## Tools

Refer to [Tools](Tools/) for any additional tooling.

## Frameworks and Toolkits

### General Machine Learning Frameworks
- **PyTorch or TensorFlow**: Extensive libraries and community support.
- **Scikit-learn**: For simpler machine learning tasks and data preprocessing.

### Speech Recognition
- **ESPnet**: End-to-end speech processing.
- **Kaldi**: A toolkit for speech recognition.

### Text-to-Speech
- **Tacotron 2**: High-quality speech output.
- **WaveGlow**: A flow-based generative network for speech synthesis.

### Audio Signal Processing
- **Librosa**: For audio feature extraction.
- **SoX (Sound eXchange)**: A command-line utility to convert, process, and play audio files.

### Video Processing
- **OpenCV**: For video processing tasks.
- **FFmpeg**: For video and audio conversions.

### Natural Language Processing
- **Hugging Face Transformers**: Pre-trained models.
- **NLTK (Natural Language Toolkit)**: For text analytics and preprocessing.

### Emotion Recognition
- **OpenSmile**: Feature extraction for audio-based emotion recognition.
- **Affectiva SDK**: Emotion recognition from facial expressions.

### Multimodal Learning
- **MMF (Multimodal Framework by Facebook)**: For joint representations.
- **TensorFlow.js**: If you plan to deploy models in web applications.

### Data Augmentation
- **Augmentor**: For audio and video.
- **Albumentations**: For image augmentation in video frames.

### Model Deployment
- **TensorFlow Serving or TorchServe**: For serving models.
- **Docker**: For containerizing your application.

### Configuration and Experiment Tracking
- **Hydra**: For dynamic configuration.
- **MLflow**: For experiment tracking and model versioning.

### Data Pipelining
- **DVC (Data Version Control)**: For data versioning.
- **Apache NiFi**: For data flow automation.

### CUDA and GPU Acceleration
- **CUDA Toolkit**: For GPU-accelerated computation.
- **cuDNN (CUDA Deep Neural Network library)**: A GPU-accelerated library for deep neural networks.

### Data Storage
- **HDF5**: For storing large datasets efficiently.
- **SQLite**: For lightweight, disk-based storage.

### Collaboration and Version Control
- **Git**: For source code version control.
- **Jupyter Notebook**: For sharing code, equations, and visualizations.


## Contributing

Please read [CONTRIBUTING.md](CONTRIBUTING.md) for details on contributions.

## License

This project is licensed under the terms of the LICENSE found in [LICENSE.md](LICENSE.md).

## Changelog

See [CHANGELOG.md](CHANGELOG.md) for version history.

## Domain and Task

- **Domain**: Audio-Visual Processing, Speech Processing, NLP, Machine Translation, etc.
- **Task**: Audio Extraction, Segmenting Video into Frames, Speech-to-Text Conversion, etc.

## Parameters/Hyperparameters

- Extraction parameters, Feature dimensions
- Learning rate, Batch size, Model architecture for each model

