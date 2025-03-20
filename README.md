# MacroHFT Implementation on Stocks

This repository contains an implementation of a MacroHFT (Hierarchical Reinforcement Learning for High-Frequency Trading) model applied to stock market data. The model trains agents to make buy/sell/hold decisions based on historical stock data using reinforcement learning.

## Features
- Implements Hierarchical Reinforcement Learning (HRL) with Macro and Micro agents
- Utilizes technical indicators for market trend and volatility classification
- Supports training on stock data and evaluating trading strategies
- Implements Deep Q-Networks (DQN) for decision-making

## Installation

Clone the repository and install the required dependencies:

```bash
git clone https://github.com/Agnibho-Saha14/MacroHFT-implementation-on-stocks.git
cd MacroHFT-implementation-on-stocks
pip install -r requirements.txt
```

## Usage

Follow these steps to train and evaluate the MacroHFT model:

### 1. Label Training Data
Run the `Train_Labeling.py` script to label the training dataset:
```bash
python Train_Labeling.py
```
This script assigns market trend and volatility labels to the dataset.

### 2. Train the Model
Run the `Training.py` script to train the MacroHFT model:
```bash
python Training.py
```
The script will train both Macro and Micro agents using reinforcement learning and save the trained models.

### 3. Preprocess and Label Stock Data
Use `stock_preprocessing.py` to preprocess stock data and generate technical indicators:
```bash
python stock_preprocessing.py
```
Enter the stock ticker when prompted to preprocess and label the data.

### 4. Evaluate the Model
Run the `evaluation.py` script to evaluate the trained models on a selected stock:
```bash
python evaluation.py
```
Enter the stock ticker when prompted, and the script will generate trading signals and a strategy performance plot.

## Project Structure
```
├── Dataset/
│   ├── Training/            # Contains labeled training data
│   ├── Evaluation/          # Contains preprocessed stock data for evaluation
│
├── Train_Labeling.py        # Labels training data with market trend and volatility
├── Training.py              # Trains the MacroHFT model
├── stock_preprocessing.py   # Preprocesses stock data and computes technical indicators
├── evaluation.py            # Evaluates the trained model
├── requirements.txt         # Required Python libraries
└── README.md                # Project documentation
```

## Results
After training and evaluation, the model outputs:
- Trading signals (`BUY`, `SELL`, `HOLD`)
- A CSV file containing predicted trading actions
- A performance plot comparing strategy returns to initial capital



