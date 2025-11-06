# Perceptron

A simple implementation of a single-layer Perceptron written in Python.  
This project demonstrates how a perceptron learns to classify data using a linear decision boundary.

---

## 📌 Overview

This repository contains:
- A minimal perceptron implementation from scratch  
- An Iris dataset classifier example  
- Clean, easy-to-understand code for beginners learning ML fundamentals  

The goal of this project is to show how classic perceptrons work internally without using any external machine-learning frameworks.

---

## 🧠 How It Works

The perceptron algorithm:

1. Initializes weights  
2. Computes prediction using:  
   `activation(weights · x + bias)`  
3. Compares the prediction with the true label  
4. Updates weights if the prediction was wrong  
5. Repeats for multiple epochs until convergence  

This algorithm is the foundation of modern neural networks.

---

## 📁 Project Structure

Perceptron/
│
├── perceptron.py # Core perceptron class and training logic
├── iris_type_predictor.py # Example using the Iris dataset
└── README.md

---

## ▶️ Usage

Run the perceptron demonstration:

```bash
python perceptron.py
Run the Iris dataset classifier:
python iris_type_predictor.py
Feel free to modify learning rate, epochs, or dataset to experiment.
🔧 Installation
Clone the repository:
git clone https://github.com/mgardzina/Perceptron.git
cd Perceptron
(Optional) Install dependencies if added later:
pip install -r requirements.txt
