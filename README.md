# Machine Learning From Scratch

A hands-on repository for understanding **Machine Learning by implementing algorithms from scratch** using Python and NumPy.

The goal is not just to use `scikit-learn`, but to understand what happens underneath the abstractions — from mathematical foundations and optimization to model training, prediction, and evaluation.

## 🎯 Objectives

* Understand the mathematics behind common ML algorithms.
* Implement algorithms without relying on high-level ML libraries.
* Build intuition by translating mathematical equations into code.
* Compare implementations against established libraries where useful.
* Write clean, modular, and testable implementations.
* Progress from classical ML toward more advanced machine learning concepts.

## 📚 Algorithms & Topics

The repository covers implementations such as:

* Linear Regression
* Logistic Regression
* K-Nearest Neighbors (KNN)
* Naive Bayes
* Support Vector Machines
* Decision Trees
* Random Forest
* PCA

More algorithms will be added progressively.

## 🧠 Philosophy

> **Don't just call the algorithm. Build it. Understand it. Break it. Fix it.**

For each algorithm, the focus is on understanding:

1. The mathematical formulation
2. The intuition behind the algorithm
3. The implementation
4. Training and prediction
5. Computational complexity
6. Limitations and assumptions
7. Comparison with standard implementations

The objective is to make the repository useful as both a **learning resource and a reference for ML fundamentals**.

## 🛠️ Tech Stack

* Python
* NumPy
* Pandas
* Matplotlib
* SciPy

High-level machine-learning libraries are intentionally avoided for the core implementations.

## 📁 Repository Structure

```text
ML-From-Scratch/
│
├── linear_regression/
├── logistic_regression/
├── knn/
├── naive_bayes/
├── svm/
├── decision_tree/
├── random_forest/
├── pca/
│
├── requirements.txt
└── README.md
```

The structure may evolve as new algorithms and experiments are added.

## 🚀 Getting Started

Clone the repository:

```bash
git clone <your-repository-url>
cd ML-From-Scratch
```

Create a virtual environment:

```bash
python -m venv .venv
```

Activate it on Windows:

```powershell
.venv\Scripts\activate
```

Or on Linux/macOS:

```bash
source .venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Run the test suite:

```bash
pytest
```

## 🔬 Example

A typical implementation follows the underlying mathematical idea rather than wrapping an existing ML library.

For example, instead of:

```python
from sklearn.linear_model import LinearRegression
```

the repository implements the learning procedure directly, allowing the reader to inspect how parameters are initialized, optimized, and used for prediction.

## 📈 Learning Path

A suggested progression through the repository:

```text
Mathematics
    ↓
Linear Regression
    ↓
Gradient Descent
    ↓
Logistic Regression
    ↓
KNN / Naive Bayes
    ↓
Decision Trees
    ↓
Random Forest
    ↓
SVM
    ↓
PCA

```

## 🤝 Contributing

Contributions are welcome.

If you find a bug, mathematical error, implementation improvement, or have an idea for another from-scratch algorithm, feel free to open an issue or submit a pull request.

## 📌 Why This Repository?

Modern ML frameworks make it possible to train sophisticated models with only a few lines of code. That's powerful — but abstraction can hide the machinery.

This repository is an attempt to go one layer deeper.

**The aim is simple: understand the algorithm before trusting the abstraction.**

## 📜 License

This project is intended for educational and research purposes. Add the appropriate license for your repository here.
