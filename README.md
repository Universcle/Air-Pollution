# Air Quality Analysis Web App

This project is a Flask-based web application for analyzing air quality data through classification, regression, and clustering techniques. The app allows users to interactively select features, classifiers, or regression methods to generate predictions and visualize results.

---

## Dataset

This project uses the **[Air Quality and Pollution Assessment](https://www.kaggle.com/datasets/mujtabamatin/air-quality-and-pollution-assessment)** dataset, which is made available under the **Apache 2.0 License**.

---

## Screenshots

<p float="left">
  <img src="assets/homepage.png" alt="Homepage" width="45%">
  <img src="assets/clustering.png" alt="Clustering" width="45%">
</p>

<p float="left">
  <img src="assets/classification.png" alt="Classification" width="45%">
  <img src="assets/regression.png" alt="Regression" width="45%">
</p>

---

## Getting Started

### Prerequisites

- Python 3.7 or higher
- Virtual environment (recommended)

### Installation

1. **Clone the repository**:

   ```bash
   git clone https://github.com/Universcle/Air-Pollution.git
   cd Air-Pollution
   ```

2. **Set up the virtual environment**:
   ```bash
   python -m venv venv
   source venv/bin/activate  # macOS/Linux
   venv\Scripts\activate     # Windows
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Run the application**:
   ```bash
   python app.py
   ```

5. Open your browser and go to [http://127.0.0.1:5000](http://127.0.0.1:5000).
