# 📘 Iris Type Classification

A machine learning model to classify the type of Iris flower based on input features using historical data.

## 🔍 Overview

This project focuses on classifying type of Iris flowers based on input features. By utilizing historical features Sepal length, Sepal width, Petal length and Petal width, the model identifies patterns to classify the type of iris into on of the three categories ; Setisoa, Versicolor and virginica. The project employs Logistic Regression model for classification, with a focus on achieving high accuracy.

## 📁 Project Structure

├── data/                # Raw and processed
├── models/              # Trained machine learning models
├── notebooks/           # Jupyter notebooks for exploratory data analysis (EDA), run_pipeline
├── src/                 # Source code for data preprocessing, model training, model evaluation, model pipeline
├── tests/               # Unit tests for individual components
├── .gitignore/          # Untracked files that Git ignore
├── requirements.txt     # Project dependencies
└── README.md            # Project documentation
└── streamlit-app.py     # Streamlit application

## 📊 Data

-- Source: Iris dataset referred by CodeAlpha.(Downloaded from Kaggle) https://www.kaggle.com/datasets/saurabh00007/iriscsv

-- Columns: ID, Sepal length, Sepal width, Petal length, Petal width, Species

-- Preprocessing: Data cleaning, Removing unwanted columns, Label Encoding, Scaling

## 🧠 Modeling

This project uses Linear Logistic Regression method to classify Iris type as either Setisoa, Versicolor or Virginica. Model classification report was generarted to show the performance evaluated using metrics such as precision, recall, F1 score, and accuracy.

## 🚀 Installation & Usage

### Install dependencies

pip install -r requirements.txt

### Run the project using Streamlit
To use the streamlit application run the below in terminal:

streamlit run .\streamlit-app.py

### To run Jupyter notebooks for experimentation:

jupyter notebook

## ✅ Testing

To ensure the quality of the code, run the unit tests from terminal using:

pytest

## 📈 Results
The final model achieves an accuracy of 100%, showing its effectiveness in detecting the correct Iris flower. This accuracy may be considered to be overfitting in other cases, but as the dataset is too simple and small it's approved.

Simple EDA for the dataset can be found in eda.ipynb, preprocessing, training, evaluation and streamlit app generation can be done by running run_pipeline.ipynb.


## 🗒️ Notes / Future Work

The model's training, testing and deployment was done on around 150 row as downloaded from Kaggle. Future work maybe a kind of image classification of Iris type using computer vision.


## 👤 Author
Mohammad Ezzeddine
github.com/Mhmd-14| in/mohammad-ezzeddine-14ae1