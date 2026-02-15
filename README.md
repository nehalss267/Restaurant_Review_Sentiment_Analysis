# Restaurant Review Sentiment Analysis

This project is a Natural Language Processing (NLP) implementation that predicts whether a restaurant review is **Positive** or **Negative**. It utilizes the **Multinomial Naive Bayes** algorithm to classify text data after processing it with various cleaning and lemmatization techniques.

## Project Overview

* **Goal:** Classify text reviews into binary sentiment categories (0 = Negative, 1 = Positive).
* **Algorithm:** Multinomial Naive Bayes (Probabilistic learning).
* **Feature Extraction:** Bag of Words (CountVectorizer).
* **Preprocessing:** NLTK (Stopword removal, Lemmatization, Regex cleaning).

## Dataset

The project relies on a dataset named `Restaurant_Reviews.tsv`.
* **Format:** Tab Separated Values (TSV).
* **Structure:** Two columns — `Review` (text) and `Liked` (0 or 1).
* **Quoting:** The code handles quoting (`quoting=3`) to avoid parsing errors.

## Key Libraries Used:

* **Pandas & Numpy:** Data manipulation and array processing.

* **NLTK (Natural Language Toolkit):** Used for removing stopwords and lemmatization.

* **Scikit-Learn:** Used for the Bag of Words model (CountVectorizer), the Classifier (MultinomialNB), and evaluation metrics.

* **Matplotlib & Seaborn:** For visualizing the Confusion Matrix.

## Methodology
* 1. Data Preprocessing
Raw text data is noisy. The following steps are taken to clean it:

* **Regex Cleaning:** Removing non-alphabetic characters (punctuation, numbers, emojis).

* **Lowercasing:** Converting all text to lowercase for uniformity.

* **Stopword Removal:** Common English words (e.g., "the", "is", "in") are removed to reduce noise.

Note: The word "not" is explicitly excluded from the stopword list to preserve negative context (e.g., "not good").

* **Lemmatization:** Converting words to their base root form (e.g., "loved" -> "love") using WordNetLemmatizer.

* 2. Feature Engineering
Bag of Words Model: We use CountVectorizer to convert text into a matrix of token counts.

* **Max Features:** Limited to the top 1,500 most frequent words to optimize performance and reduce dimensionality.

* 3. Model Training
Split: The dataset is split 80% for training and 20% for testing.

Classifier: Multinomial Naive Bayes is used, which is highly effective for text classification with discrete features.

* 4. Hyperparameter Tuning
The script iterates through alpha values (smoothing parameter) from 0.1 to 1.0 to find the model with the highest accuracy.

## Evaluation Results
The model performance is evaluated using:

* **Accuracy Score**

* **Precision & Recall**

* **Confusion Matrix:** Visualized using a Seaborn Heatmap.

## Usage
* **Load the Data:** Ensure the file path to the .tsv file is correct in the script.

* **Run the Script:** Execute the Python script or Jupyter Notebook.

* **Make Predictions:** The script includes a sentiment_analysis() function to test new, custom reviews.

Example Code
Python
sample_review = 'The food was absolutely wonderful'

if sentiment_analysis(sample_review):
    print("Positive Sentiment")
else:
    print("Negative Sentiment")

## 🛠 Prerequisites

To run this project, you need Python installed along with the following libraries:

```bash
pip install pandas numpy matplotlib seaborn nltk scikit-learn
