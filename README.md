# Restaurant Review Sentiment Analysis

This project is a Natural Language Processing (NLP) implementation that predicts whether a restaurant review is **Positive** or **Negative**. It utilizes the **Multinomial Naive Bayes** algorithm to classify text data after processing it with various cleaning and lemmatization techniques.

## 📌 Project Overview

* **Goal:** Classify text reviews into binary sentiment categories (0 = Negative, 1 = Positive).
* **Algorithm:** Multinomial Naive Bayes (Probabilistic learning).
* **Feature Extraction:** Bag of Words (CountVectorizer).
* **Preprocessing:** NLTK (Stopword removal, Lemmatization, Regex cleaning).

## 📂 Dataset

The project relies on a dataset named `Restaurant_Reviews.tsv`.
* **Format:** Tab Separated Values (TSV).
* **Structure:** Two columns — `Review` (text) and `Liked` (0 or 1).
* **Quoting:** The code handles quoting (`quoting=3`) to avoid parsing errors.

## 🛠 Prerequisites

To run this project, you need Python installed along with the following libraries:

```bash
pip install pandas numpy matplotlib seaborn nltk scikit-learn
