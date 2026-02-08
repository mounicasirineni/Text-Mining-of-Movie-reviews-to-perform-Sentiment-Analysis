# Text Mining of Movie Reviews for Sentiment Analysis

> **📚 Course Project** | IDS 572: Data Mining for Business  
> **🏫 University of Illinois at Chicago**  
> **⏰ Completed:** Fall 2015  
> **📝 Last Updated:** February 2026 (documentation only)  
> **🔍 [View Development Timeline](../../commits/main)** - See original commits from Fall 2015

---

## Overview

Built a sentiment classification system to analyze movie reviews and predict positive/negative sentiment using natural language processing and machine learning techniques. Completed as part of the Data Mining for Business course, this project implemented comprehensive text preprocessing pipeline and compared multiple classification algorithms.

**Course:** IDS 572 - Data Mining for Business  
**Technologies:** Python, NLTK, Scikit-learn, TF-IDF, Pandas

---

## Technical Approach

### 1. Text Preprocessing Pipeline
- **Tokenization:** Split review text into individual words
- **Stopword Removal:** Removed common words ("the", "is", "and") that don't carry sentiment
- **Stemming:** Reduced words to root form using Porter Stemmer
- **TF-IDF Vectorization:** Converted text to numerical features weighted by importance
- **N-gram Features:** Extracted unigrams and bigrams for better context capture

### 2. Classification Algorithms

**Naive Bayes:**
- Implemented Multinomial Naive Bayes for text classification
- Leveraged probabilistic approach for fast training
- Baseline model for comparison

**Support Vector Machines (SVM):**
- Used linear SVM with TF-IDF features
- Tuned hyperparameters (C, kernel) for optimal decision boundary
- Best overall performance

**Logistic Regression:**
- Applied L1/L2 regularization to prevent overfitting
- Interpretable coefficients for feature importance analysis
- Good balance of accuracy and speed

### 3. Evaluation
- Accuracy, Precision, Recall, F1-Score
- Confusion matrix analysis
- Cross-validation for robust performance estimates
- Feature importance analysis

---

## Results

- Achieved **85%+ accuracy** in sentiment classification
- **SVM with TF-IDF** performed best overall
- Bigram features improved performance over unigrams alone
- Successfully identified key sentiment-bearing words and phrases

**Key Findings:**
- Preprocessing (stemming, stopword removal) significantly impacted model performance
- TF-IDF outperformed simple word count vectorization
- SVM showed best generalization to unseen reviews

---

## Key Learnings

1. **Text Preprocessing:** Critical importance of cleaning and normalizing text data before modeling
2. **Feature Engineering:** TF-IDF captures word importance better than raw frequency counts
3. **Algorithm Selection:** Different classifiers excel with different text representations
4. **Vocabulary Management:** Handling large vocabularies and rare words effectively
5. **NLP Fundamentals:** Foundation for understanding modern language models and transformers

---

## Skills Demonstrated

- Natural Language Processing (NLP)
- Sentiment Analysis
- Text Mining
- TF-IDF Vectorization
- Machine Learning Classification
- Feature Engineering
- Python Programming
- Model Comparison

---

## Academic Context

This project was part of **IDS 572: Data Mining for Business** at UIC, completed in **Fall 2015**. This work demonstrated early hands-on experience with NLP and text classification. These fundamental techniques—tokenization, TF-IDF, and classification algorithms—form the basis of modern language understanding systems, including the transformers and large language models developed years later.

---

**[← Back to Portfolio](https://github.com/mounicasirineni/masters-ml-portfolio)**
