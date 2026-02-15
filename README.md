# Rust Naive Bayes Spam Filter

A Rust implementation of a Naive Bayes classifier for email spam filtering. This project re-implements the concepts and algorithms from the "Email Spam Filter in Go" stream by Tsoding Daily.

## Overview

This project demonstrates a Naive Bayes classifier for distinguishing between legitimate emails (ham) and unsolicited emails (spam). It uses a probabilistic and statistical model to classify new emails based on trained data.

## How It Works

The spam filter is built on Bayes' Theorem, specifically its application in Naive Bayes classifiers for text classification:

1. **Dataset Preparation**: Gather two sets of emails - known spam and known ham messages.

2. **Tokenization**: Break down email content into individual words or tokens. The implementation splits text by whitespace.

3. **Frequency Analysis**: Count word occurrences in both spam and ham datasets to build a "bag of words" for each class.

4. **Probability Calculation**: Determine the probability of words appearing in spam versus ham emails. The system uses log probabilities to prevent underflow issues during calculations.

5. **Classification**: For a new email, calculate the probability of it being spam versus ham, and classify it based on a defined threshold.

## Dataset

The project uses the Enron Spam Dataset for training and evaluation. More information about the dataset is available at the link in the references section below.

## Key Features

- **File Reading**: Efficiently reads entire email files for processing
- **Tokenization and Frequency Analysis**: Converts raw email text into word frequencies
- **Directory Analysis**: Analyzes entire directories of emails to build comprehensive statistical models
- **Log Probabilities**: Employs logarithmic probabilities to handle very small values without loss of precision
- **Modular Design**: Functions for tokenization, frequency analysis, and classification are structured for clarity and reusability

## Usage

Clone the repository:

```bash
git clone git@github.com:ishworii/Email-Spam-Filter.git
cd Email-Spam-Filter
```

Prepare your dataset by placing spam and ham email files in designated directories.

Build and run:

```bash
cargo run
```

The main logic demonstrates how to train the model on ham and spam directories and then classify individual files or entire directories.

## Why Naive Bayes?

Naive Bayes is a simple yet powerful probabilistic classifier and a classical algorithm often taught in computer science courses. It's particularly well-suited for spam filtering tasks. The "naive" assumption is that all features (words) are independent of each other, which simplifies calculations while still producing effective results.

## Future Plans

This implementation could serve as a foundation for other probabilistic text classification tasks, such as sentiment analysis, intent classification for chatbots, or other natural language processing applications.

## References

- **Original Stream**: [Email Spam Filter in Go by Tsoding Daily](https://www.youtube.com/watch?v=JsfOXk7qmSM)
- **Naive Bayes Classifier**: https://en.wikipedia.org/wiki/Naive_Bayes_classifier
- **Tsoding's Haskell Implementation**: https://github.com/tsoding/vetcheena
- **Enron Spam Dataset**: https://www2.aueb.gr/users/ion/data/enron-spam/
