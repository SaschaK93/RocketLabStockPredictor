🚀 RocketLabStockPredictor

A machine learning project that predicts the next-day direction of Rocket Lab (RKLB) stock using technical indicators, financial news sentiment analysis, and multiple machine learning models.

📈 Project Overview

The goal of this project is to investigate whether stock price movements can be predicted using:

Technical indicators
Market context
Trading volume
Financial news sentiment
Machine learning models

The project uses Rocket Lab (RKLB) as the target stock and compares its performance against the broader market.

🛠 Technologies Used
Data Sources
Yahoo Finance (historical stock data)
Finnhub API (financial news)
Machine Learning
Logistic Regression
Random Forest
NLP
FinBERT (ProsusAI/FinBERT)
Python Libraries
pandas
numpy
scikit-learn
yfinance
transformers
matplotlib
📊 Features
Technical Indicators
RSI (Relative Strength Index)
Moving Average (MA30)
Exponential Moving Average (EMA30)
Volatility
Price vs EMA
Volume Features
Volume Moving Average
Volume Ratio
Volume Trend Features
Market Features
SPY Daily Returns
Market Outperformance
News Sentiment Features

Financial news headlines and summaries are analyzed using FinBERT.

Pipeline:

Financial News → FinBERT → Daily Sentiment Score → Machine Learning Model

🤖 Models Tested
Logistic Regression

Used as a baseline model for binary classification.

Random Forest

Used to capture non-linear relationships between features.

📈 Evaluation Metrics

The project evaluates model performance using:

Accuracy
Precision
Recall
F1 Score
ROC-AUC
Cross Validation
🔍 Key Findings
Technical indicators provide useful predictive signals.
Volume-based features showed measurable influence.
Financial news sentiment contributed modestly to model performance.
Predicting next-day stock direction remains extremely challenging.
Random Forest slightly outperformed Logistic Regression.
📂 Project Structure
RocketLabStockPredictor/
│
├── news_downloader.py
├── news_processing.py
├── stock_analyzer_v2.py
├── README.md
🚀 Future Improvements
QQQ and IWM market features
Deep Learning models
Social media sentiment analysis
Hype Stock Finder integration
Portfolio-level predictions
Disclaimer

This project was created for educational and portfolio purposes only.

It does not constitute financial advice.
