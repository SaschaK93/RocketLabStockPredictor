# ---------------------------------------------------
# 1. Import Libraries
# ---------------------------------------------------

# Data Handling
import yfinance as yf
import pandas as pd
import numpy as np

# Modeling Libraries
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

# Evaluation Metrics
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import cross_val_score

# Preprocessing
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

# Visualization
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.metrics import roc_curve

# News API
import requests
from datetime import date, timedelta
from transformers import pipeline
from datetime import datetime

# ---------------------------------------------------
# 2. Choose Stock Tickers
# ---------------------------------------------------

tickers = ["RKLB", "SPY"]   # RKLB is the target stock, SPY is the S&P 500 ETF for market context

# ---------------------------------------------------
# 2.1 Financial Sentiment Analysis with FinBERT
# ---------------------------------------------------

# Load Financial Sentiment Analysis Results from news_processing.py
news_df = pd.read_csv(r"C:\Users\sasch\OneDrive\Desktop\Python\Projekte\AI Finance\rklb_news_with_sentiment.csv")

# Convert the Datetime Column from Unix Timestamp to Readable Date Format and Extract Only the Date Part for Grouping
news_df["Date"] = pd.to_datetime(news_df["datetime"],unit="s").dt.date

# Group the News Data by Date and Calculate the Average Sentiment Value for Each Day to Create a Daily Sentiment Score that Can Be Merged with Stock Price Data for Analysis
daily_sentiment = (news_df.groupby("Date")["Sentiment_Value"].mean().reset_index())

# Convert the Date Column to Datetime Format and Set it as the Index of the DataFrame to Facilitate Merging with Stock Price Data Based on Date
daily_sentiment["Date"] = pd.to_datetime(daily_sentiment["Date"])

# Set the Date Column as the Index of the DataFrame to Facilitate Merging with Stock Price Data Based on Date
sentiment_df = daily_sentiment.set_index("Date")

print("\n--- Daily Sentiment DataFrame ---")
print(sentiment_df.head())
print("Daily Sentiment Days:", len(sentiment_df))

# ---------------------------------------------------
# 3. Download Stock Data
# ---------------------------------------------------

stocks = yf.download(tickers, period="5y")  # Download 5 years of daily data for the selected tickers

# ---------------------------------------------------
# 4. Calculate Daily Returns and Volatility
# ---------------------------------------------------

returns = stocks["Close"].pct_change()      # Calculate daily percentage change in closing prices
volatility = (returns.rolling(30).std())    # Calculate rolling volatility (standard deviation of returns) over a 30-day window

# ---------------------------------------------------
# 5. Create RKLB DataFrame
# ---------------------------------------------------

rklb = stocks["Close"][["RKLB"]].copy()     # Create a DataFrame for RKLB closing prices
rklb["Volume"] = stocks["Volume"]["RKLB"]   # Add the trading volume for RKLB to the RKLB DataFrame

# ---------------------------------------------------
# 6. Create Tomorrow Price and Target
# ---------------------------------------------------

rklb["Tomorrow"] = (rklb["RKLB"].shift(-1))                     # Create a new column "Tomorrow" that contains the closing price of RKLB for the next day by shifting the "RKLB" column up by one row
rklb["Target"] = (rklb["Tomorrow"] > rklb["RKLB"]).astype(int)  # Create a binary target variable "Target" that indicates whether the price of RKLB will go up (1) or down (0) the next day

# ---------------------------------------------------
# 7. Calculate RSI
# ---------------------------------------------------

delta = stocks["Close"].diff()          # Calculate daily price changes
gain = delta.where(delta > 0, 0)        # Separate gains (positive changes) from losses
loss = -delta.where(delta < 0, 0)       # Separate losses (negative changes) from gains and convert to positive values

avg_gain = (gain.rolling(14).mean())    # Calculate average gain over a 14-day window
avg_loss = (loss.rolling(14).mean())    # Calculate average loss over a 14-day window

rs = avg_gain / avg_loss                # Calculate Relative Strength (RS)
rsi = 100 - (100 / (1 + rs))            # Calculate Relative Strength Index (RSI) using the RS value

# ---------------------------------------------------
# 8. Add Features RSI, Return, and Volatility to RKLB DataFrame
# ---------------------------------------------------

rklb["RSI"] = rsi["RKLB"]                                   # Add the RSI values for RKLB to the RKLB DataFrame
rklb["Return"] = (rklb["RKLB"].pct_change())                # Add the daily returns for RKLB to the RKLB DataFrame by calculating the percentage change in closing prices
rklb["Volatility"] = (rklb["Return"].rolling(30).std())     # Add the rolling volatility for RKLB to the RKLB DataFrame by calculating the standard deviation of returns over a 30-day window
rklb["SPY_Return"] = (stocks["Close"]["SPY"].pct_change())  # Add the daily returns for SPY to the RKLB DataFrame by calculating the percentage change in closing prices for SPY to provide market context

# ---------------------------------------------------
# 8.1 Add Addition Features to RKLB DataFrame
# ---------------------------------------------------

rklb["Volume_MA30"] = (stocks["Volume"]["RKLB"].rolling(30).mean())     # Calculate the 30-day moving average of the trading volume for RKLB and add it as a new column "Volume_MA30" to the RKLB DataFrame
rklb["Volume_Ratio"] = (stocks["Volume"]["RKLB"]/rklb["Volume_MA30"])   # Calculate the volume ratio by dividing the current day's trading volume for RKLB by the 30-day moving average of the trading volume 
rklb["EMA30"] = (rklb["RKLB"].ewm(span=30).mean())                      # Calculate the 30-day Exponential Moving Average (EMA) for RKLB and add it as a new column "EMA30" to the RKLB DataFrame
rklb["MA30"] = (rklb["RKLB"].rolling(30).mean())                        # Calculate the 30-day Simple Moving Average (MA) for RKLB and add it as a new column "MA30" to the RKLB DataFrame
rklb["Price_vs_EMA"] = (rklb["RKLB"]/rklb["EMA30"])                     # Calculate the price vs EMA ratio by dividing the current day's closing price for RKLB by the 30-day Exponential Moving Average
rklb["Price_vs_MA"] = (rklb["RKLB"]/rklb["MA30"])                       # Calculate the price vs MA ratio by dividing the current day's closing price for RKLB by the 30-day Simple Moving Average
rklb["Market_Outperformance"] = (rklb["Return"]-rklb["SPY_Return"])     # Calculate the market outperformance by subtracting the SPY returns from the RKLB returns
rklb["Volume_Change"] = rklb["Volume"].pct_change()                     # Calculate the percentage change in trading volume for RKLB and add it as a new column "Volume_Change" to the RKLB DataFrame
rklb["Volume_5d_Avg"] = (rklb["Volume"].rolling(5).mean())              # Calculate the 5-day moving average of the trading volume for RKLB and add it as a new column "Volume_5d_Avg" to the RKLB DataFrame
rklb["Volume_10d_Avg"] = (rklb["Volume"].rolling(10).mean())            # Calculate the 10-day moving average of the trading volume for RKLB and add it as a new column "Volume_10d_Avg" to the RKLB DataFrame
rklb["Return_5d"] = rklb["RKLB"].pct_change(5)                          # Calculate the 5-day return for RKLB by calculating the percentage change in closing prices over a 5-day window and add it as a new column "Return_5d" to the RKLB DataFrame
rklb["Return_10d"] = rklb["RKLB"].pct_change(10)                        # Calculate the 10-day return for RKLB by calculating the percentage change in closing prices over a 10-day window and add it as a new column "Return_10d" to the RKLB DataFrame
rklb["Return_20d"] = rklb["RKLB"].pct_change(20)                        # Calculate the 20-day return for RKLB by calculating the percentage change in closing prices over a 20-day window and add it as a new column "Return_20d" to the RKLB DataFrame

# ---------------------------------------------------
# 9. Remove Missing Values
# ---------------------------------------------------

rklb = rklb.dropna()    # Remove any rows with missing values from the RKLB DataFrame to ensure that the dataset is clean and ready for modeling

# ---------------------------------------------------
# 9.1 Add News Sentiment to RKLB DataFrame
# ---------------------------------------------------

rklb = rklb.join(sentiment_df)                                                  # Join the sentiment_df DataFrame with the RKLB DataFrame  
rklb["Sentiment_Value"] = (rklb["Sentiment_Value"].fillna(0))                   # Fill any remaining missing values in the shifted sentiment column
rklb["Sentiment_Value_Shifted"] = (rklb["Sentiment_Value"].shift(1).fillna(0))  # Fill missing values with 0 and shift news dates by 1 day
rklb = rklb.dropna() # Remove any rows with missing values that may have been introduced by the shifting process to ensure that the dataset is clean and ready for modeling

# ---------------------------------------------------
# 10. Select Features and Target
# ---------------------------------------------------

FEATURES = [
    "RSI",
    "MA30",
    "Volatility",
    "Price_vs_EMA",
    "Sentiment_Value_Shifted",
    "Volume_10d_Avg",
]

X = rklb[FEATURES]  
y = rklb["Target"]

# ---------------------------------------------------
# 11. Train/Test Split
# ---------------------------------------------------

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2,random_state=42)
print("Train Shape:", X_train.shape)
print("Test Shape:", X_test.shape)

# ---------------------------------------------------
# 12. Train Logistic Regression
# ---------------------------------------------------

model = RandomForestClassifier(
    n_estimators=200,
    max_depth=5,
    random_state=42
)  # Initialize a Random Forest Classifier with 200 trees and a fixed random state for reproducibility
#model = make_pipeline(StandardScaler(),LogisticRegression(max_iter=1000))   # Scale features to avoid iteration issues with Logistic Regression
model.fit(X_train, y_train)

# ---------------------------------------------------
# 13. Make Predictions
# ---------------------------------------------------

probabilities_full = model.predict_proba(X_test)            # Get predicted probabilities for the test set
predictions = (probabilities_full[:,1] > 0.5).astype(int)   # Tuning the threshold to 0.5 to potentially improve precision at the cost of recall
print("\nFirst Predictions:")
print(predictions[:20])

# ---------------------------------------------------
# 14. Evaluate Model
# ---------------------------------------------------

print("\nTarget Distribution:")
print(y.value_counts())
print("\nAccuracy:")
print(accuracy_score(y_test,predictions))

print("\nPredicted Classes:")
print(np.unique(predictions))

print("\nClassification Report:")
print(classification_report(y_test,predictions))

# ---------------------------------------------------
# 15. Feature Importance
# ---------------------------------------------------

#feature_importance = model.named_steps["logisticregression"].coef_[0]
for feature, importance in zip(
    FEATURES,
    model.feature_importances_
):
    print(
        feature,
        round(importance,4)
    )
#for feature, importance in zip(FEATURES, feature_importance):
#    print(feature, round(importance, 4))

# ---------------------------------------------------
# 16. Cross-Validation Scores
# ---------------------------------------------------

scores = cross_val_score(make_pipeline(StandardScaler(),LogisticRegression(max_iter=1000)),X,y,cv=5,scoring="f1_macro")
print(scores)
print("Average:",scores.mean())

# ---------------------------------------------------
# 17. ROC-AUC Score
# ---------------------------------------------------

positive_probabilities = probabilities_full[:, 1] # [:,1] to get the probabilities for ALL the positive class (class 1) because ROC-AUC tests ability to detect class 1 / Spalte 1
auc = roc_auc_score(y_test,positive_probabilities)
print("ROC-AUC:", auc)

# ---------------------------------------------------
# 18. Visualization
# ---------------------------------------------------
"""
cm = confusion_matrix(y_test, predictions)          # Visualizing the Confusion Matrix
disp = ConfusionMatrixDisplay(confusion_matrix=cm)  

disp.plot()
plt.title("Confusion Matrix")
plt.show()

fpr, tpr, thresholds = roc_curve(y_test,positive_probabilities) # Visualizing the ROC Curve

plt.figure(figsize=(6,6))
plt.plot(fpr, tpr, label=f"AUC = {auc:.3f}")
plt.plot([0,1], [0,1], "--")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve")
plt.legend()
plt.show()
"""
print("\n--- FIN ---")

'''
# ---------------------------------------------------
# Vader Sentiment Analysis Test
# ---------------------------------------------------
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
analyzer = SentimentIntensityAnalyzer()

print("\n--- News Sentiment Test ---")

for item in news_data[:5]:
    headline = item["headline"]
    sentiment = analyzer.polarity_scores(headline)["compound"]

    print(headline)
    print("Sentiment:", sentiment)
    print()
Verdict: Vader was unable to understand financial vocabulary and context, resulting in mostly neutral sentiment scores. 
         This highlights the need for a more specialized financial sentiment analysis approach in future iterations of the project.
'''