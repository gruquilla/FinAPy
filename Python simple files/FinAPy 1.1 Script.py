#!/usr/bin/env python
# coding: utf-8

# In[12]:


import pandas as pd
import numpy as np
import datetime as dt
import yfinance as yfin
from pandas_datareader import data as pdr
import plotly.express as px
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from pygooglenews import GoogleNews
from datetime import datetime, timedelta
import time
import sys
from pyboxen import boxen
import tabulate
import colorama
from termcolor import colored
from ollama import chat, ChatResponse
from rich.console import Console
from threading import Thread
import warnings
import uuid
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Input, Dropout
from IPython.display import display, Markdown, update_display
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
import scipy.stats as stats
import torch
warnings.filterwarnings("ignore", category=DeprecationWarning) #A deprecation warning shows up every time concerning date parsing format, however the usage of the code excludes the situation in which this would be problematic.
warnings.filterwarnings("ignore", category=FutureWarning) #Error related to working with different timezones. You might not encounter this error if you don't change timezones yourself.

## Prologue: Ticker input collection and essential functions and data ##------------------------------------------------------------------
print(boxen("FinAPy: Financial Analysis with Python", color="cyan"))
ticker=input("Enter the ticker of the stock: ") #I mainly use Input instead of widgets in Jupyter Notebook so the script can later be easily edited outside Jupyter.
stock= yfin.Ticker(ticker)
AI_analysis_request=input('Would you like to have local AI synthetic analysis? This feature requires Ollama and demands both time and processing power. Type y for yes, n for no')
if AI_analysis_request=='y':
    AI_analysis=True
else: AI_analysis=False

try:
    stock_info=yfin.Ticker(ticker).info
    #Let's get basic information about the stock to provide a quick summary.
    stock_name=stock_info.get('shortName')
    stock_industry=stock_info.get('industry')
    stock_sector=stock_info.get('sector')
    stock_summary=stock_info.get('longBusinessSummary')
    stock_employees=stock_info.get('fullTimeEmployees')
    stock_city=stock_info.get('city')
    stock_country=stock_info.get('country')
    print(boxen(f'I identified the company as {stock_name}. It is a company of the {stock_industry} industry, part of the {stock_sector} sector, based in {stock_city}, {stock_country}. It employs {stock_employees} full time employees. I am now fetching the news, it can take a few minutes...', color="green"))
    display(Markdown(f"""In the meantime, for your information, {stock_summary}"""))
except: 
    print(boxen(f'Sorry, no results for {ticker}. Please make sure the ticker is correct and refers to a company. The analysis can"t proceed.', color="red"))
    sys.exit(1) #Radical, but we need a new valid input to go further on.

def ollama_chat(prompt, model_used='gpt-oss:20b'):
    #This function will be used later if the user enables the AI analysis.
    #You need Ollama installed on your computer to run it, as well as GPT-OSS:20B. 
    #You can also change the name of the model in this function if you prefer downloading and using another one. Deepseek R1:8B provides good results but sometimes requires more processing power and time to execute.
    if AI_analysis==True:
        # Generate a unique display ID for each call
        display_id = str(uuid.uuid4())
        display(Markdown("🌀 **Thinking using Ollama... This can take some time.**"), display_id=display_id)
        def task():
            try:
                response = chat(model=model_used, messages=[{'role': 'user', 'content': prompt},])
                resp = response['message']['content'].replace('<br>', '\n')
            except:
                resp = 'Apologies, the Ollama AI assistant could not provide additional analysis. Make sure Ollama is installed on your computer.'
            update_display(Markdown(f"✅ **Ollama analysis (using {model_used}):**\n\n{resp}"), display_id=display_id)
        Thread(target=task).start()
    else: print('AI Analysis has been disabled at the beginning of the analysis.')

stock_df = pd.DataFrame(stock.history(period="1y", interval="1d"))
#Computing basic metrics and filling the N/A values.
stock_df['Daily Return'] = stock_df['Close'].pct_change(1) * 100
stock_df.fillna({"Daily Return": 0}, inplace=True)
stock_df['monthly volatility'] = stock_df['Daily Return'].rolling(window=21).std()
stock_df.fillna({"monthly volatility": 0}, inplace=True)
stock_df['weekly volatility'] = stock_df['Daily Return'].rolling(window=5).std()
stock_df.fillna({"weekly volatility": 0}, inplace=True)
stock_df.describe().round(2)
stock_df = stock_df.reset_index()
stock_df = stock_df.rename(columns=lambda x: x.lower())
stock_df['date'] = pd.to_datetime(stock_df['date']).dt.tz_localize(None)


## Step 1: Events and news fetching ##------------------------------------------------------------------
try:
    stock_calendar=pd.DataFrame(yfin.Ticker(ticker).calendar)
    stock_calendar=stock_calendar.drop(['Earnings High', 'Earnings Low', 'Earnings Average','Revenue High','Revenue Low','Revenue Average'], axis=1)
    stock_calendar=stock_calendar.transpose()
    stock_calendar=stock_calendar.reset_index()
    stock_calendar=stock_calendar.rename(columns={'index':'Event',0:'Date'})
    stock_calendar=stock_calendar[stock_calendar.columns[::-1]]
    event_data=True # Using a boolean to know wether or not we have been able to fetch event data so the graph doesn't crash the code in case an error has been encountered.
except: 
    print(boxen("Unfortunately, I have not been able to gather information about the events of the company. The analysis will proceed, but I won't be able to provide any data about earnings date, dividends date and so on.", color="red"))
    event_data=False

def get_ticker_news_with_sentiment(ticker_name, ticker, max_results=100):
    # Load FinBERT model and tokenizer
    model_name = "ProsusAI/finbert"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    finbert = pipeline("sentiment-analysis", model=model, tokenizer=tokenizer, device=0 if torch.cuda.is_available() else -1)

    # Initialize GoogleNews
    gn = GoogleNews(lang='en', country='US')

    # Date range: last 12 months
    end_date = datetime.now()
    start_date = end_date - timedelta(days=365)
    start_date_str = start_date.strftime('%Y-%m-%d')
    end_date_str = end_date.strftime('%Y-%m-%d')

    # Search query
    search_term = f"{ticker_name} stock OR {ticker} stock"
    search = gn.search(search_term, from_=start_date_str, to_=end_date_str)

    articles = []
    count = 0
    for entry in search['entries']:
        if count >= max_results:
            break
        try:
            date = pd.to_datetime(entry.published)
            if date.tzinfo is not None:
                date = date.tz_localize(None)

            title = entry.title
            summary = entry.get('summary', '').strip()
            text = summary if summary else title # Trying the sentiment analysis on the summary rather than on simply the title for more accurate results. However, the summary is not always available.

            # Run FinBERT sentiment analysis
            result = finbert(text)[0]
            label = result['label'].lower()
            raw_score = result['score']

            # Convert label to signed score
            if label == 'positive':
                sentiment = round(raw_score, 4)
            elif label == 'negative':
                sentiment = round(-raw_score, 4)
            else:  # neutral
                sentiment = 0.0

            articles.append({
                'date': date,
                'headline': title,
                'summary': summary,
                'sentiment': sentiment
            })
            count += 1
            time.sleep(0.1)
        except Exception as e:
            print(f"Error processing an article: {e}")
            continue

    df = pd.DataFrame(articles)
    if not df.empty:
        df['date'] = pd.to_datetime(df['date'])
        df = df.sort_values('date', ascending=False)

    return df
    
news_df = get_ticker_news_with_sentiment(stock_name, ticker)
print(f"\nFound {len(news_df)} news articles for {stock_name}. I will now proceed to the analysis, this can take a few minutes.") #Essentially a log to track the news fetching and indicate how rich the sualitative data treated are.
if len(news_df)==0:
    print(boxen("Unfortunately, I have not been able to find any news about your company. The analysis will proceed, but you won't get any news-related data.", color="red"))
    news_data=False
else:
    news_data=True

## Step 2: Forecast using Machine Learning LSTM ##--------------------------------------------------------------------------------------------
# --- Prepare Data with Multiple Features ---
# Use close, high, and low prices for optimal accuracy.
feature_columns = ['close', 'high', 'low']
price_data = stock_df[feature_columns].values
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(price_data)

# --- Create Sequences with Multiple Features ---
def create_multivariate_sequences(data, seq_length):
    X, y = [], []
    for i in range(seq_length, len(data)):
        # Each X sequence now contains all features
        X.append(data[i-seq_length:i])
        # Target remains the closing price only
        y.append(data[i, 0])  # 0 index is for close price
    return np.array(X), np.array(y)

seq_length = 90
X, y = create_multivariate_sequences(scaled_data, seq_length)

# --- Train/Test Split ---
split = int(len(X) * 0.8)
X_train, X_test = X[:split], X[split:]
y_train, y_test = y[:split], y[split:]

# --- Build Enhanced LSTM Model ---
def build_enhanced_model(input_shape, dropout_rate=0.2):
    model = Sequential()
    model.add(Input(shape=input_shape))
    model.add(LSTM(units=100, return_sequences=True))
    model.add(Dropout(dropout_rate))
    model.add(LSTM(units=100))
    model.add(Dropout(dropout_rate))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='huber')
    return model

# Train enhanced model with multiple features
enhanced_model = build_enhanced_model(input_shape=(seq_length, len(feature_columns)))
enhanced_model.fit(X_train, y_train, epochs=30, batch_size=32, verbose=0)

# --- Modified Generate Forecast Function ---
def generate_multivariate_forecast(model, last_sequence, periods=60, noise_level=0.0, trend_factor=0.0):
    forecast = []
    current_sequence = last_sequence.copy()
    
    for i in range(periods):
        input_seq = current_sequence.reshape(1, seq_length, len(feature_columns))
        pred = model.predict(input_seq, verbose=0)[0][0]
        
        # Add noise and trend adjustments
        adjusted_pred = pred + np.random.normal(0, noise_level) + (i * trend_factor)
        adjusted_pred = max(0, min(1, adjusted_pred))
        
        forecast.append(adjusted_pred)
        
        # Create a new row with predicted close and estimated high/low
        # For simplicity, we'll use the same value for all features
        # In a more sophisticated approach, you could predict high/low separately
        new_row = np.array([[adjusted_pred] * len(feature_columns)])
        
        # Update the sequence by removing the first row and adding the new prediction
        current_sequence = np.append(current_sequence[1:], new_row, axis=0)
    return np.array(forecast)

# Get the last sequence from our multivariate data
last_sequence = scaled_data[-seq_length:]

# Generate different scenarios with the enhanced model
forecast_periods = 60

# Generate forecasts with the enhanced model and variants. The variants are statistically generated at the moment, the integration of qualitative information is something I am working on at the moment.
baseline_forecast = generate_multivariate_forecast(enhanced_model, last_sequence, forecast_periods)
optimistic_forecast = generate_multivariate_forecast(enhanced_model, last_sequence, forecast_periods, 
                                                   noise_level=0.005, trend_factor=0.001)
pessimistic_forecast = generate_multivariate_forecast(enhanced_model, last_sequence, forecast_periods, 
                                                    noise_level=0.005, trend_factor=-0.001)

# --- Inverse Transform & Create Forecast DataFrame ---
# We need to reshape and pad the forecasts to match the original feature dimensions
def inverse_transform_forecast(forecast, feature_index=0):
    # Create a placeholder array with the same number of features as the original data
    padded_forecast = np.zeros((len(forecast), len(feature_columns)))
    # Place the forecast values in the appropriate feature column (0 for close price)
    padded_forecast[:, feature_index] = forecast
    # Inverse transform
    return scaler.inverse_transform(padded_forecast)[:, feature_index]

baseline_prices = inverse_transform_forecast(baseline_forecast)
optimistic_prices = inverse_transform_forecast(optimistic_forecast)
pessimistic_prices = inverse_transform_forecast(pessimistic_forecast)

# Create future dates
future_dates = pd.date_range(start=stock_df['date'].iloc[-1] + pd.Timedelta(days=1), periods=forecast_periods)

# Create forecast dataframe with all scenarios
forecast_df = pd.DataFrame({
    'date': future_dates,
    'Baseline': baseline_prices,
    'Optimistic': optimistic_prices,
    'Pessimistic': pessimistic_prices,
})
## Step 3: Market data restitution ##----------------------------------------------------------------------------------------------------------
print(boxen("I. Market data analysis", color="cyan"))
fig = make_subplots(rows=4, cols=1, shared_xaxes=True,
                    vertical_spacing=0.05, subplot_titles=(f'{ticker} ({stock_name}) Stock Price', ' ', ' '), 
                    row_heights=[0.7, 0.3, 0.3, 0.1]) #Using a single shart so using customised windows will update all the visuals at once. It should improve the user experience.

# Add candlestick chart
fig.add_trace(go.Candlestick(
    x=stock_df['date'],
    open=stock_df['open'],
    high=stock_df['high'],
    low=stock_df['low'],
    close=stock_df['close'],
    name='Price',
    legendgroup='Base information'
), row=1, col=1)


if news_data == True:
    sentiment_values = news_df['sentiment'].clip(-1, 1)  # Ensure values are in [-1, 1]
    abs_sentiment = sentiment_values.abs()
    # Scale marker size: map abs sentiment [0,1] to size range [6, 20].
    marker_sizes = abs_sentiment * 14 + 6
    colorscale = [
        [0.0, 'crimson'],     # Highly negative
        [0.5, 'white'],       # Neutral
        [1.0, 'springGreen']  # Highly positive
    ]
    fig.add_trace(go.Scatter(
        x=news_df['date'],
        y=stock_df['close'],
        name='News Headlines',
        legendgroup='Events',
        mode='markers',
        text=news_df['headline'],
        hoverinfo='text+x',
        marker=dict(
            color=sentiment_values,         # Color by raw sentiment
            size=marker_sizes,              # Size by absolute sentiment
            colorscale=colorscale,
            cmin=-1,
            cmax=1,
            colorbar=dict(title='Sentiment'),
            symbol='circle',
            line=dict(width=1, color='dodgerblue')
        )
    ), row=1, col=1)

if event_data==True:
    fig.add_trace(go.Scatter(
        x=stock_calendar['Date'],
        y=stock_df['close'],
        name='Events',  # Use a simple string name for the series
        legendgroup='Events',
        mode='markers',  # Add mode='markers' to show points
        text=stock_calendar['Event'],  # Use the headlines as hover text
        hoverinfo='text+x',  # Show the text and x-value on hover
        marker=dict(
            color='rgba(0, 200, 200, 1)',  # Set marker color to blue
            size=8,  # Optional: adjust the size of the markers
            symbol='diamond'
            
        )
    ), row=1, col=1)

fig.add_trace(go.Scatter(
    x=stock_df['date'],
    y=stock_df['close'],
    name='Closing price',
    legendgroup='Base information',
    line=dict(color='rgba(0, 0, 0, 0.25)', width=1)
), row=1, col=1)

# Add moving averages to give more data and give perspective on the LSTM forecast.
ma10 = stock_df['close'].rolling(window=10).mean()
ma20 = stock_df['close'].rolling(window=20).mean()
ma50 = stock_df['close'].rolling(window=50).mean()

fig.add_trace(go.Scatter(
    x=stock_df['date'],
    y=ma10,
    name='10-day MA',
    legendgroup='SMA',
    line=dict(color='rgba(80, 80, 255, 1)', width=1)
), row=1, col=1)

fig.add_trace(go.Scatter(
    x=stock_df['date'],
    y=ma20,
    name='20-day MA',
    legendgroup='SMA',
    line=dict(color='rgba(130, 80, 200, 1)', width=1)
), row=1, col=1)

fig.add_trace(go.Scatter(
    x=stock_df['date'],
    y=ma50,
    name='50-day MA',
    legendgroup='SMA',
    line=dict(color='rgba(255, 80, 255, 1)', width=1)
), row=1, col=1)

#Add to Plot forecast
fig.add_trace(go.Scatter(
    x=forecast_df['date'],
    y=forecast_df['Baseline'],
    name='Baseline scenario',
    legendgroup='LSTM forecast',
    line=dict(color='cyan', width=1.5, dash='dash')
), row=1, col=1)

fig.add_trace(go.Scatter(
    x=forecast_df['date'],
    y=forecast_df['Optimistic'],
    name='Optimistic scenario',
    legendgroup='LSTM forecast',
    line=dict(color='rgba(85, 255, 210, 1)', width=1, dash='dash')
), row=1, col=1)

fig.add_trace(go.Scatter(
    x=forecast_df['date'],
    y=forecast_df['Pessimistic'],
    name='Pessimistic scenario',
    legendgroup='LSTM forecast',
    line=dict(color='rgba(147, 146, 255, 1)', width=1, dash='dash')
), row=1, col=1)

fig.add_trace(go.Scatter(
    x=stock_df['date'],
    y=stock_df['daily return'],
    name='% dly rtn',
    legendgroup='Daily returns',
    line=dict(color='rgba(117, 183, 98, 1)', width=1)
), row=2, col=1)

fig.add_trace(go.Scatter(
    x=stock_df['date'],
    y=stock_df['weekly volatility'],
    name='weekly volatility',
    legendgroup='Volatility',
    line=dict(color='rgba(249, 180, 78, 1)', width=1)
), row=3, col=1)

fig.add_trace(go.Scatter(
    x=stock_df['date'],
    y=stock_df['monthly volatility'],
    name='monthly volatility',
    legendgroup='Volatility',
    line=dict(color='rgba(255, 121, 100, 1)', width=1)
), row=3, col=1)

# Add volume bar chart
fig.add_trace(go.Bar(
    x=stock_df['date'],
    y=stock_df['volume'],
    name='Volume',
    legendgroup='Volume',
    marker_color='rgba(0, 150, 255, 0.6)'
), row=4, col=1)

# Update layout
fig.update_layout(
    title=f'{ticker}',
    xaxis_title=' ',
    yaxis_title='Price ($)',
    xaxis_rangeslider_visible=False,
    plot_bgcolor='white',
    height=1000,
    width=1000,
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
)

# Update y-axes
fig.update_yaxes(title_text="Price ($)", row=1, col=1)
fig.update_yaxes(title_text="Daily returns", row=2, col=1)
fig.update_yaxes(title_text="Volatility", row=3, col=1)
fig.update_yaxes(title_text="Volume", row=4, col=1)
fig.update_yaxes(showgrid=True,gridwidth=0.5,gridcolor='lightgrey')
fig.update_xaxes(showgrid=True,gridwidth=0.5,gridcolor='lightgrey', nticks=24)
# Show the figure
fig.show()

figu = px.histogram(stock_df, x='daily return',marginal="box", title='Daily returns histogram')
figu.update_traces(marker_color='rgba(117, 183, 98, 1)')
figu.show()

#Treating statistical data often studied on the CFA L1 for context, with explanation. This way, candidates can visualise real-life data of stocks to give a practical context to the concept.

sk=stock_df['daily return'].skew()
if sk>0:
    sk_type="skewed to the right: there is a long tail of large positive outliers."
elif sk<0:
    sk_type="skewed to the left: there is a long tail of large negative outliers."
else: sk_type="returns are not skewed. Returns are usually stable and predictable."

kurt=stock_df['daily return'].kurt()
excess_kurt=kurt-3
if kurt>3 :
    kr_type="leptokurtic distribution: the tails are heavy, there is a high tail risk and hence usually more erratic returns."
elif kurt==3 :
    kr_type="mesokurtic distribution: the distribution is relatively normal."
else: kr_type="platykurtic distribution: the tails are light, there is a lower tail risk and hence usually more stable returns."
import plotly.graph_objects as go

fig = go.Figure()

# Skewness Indicator
fig.add_trace(go.Indicator(
    mode="gauge+number",
    value=sk,
    delta={'reference': 0, 'relative': False},
    title={"text": "Skewness"},
    number={"valueformat": ".2f"},
    domain={'row': 0, 'column': 0},
    gauge={
        'axis': {'range': [-5, 5]},
        'bar': {'color': "green"},
        'steps': [
            {'range': [-5, -2], 'color': "lightcoral"},
            {'range': [-2, 2], 'color': "lightgray"},
            {'range': [2, 5], 'color': "lightgreen"}
        ],
        'threshold': {
            'line': {'color': "blue", 'width': 4},
            'thickness': 0.75,
            'value': sk
        }
    }
))

# Kurtosis Indicator
fig.add_trace(go.Indicator(
    mode="gauge+number",
    value=kurt,
    delta={'reference': 3, 'relative': False},
    title={"text": "Kurtosis"},
    number={"valueformat": ".2f"},
    domain={'row': 0, 'column': 1},
    gauge={
        'axis': {'range': [0, 6]},
        'bar': {'color': "darkblue"},
        'steps': [
            {'range': [0, 2], 'color': "lightgreen"},
            {'range': [2, 4], 'color': "lightgray"},
            {'range': [4, 6], 'color': "lightcoral"}
        ],
        'threshold': {
            'line': {'color': "blue", 'width': 4},
            'thickness': 0.75,
            'value': kurt
        }
    }
))

# Layout and annotations
fig.update_layout(
    grid={'rows': 1, 'columns': 2, 'pattern': "independent"},
    template="plotly_white",
)

fig.show()

print(f'Skewness: {round(sk,4)}, {sk_type}')
print(f'Kurtosis: {round(kurt,4)}, Excess kurtosis: {round(excess_kurt,4)}. This is a {kr_type}')
print(' ')

print('AI financial analysis:')
ollama_chat(f'You are a senior financial analyst, with more than 20 years of experience and a proven track of success and expertise. Your team, clients and colleagues appreciate not only your expertise, but also your ability to put things in perspective, your pedagogy, and your reserve about things you are not sure about. One of your client is considering a company for his investments. You will have to do your best to advise him. He is not a finance expert, but he is capable of understanding intermediate to high-end financial concepts once explained. At your disposal, you have informations about the stock {ticker}, such as the industry of the company which is {stock_industry}, as well as {news_df} the latest news of the company, as well as {stock_df} the historical valus of the stock, as well as {stock_calendar} the main events related to the stock, as well as {kurt} the kurtosis of the returns and {sk} their skewness. You will output a very brief summary of your analysis of the company (8 sentences maximum). This summary shall include: 1: A brief interpretation of the stock price evolution over the last year. 2: an immediate appreciation of the qualities and defaults of the stock right now, taking into consideration the industry in the context of its sector and of the current PESTEL landscape. 3: an investment recommendation, declined depending on the investment objectives of the client. 4: a simple 60 days horizon qualitative prevision of how the stock price will evolve.')

## Step 4: Financial statement analysis ##-----------------------------------------------------------------------------------------------
print(boxen("II. Financial statements analysis", color="cyan"))
print(' ')
print(f'Industry of your company: {stock_industry}')
print(' ')

def Ollama_Analysis_Quant(ratiosDf, metric):
    time.sleep(1) # Sometimes, not adding a delay doesn't allow Ollama to collect the accurate information about the company, especially during the first iterations.
    print(' ')
    print(f'AI financial analysis on {metric} ratios:')
    ollama_chat(f'You are a senior financial analyst, with more than 20 years of experience and a proven track of success and expertise. Your team, clients and colleagues appreciate not only your expertise, but also your ability to put things in perspective, your pedagogy, and your reserve about things you are not sure about. One of your young colleagues is considering a company for his investments. You will have to do your best to advise him. He doesn’t have much experience, but he is capable of understanding intermediate to high-end financial concepts once explained. You have to deliver to that young colleague your sentiment about the {metric} of the company {stock_name}. You will begin by analysing the ratios of {ratiosDf}, and then analyse the financial statements {income_statement}, {balance_sheet} and {cashflow} by highlighting the information about {metric} the ratios do not highlight on their own or even conceal. Enrich your analysis with elements in context with the current situation of the company in its sector, and overall the current PESTEL environment that might affect the company and its sector. You will output a 8 sentences maximum summary that covers all these elements.')
    print(' ')

def scale_dataframe(df, numeric_columns=None):
    import re
    # Create a copy to avoid modifying the original
    scaled_df = df.copy()
    # Format date column names
    new_column_names = {}
    for col in scaled_df.columns:
        # Check if column name matches date format YYYY-MM-DD XX:XX:XX
        date_match = re.match(r'(\d{4}-\d{2}-\d{2})\s\d{2}:\d{2}:\d{2}', str(col))
        if date_match:
            # Extract just the YYYY-MM-DD part
            new_column_names[col] = date_match.group(1)
    # Rename columns with date format
    scaled_df = scaled_df.rename(columns=new_column_names)
    # Identify numeric columns if not specified
    if numeric_columns is None:
        numeric_columns = scaled_df.select_dtypes(include=[np.number]).columns.tolist()
    # Find the maximum absolute value to determine scale
    max_value = np.nanmax(np.abs(scaled_df[numeric_columns].values))
    # Determine appropriate scale
    if max_value >= 2_000_000_000:
        scale = 1_000_000_000
        unit = 'Billions'  # Billions
    elif max_value >= 2_000_000:
        scale = 1_000_000
        unit = 'Millions'  # Millions
    else:
        scale = 1_000
        unit = 'K Thousands'  # Thousands
    # Apply scaling to all numeric columns
    for col in numeric_columns:
        scaled_df[col] = scaled_df[col] / scale    
    return scaled_df, unit

try:
    #Fetching income statement
    income_statement=yfin.Ticker(ticker).income_stmt
    income_statement, unit=scale_dataframe(income_statement)
    income_statement=income_statement.iloc[:, :3]

    #Fetching balance sheet
    balance_sheet=yfin.Ticker(ticker).balance_sheet
    balance_sheet, unit=scale_dataframe(balance_sheet)
    balance_sheet=balance_sheet.iloc[:, :3]
    
    #fetching cashflow statement
    cashflow=yfin.Ticker(ticker).cashflow
    cashflow, unit=scale_dataframe(cashflow)
    cashflow=cashflow.iloc[:, :3]
    
    marketcap=yfin.Ticker(ticker).info.get('marketCap')
    forward_pe=yfin.Ticker(ticker).info.get('forwardPE')
    div_yield=yfin.Ticker(ticker).info.get('dividendYield')
    trailing_eps=yfin.Ticker(ticker).info.get('trailingEps')
except: 
    print(boxen(f'Sorry, I could not fetch the financial statement data for your company.', color="red"))
    sys.exit(1)

print(colored('   Profitability ratios:', 'blue', 'on_white', attrs=['bold']))
#value = df.loc[name_of_row].iloc[column_number]

def variation_over_years(df, NbRow, Reverse=False):
    # Convert percentage strings to floats before calculations
    try:
        # Get the values for all three years
        Ratio = df.iloc[(NbRow-1), 0]
        val_year1 = df.iloc[(NbRow-1), 1]  # First year (most recent)
        val_year2 = df.iloc[(NbRow-1), 2]  # Second year
        val_year3 = df.iloc[(NbRow-1), 3]  # Third year (oldest)
        
        # Convert percentage strings to floats by removing '%' and dividing by 100
        if isinstance(val_year1, str) and '%' in val_year1:
            val_year1 = float(val_year1.strip('%')) / 100
        else:
            val_year1 = float(val_year1)
            
        if isinstance(val_year2, str) and '%' in val_year2:
            val_year2 = float(val_year2.strip('%')) / 100
        else:
            val_year2 = float(val_year2)
            
        if isinstance(val_year3, str) and '%' in val_year3:
            val_year3 = float(val_year3.strip('%')) / 100
        else:
            val_year3 = float(val_year3)
        
        # Calculate variations - handle negative values properly
        # For negative values, we need to be careful with the calculation
        if val_year2 != 0:
            var1 = (val_year1 - val_year2) / abs(val_year2)
        else:
            var1 = 0
            
        if val_year3 != 0:
            var2 = (val_year2 - val_year3) / abs(val_year3)
        else:
            var2 = 0
        
        # Determine colors based on Reverse parameter
        pos_color = 'red' if Reverse else 'green'
        neg_color = 'green' if Reverse else 'red'
        
        # Format variations as colored percentages with "+" for positive values
        var1_formatted = colored("+{:.2%}".format(var1) if var1 >= 0 else "{:.2%}".format(var1), 
                                pos_color if var1 >= 0 else neg_color)
        var2_formatted = colored("+{:.2%}".format(var2) if var2 >= 0 else "{:.2%}".format(var2), 
                                pos_color if var2 >= 0 else neg_color)
        
        # Create the new row with colored variations
        df.loc[NbRow] = [colored(f'{Ratio} variation','blue'), 
                         var1_formatted,
                         var2_formatted,
                         "----", 
                         colored(f'Variation of previous ratio','blue')]
    except Exception as e:
        print(f"Error in variation_over_years: {e}")
        # Add a placeholder row instead of failing
        df.loc[NbRow] = [colored(f'{Ratio} variation', 'blue'), 
                         "Error",
                         "Error",
                         "----", 
                         colored(f'Variation of previous ratio', 'blue')]
    
    return df.loc[NbRow]
#Profitability ratios:
try :
    prof_ratios_df = pd.DataFrame(columns=['Metric'] + income_statement.columns[-3:].tolist() + ['Description'])
    prof_ratios_df.loc[0]=['Gross profit margin', 
                       "{:.2%}".format(income_statement.loc['Gross Profit'].iloc[0]/income_statement.loc['Total Revenue'].iloc[0]),
                       "{:.2%}".format(income_statement.loc['Gross Profit'].iloc[1]/income_statement.loc['Total Revenue'].iloc[1]),
                       "{:.2%}".format(income_statement.loc['Gross Profit'].iloc[2]/income_statement.loc['Total Revenue'].iloc[2]),
                       'Gross profit/ revenue']

    variation_over_years(prof_ratios_df,1)

    prof_ratios_df.loc[2]=['Operating profit margin', 
                       "{:.2%}".format(income_statement.loc['Operating Income'].iloc[0]/income_statement.loc['Total Revenue'].iloc[0]),
                       "{:.2%}".format(income_statement.loc['Operating Income'].iloc[1]/income_statement.loc['Total Revenue'].iloc[1]),
                       "{:.2%}".format(income_statement.loc['Operating Income'].iloc[2]/income_statement.loc['Total Revenue'].iloc[2]),
                       'Operating income/ revenue']

    variation_over_years(prof_ratios_df,3)

    prof_ratios_df.loc[4]=['Net profit margin',
                       "{:.2%}".format(income_statement.loc['Net Income'].iloc[0]/income_statement.loc['Total Revenue'].iloc[0]),
                       "{:.2%}".format(income_statement.loc['Net Income'].iloc[1]/income_statement.loc['Total Revenue'].iloc[1]),
                       "{:.2%}".format(income_statement.loc['Net Income'].iloc[2]/income_statement.loc['Total Revenue'].iloc[2]),
                       'Net income/ revenue']

    variation_over_years(prof_ratios_df,5)

    prof_ratios_df.loc[6]=['--------', '--------', '--------', '--------', '------------------------------']

    prof_ratios_df.loc[7]=['ROA (Return On Assets)', 
                       "{:.2%}".format(income_statement.loc['Net Income'].iloc[0]/balance_sheet.loc['Total Assets'].iloc[0]),
                       "{:.2%}".format(income_statement.loc['Net Income'].iloc[1]/balance_sheet.loc['Total Assets'].iloc[1]),
                       "{:.2%}".format(income_statement.loc['Net Income'].iloc[2]/balance_sheet.loc['Total Assets'].iloc[2]),
                       'Net income/ Total assets']

    variation_over_years(prof_ratios_df,8)

    prof_ratios_df.loc[9]=['Operating return on assets', 
                       "{:.2%}".format(income_statement.loc['Operating Income'].iloc[0]/balance_sheet.loc['Total Assets'].iloc[0]),
                       "{:.2%}".format(income_statement.loc['Operating Income'].iloc[1]/balance_sheet.loc['Total Assets'].iloc[1]),
                       "{:.2%}".format(income_statement.loc['Operating Income'].iloc[2]/balance_sheet.loc['Total Assets'].iloc[2]),
                       'Operating income/ total assets']

    variation_over_years(prof_ratios_df,10)

    prof_ratios_df.loc[11]=['Simp.ROCE (Return On Capital Employed)', 
                       "{:.2%}".format(income_statement.loc['EBIT'].iloc[0]/(balance_sheet.loc['Total Debt'].iloc[0]+balance_sheet.loc['Total Equity Gross Minority Interest'].iloc[0])),
                       "{:.2%}".format(income_statement.loc['EBIT'].iloc[1]/(balance_sheet.loc['Total Debt'].iloc[1]+balance_sheet.loc['Total Equity Gross Minority Interest'].iloc[1])),
                       "{:.2%}".format(income_statement.loc['EBIT'].iloc[2]/(balance_sheet.loc['Total Debt'].iloc[2]+balance_sheet.loc['Total Equity Gross Minority Interest'].iloc[2])),
                       'EBIT/ (total debt + Total equity)']

    variation_over_years(prof_ratios_df,12)

    prof_ratios_df.loc[13]=['ROE (Return on Equity)', 
                       "{:.2%}".format(income_statement.loc['Net Income'].iloc[0]/balance_sheet.loc['Total Equity Gross Minority Interest'].iloc[0]),
                       "{:.2%}".format(income_statement.loc['Net Income'].iloc[1]/balance_sheet.loc['Total Equity Gross Minority Interest'].iloc[1]),
                       "{:.2%}".format(income_statement.loc['Net Income'].iloc[2]/balance_sheet.loc['Total Equity Gross Minority Interest'].iloc[2]),
                       'Net income/ total equity']

    variation_over_years(prof_ratios_df,14)

    print(prof_ratios_df.to_markdown())
    Ollama_Analysis_Quant(prof_ratios_df, 'profitability')
except: print('Computing the ratios of this section failed. This might be due to a change of the financial statement layout on Yahoo Finance which varies when fetched from certain providers.')
print(' ')

print(colored('   Asset related ratios:', 'blue', 'on_white', attrs=['bold']))
assets_ratios_df = pd.DataFrame(columns=['Metric'] + income_statement.columns[-3:].tolist() + ['Description'])

try:
    assets_ratios_df.loc[0]=['Fixed assets turnover', 
                       round(income_statement.loc['Total Revenue'].iloc[0]/balance_sheet.loc['Net PPE'].iloc[0],2),
                       round(income_statement.loc['Total Revenue'].iloc[1]/balance_sheet.loc['Net PPE'].iloc[1],2),
                       round(income_statement.loc['Total Revenue'].iloc[2]/balance_sheet.loc['Net PPE'].iloc[2],2),
                       'revenue/ net fixed assets']
    variation_over_years(assets_ratios_df,1)

    assets_ratios_df.loc[2]=['Total asset turnover', 
                       round(income_statement.loc['Total Revenue'].iloc[0]/balance_sheet.loc['Total Assets'].iloc[0],2),
                       round(income_statement.loc['Total Revenue'].iloc[1]/balance_sheet.loc['Total Assets'].iloc[1],2),
                       round(income_statement.loc['Total Revenue'].iloc[2]/balance_sheet.loc['Total Assets'].iloc[2],2),
                       'revenue/ total assets']
    variation_over_years(assets_ratios_df,3)

    assets_ratios_df.loc[4]=['Working capital turnover', 
                       "{:.2%}".format(income_statement.loc['Total Revenue'].iloc[0]/((balance_sheet.loc['Working Capital'].iloc[0]+balance_sheet.loc['Working Capital'].iloc[1])/2)),
                        "{:.2%}".format(income_statement.loc['Total Revenue'].iloc[1]/((balance_sheet.loc['Working Capital'].iloc[1]+balance_sheet.loc['Working Capital'].iloc[2])/2)),
                        '----',
                       'Total revenue/ Avg working capital using previous year data']

    print(assets_ratios_df.to_markdown())
    Ollama_Analysis_Quant(assets_ratios_df, 'asset ratios')
except: print('Computing the ratios of this section failed. This might be due to a change of the financial statement layout on Yahoo Finance which varies when fetched from certain providers.')
print(' ')

print(colored('   Liquidity ratios:', 'blue', 'on_white', attrs=['bold']))
#Liquidity ratios:
liq_ratios_df = pd.DataFrame(columns=['Metric'] + income_statement.columns[-3:].tolist() + ['Description'])

try:

    liq_ratios_df.loc[0]=['Full current ratio', 
                       round((balance_sheet.loc['Current Assets'].iloc[0])/balance_sheet.loc['Current Liabilities'].iloc[0],2),
                       round((balance_sheet.loc['Current Assets'].iloc[1])/balance_sheet.loc['Current Liabilities'].iloc[1],2),
                       round((balance_sheet.loc['Current Assets'].iloc[2])/balance_sheet.loc['Current Liabilities'].iloc[2],2),
                       'Uses total current assets, better for sector comparison']
    variation_over_years(liq_ratios_df,1)


    liq_ratios_df.loc[2]=['Conservative current ratio', 
                       round((balance_sheet.loc['Cash Cash Equivalents And Short Term Investments'].iloc[0]+balance_sheet.loc['Accounts Receivable'].iloc[0]+balance_sheet.loc['Inventory'].iloc[0])/balance_sheet.loc['Current Liabilities'].iloc[0],2),
                       round((balance_sheet.loc['Cash Cash Equivalents And Short Term Investments'].iloc[1]+balance_sheet.loc['Accounts Receivable'].iloc[1]+balance_sheet.loc['Inventory'].iloc[1])/balance_sheet.loc['Current Liabilities'].iloc[1],2),
                       round((balance_sheet.loc['Cash Cash Equivalents And Short Term Investments'].iloc[2]+balance_sheet.loc['Accounts Receivable'].iloc[2]+balance_sheet.loc['Inventory'].iloc[2])/balance_sheet.loc['Current Liabilities'].iloc[2],2),
                       'Takes in count cash/receivables/inventory only.']
    variation_over_years(liq_ratios_df,3)

    liq_ratios_df.loc[4]=['Quick ratio', 
                       round((balance_sheet.loc['Cash Cash Equivalents And Short Term Investments'].iloc[0]+balance_sheet.loc['Accounts Receivable'].iloc[0])/balance_sheet.loc['Current Liabilities'].iloc[0],2),
                       round((balance_sheet.loc['Cash Cash Equivalents And Short Term Investments'].iloc[1]+balance_sheet.loc['Accounts Receivable'].iloc[1])/balance_sheet.loc['Current Liabilities'].iloc[1],2),
                       round((balance_sheet.loc['Cash Cash Equivalents And Short Term Investments'].iloc[2]+balance_sheet.loc['Accounts Receivable'].iloc[2])/balance_sheet.loc['Current Liabilities'].iloc[2],2),
                       'Current ratio without inventory']
    variation_over_years(liq_ratios_df,5)

    liq_ratios_df.loc[6]=['Cash ratio', 
                       round((balance_sheet.loc['Cash Cash Equivalents And Short Term Investments'].iloc[0])/balance_sheet.loc['Current Liabilities'].iloc[0],2),
                       round((balance_sheet.loc['Cash Cash Equivalents And Short Term Investments'].iloc[1])/balance_sheet.loc['Current Liabilities'].iloc[1],2),
                       round((balance_sheet.loc['Cash Cash Equivalents And Short Term Investments'].iloc[2])/balance_sheet.loc['Current Liabilities'].iloc[2],2),
                       'Current ratio without inventory/ receivables']
    variation_over_years(liq_ratios_df,7)

    print(liq_ratios_df.to_markdown())
    Ollama_Analysis_Quant(liq_ratios_df, 'liquidity')
except: print('Computing the ratios of this section failed. This might be due to a change of the financial statement layout on Yahoo Finance which varies when fetched from certain providers.')
print(' ')

print(colored('   Solvency ratios:', 'blue', 'on_white', attrs=['bold']))
print(' ')
#Solvency ratios
print(colored('      Level of debt ratios:', 'blue', attrs=['bold']))
dbt_ratios_df = pd.DataFrame(columns=['Metric'] + income_statement.columns[-3:].tolist() + ['Description'])

try:
    dbt_ratios_df.loc[0]=['Debt to assets ratio', 
                       round((balance_sheet.loc['Total Debt'].iloc[0])/balance_sheet.loc['Total Liabilities Net Minority Interest'].iloc[0],2),
                       round((balance_sheet.loc['Total Debt'].iloc[1])/balance_sheet.loc['Total Liabilities Net Minority Interest'].iloc[1],2),
                       round((balance_sheet.loc['Total Debt'].iloc[2])/balance_sheet.loc['Total Liabilities Net Minority Interest'].iloc[2],2),
                       'Total debt/ total assets']
    variation_over_years(dbt_ratios_df,1, True)

    dbt_ratios_df.loc[2]=['Debt to capital ratio', 
                       round(balance_sheet.loc['Total Debt'].iloc[0]/(balance_sheet.loc['Total Debt'].iloc[0]+balance_sheet.loc['Stockholders Equity'].iloc[0]),2),
                       round(balance_sheet.loc['Total Debt'].iloc[1]/(balance_sheet.loc['Total Debt'].iloc[1]+balance_sheet.loc['Stockholders Equity'].iloc[1]),2),
                       round(balance_sheet.loc['Total Debt'].iloc[2]/(balance_sheet.loc['Total Debt'].iloc[2]+balance_sheet.loc['Stockholders Equity'].iloc[2]),2),
                       'Total debt/(total debt + total shareholders equity)']
    variation_over_years(dbt_ratios_df,3, True)

    dbt_ratios_df.loc[4]=['Debt to equity ratio', 
                       round((balance_sheet.loc['Total Debt'].iloc[0])/balance_sheet.loc['Stockholders Equity'].iloc[0],2),
                       round((balance_sheet.loc['Total Debt'].iloc[1])/balance_sheet.loc['Stockholders Equity'].iloc[1],2),
                       round((balance_sheet.loc['Total Debt'].iloc[2])/balance_sheet.loc['Stockholders Equity'].iloc[2],2),
                       'Total debt/ total shareholders equity']
    variation_over_years(dbt_ratios_df,5, True)

    dbt_ratios_df.loc[6]=['Financial leverage ratio', 
                       round((balance_sheet.loc['Total Assets'].iloc[0])/balance_sheet.loc['Stockholders Equity'].iloc[0],2),
                       round((balance_sheet.loc['Total Assets'].iloc[1])/balance_sheet.loc['Stockholders Equity'].iloc[1],2),
                       round((balance_sheet.loc['Total Assets'].iloc[2])/balance_sheet.loc['Stockholders Equity'].iloc[2],2),
                       'Total assets/ total equity. Higher=more debt used']
    variation_over_years(dbt_ratios_df,7, True)

    print(dbt_ratios_df.to_markdown())
    Ollama_Analysis_Quant(dbt_ratios_df, 'level of debt')
except: print('Computing the ratios of this section failed. This might be due to a change of the financial statement layout on Yahoo Finance which varies when fetched from certain providers.')
print(' ')
print(colored('      Coverage ratios:', 'blue', attrs=['bold']))

cov_ratios_df = pd.DataFrame(columns=['Metric'] + income_statement.columns[-3:].tolist() + ['Description'])

try:
    cov_ratios_df.loc[0]=['Interest coverage', 
                       round(income_statement.loc['EBIT'].iloc[0]/income_statement.loc['Interest Expense'].iloc[0],2),
                       round(income_statement.loc['EBIT'].iloc[1]/income_statement.loc['Interest Expense'].iloc[1],2),
                       round(income_statement.loc['EBIT'].iloc[2]/income_statement.loc['Interest Expense'].iloc[2],2),
                       'EBIT/ interest payments. Ability to pay debt interests']
    variation_over_years(cov_ratios_df,1)

    cov_ratios_df.loc[2]=['Debt to EBITDA', 
                       round(balance_sheet.loc['Total Debt'].iloc[0]/income_statement.loc['EBITDA'].iloc[0],2),
                       round(balance_sheet.loc['Total Debt'].iloc[1]/income_statement.loc['EBITDA'].iloc[1],2),
                       round(balance_sheet.loc['Total Debt'].iloc[2]/income_statement.loc['EBITDA'].iloc[2],2),
                       'Number of years to pay off debt from earnings']
    variation_over_years(cov_ratios_df,3, True)

    print(cov_ratios_df.to_markdown())
    Ollama_Analysis_Quant(cov_ratios_df, 'coverage')
except: print('Computing the ratios of this section failed. This might be due to a change of the financial statement layout on Yahoo Finance which varies when fetched from certain providers.')
print(' ')

print(colored('   Cash flows calculations: Operational ratios:', 'blue', 'on_white', attrs=['bold']))

cf_ratios_df = pd.DataFrame(columns=['Metric'] + income_statement.columns[-3:].tolist() + ['Description'])

try:
    cf_ratios_df.loc[0]=['CFI: Investing Cashflow', 
                       round(cashflow.loc['Investing Cash Flow'].iloc[0],2),
                       round(cashflow.loc['Investing Cash Flow'].iloc[1],2),
                       round(cashflow.loc['Investing Cash Flow'].iloc[2],2),
                       'cash received on asset sales - investment in assets']
    variation_over_years(cf_ratios_df,1)

    cf_ratios_df.loc[2]=['CFF: Financing Cashflow', 
                       round(cashflow.loc['Financing Cash Flow'].iloc[0],2),
                       round(cashflow.loc['Financing Cash Flow'].iloc[1],2),
                       round(cashflow.loc['Financing Cash Flow'].iloc[2],2),
                       'change in debt/common stock + cash dividends paid']
    variation_over_years(cf_ratios_df,3)

    cf_ratios_df.loc[4]=['CFO: Cashflow from Operations', 
                       round(cashflow.loc['Operating Cash Flow'].iloc[0],2),
                       round(cashflow.loc['Operating Cash Flow'].iloc[1],2),
                       round(cashflow.loc['Operating Cash Flow'].iloc[2],2),
                       'Qualitative if higher than reported earnings']
    variation_over_years(cf_ratios_df,5)

    cf_ratios_df.loc[6]=['--------', '--------', '--------', '--------', '------------------------------']

    cf_ratios_df.loc[7]=['CFO to revenue', 
                       round(cashflow.loc['Operating Cash Flow'].iloc[0]/income_statement.loc['Total Revenue'].iloc[0],2),
                       round(cashflow.loc['Operating Cash Flow'].iloc[1]/income_statement.loc['Total Revenue'].iloc[1],2),
                       round(cashflow.loc['Operating Cash Flow'].iloc[2]/income_statement.loc['Total Revenue'].iloc[2],2),
                       'CFO/ net revenue']
    variation_over_years(cf_ratios_df,8)

    cf_ratios_df.loc[9]=['Cash return on assets', 
                       round(cashflow.loc['Operating Cash Flow'].iloc[0]/balance_sheet.loc['Total Assets'].iloc[0],2),
                       round(cashflow.loc['Operating Cash Flow'].iloc[1]/balance_sheet.loc['Total Assets'].iloc[1],2),
                       round(cashflow.loc['Operating Cash Flow'].iloc[2]/balance_sheet.loc['Total Assets'].iloc[2],2),
                       'CFO/ total assets']
    variation_over_years(cf_ratios_df,10)

    cf_ratios_df.loc[11]=['Debt coverage', 
                       round(cashflow.loc['Operating Cash Flow'].iloc[0]/balance_sheet.loc['Total Debt'].iloc[0],2),
                       round(cashflow.loc['Operating Cash Flow'].iloc[1]/balance_sheet.loc['Total Debt'].iloc[1],2),
                       round(cashflow.loc['Operating Cash Flow'].iloc[2]/balance_sheet.loc['Total Debt'].iloc[2],2),
                       'CFO/ total debt']
    variation_over_years(cf_ratios_df,12)

    print(cf_ratios_df.to_markdown())
    Ollama_Analysis_Quant(cf_ratios_df, 'solvency')
except: print('Computing the ratios of this section failed. This might be due to a change of the financial statement layout on Yahoo Finance which varies when fetched from certain providers.')

## Step 5: Appendix ##------------------------------------------------------------------
print(boxen("Appendix: Financial statements in detail", color="cyan"))
#This way, if something seems unusual in the ratios/ AI commentaries, the user can check out for himself the raw data. It is also used for debugging when adding new features. Limitation: The conversion in billions/ millions/... also affects percentages such as tax rates.

print(boxen(f'Income statement (expressed in {unit}):'))
print(income_statement.to_markdown())
print(boxen(f'Balance sheet (expressed in {unit}):'))
print(balance_sheet.to_markdown())
print(boxen(f'Cashflow Statement (expressed in {unit}):'))
print(cashflow.to_markdown())

## Credits ##------------------------------------------------------------------
print(boxen("Code: © G.RUQUILLA - CC BY-NC-SA"))
print("This license enables reusers to distribute, remix, adapt, and build upon the material in any medium or format for noncommercial purposes only, and only so long as attribution is given to the creator. If you remix, adapt, or build upon the material, you must license the modified material under identical terms. Data is fetched from Yahoo Finance through Yfinance and Google RSS feeds through pygooglenews. The information provided is for general informational and educational purposes only and does not constitute investment advice, financial advice, trading advice, or any other form of advice.")


# In[ ]:




