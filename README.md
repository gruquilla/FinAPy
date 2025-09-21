# FinAPy
![alt text](https://github.com/gruquilla/FinAPy/blob/main/FINAPY%20logo.jpg "Logo")
Single-stock analysis using Python and machine learning tools (Ollama, LSTM).

## Table of content
-[What is FinAPy?](#What is FinAPy)
## What is FinAPy?
FinAPY (FINancial Analysis with Python) is a research project using Python that aims to provide a synthetic and yet comprehensive overview of the main characteristics of a company and its stock from simply its ticker.<br />
<br />
The project relies on several libraries, but gets its data using yfinance and pygooglenews mainly. The main features include: <br />
* A basic synthesis about the company: name, location, sector, industry,...
* A market data analysis:
  * The visualisation of 1 year of stock prices with a candlestick chart, SMA, volatility, volume, daily returns, main events, and also important news from various websites and newspapers
  * A 60-day LSTM price forecast with 3 scenarios
  * The daily returns histogram, with computation of basic statistics such as kurtosis and skewness
  * An optional AI analysis (using Ollama and GPT OSS:20B) that provides additional insight and interpretation in only a few lines.
    
* A ratio-based analysis, computed from the financial statements of the last 3 years:
  * The calculation and variation of the main financial ratios, in terms of profitability/ assets/ liquidity/ solvency (level of debt + coverage)/ operational ratios.
  * An additional AI analysis (still using Ollama) providing an interpretation of the ratios in link with the industry and the PESTEL environment, and also highlighting aspects left out by the ratios.
