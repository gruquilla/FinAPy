![alt text](https://github.com/gruquilla/FinAPy/blob/main/FINAPY%20logo.jpg "Logo")
Single-stock analysis using Python and machine learning tools (Ollama, LSTM). CC BY-NC-SA.

## Table of content
Work in progress - To be published soon.

## What is FinAPy?
FinAPY (FINancial Analysis with PYthon) is a research project using Python that aims to provide a synthetic and yet comprehensive overview of the main characteristics of a company and its stock from simply its ticker.<br />
The main challenge was to work on a way to process financial data using AI locally, without sending any data to servers on the other side of the globe. Using Ollama and certain Python libraries helped to reach that objective.<br />
<br />
The project relies on several libraries, but gets its data using yfinance and pygooglenews mainly. The main features include: <br />
* A basic synthesis about the company: name, location, sector, industry,...
* A market data analysis:
  * The visualisation of 1 year of stock prices with a candlestick chart, SMA, volatility, volume, daily returns, main events, and also important news from various websites and newspapers.
  * A 60-day LSTM (Long-Short Term Memory - A neural network technique) price forecast with 3 scenarios.
  * The daily returns histogram, with computation of basic statistics such as kurtosis and skewness.
  * An optional AI analysis (using Ollama and GPT OSS:20B) that provides additional insight and interpretation in only a few lines.
    
* A ratio-based analysis, computed from the financial statements of the last 3 years:
  * The calculation and variation of the main financial ratios, in terms of profitability/assets/liquidity/solvency (level of debt + coverage)/ operational ratios.
  * An additional AI analysis (still using Ollama) providing an interpretation of the ratios in link with the industry and the PESTEL environment, and also highlighting aspects left out by the ratios.

## Video demonstration
Work in progress - To be published soon.

## Quick preview
Example using the ticker of Netflix (NFLX):
Example of the market data analysis component:
![alt text](https://github.com/gruquilla/FinAPy/blob/main/previewmarket.jpg "Candlestick graph")

Example of the Operational ratios component:
![alt text](https://github.com/gruquilla/FinAPy/blob/main/ratiospreview.jpg "Ratio analysis")

The same kind of analysis is provided with the different ratio components (Profitability/ assets/ liquidity/ solvency (level of debt + coverage)).

## The project in detail
Work in progress - To be published soon.

## Potential practical applications
As is: Work in progress - To be published soon. <br />
Once refined/ adapted/ modified: Work in progress - To be published soon. <br />
## Limitations and planned developments
My work focuses at the moment on improving the following aspects:
* Improving the sentiment analysis of the news with Finbert, as the accuracy is sometimes off when articles talk about several companies.
* Integrating sector/industry averages of the ratios, and potentially Q1/Q3 to help assess the financial situation of a company within its industry.
* Adding metrics to assess the relevancy of the LSTM forecast.
* Certain providers don't issue the financial statements in the same format as others on Yahoo Finance, which leads to the ratio not being calculated at all in certain rare cases.
* Converting certain elements/ blocks into a library for educational purposes.
* I'm planning the integration of elements of this future library to a portfolio analysis project.
<br />
<br />
Code: © G.RUQUILLA - CC BY-NC-SA <br />
This license enables reusers to distribute, remix, adapt, and build upon the material in any medium or format for noncommercial purposes only, and only so long as attribution is given to the creator. If you remix, adapt, or build upon the material, you must license the modified material under identical terms. Data is fetched from Yahoo Finance through Yfinance and Google RSS feeds through pygooglenews. The information provided is for general informational and educational purposes only and does not constitute investment advice, financial advice, trading advice, or any other form of advice.
