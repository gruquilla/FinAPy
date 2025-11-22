![alt text](https://github.com/gruquilla/FinAPy/blob/main/media/finapyshowcase.jpg "Logo")
Single-stock analysis using Python and local machine learning/ AI tools (Ollama, LSTM). CC BY-NC-SA.<br />
<br />
📘 Access the FinAPy Notebook: <br />
[Open Version 1.2](https://github.com/gruquilla/FinAPy/blob/main/Jupyter%20Notebook%20files/FinAPy%20V1.2%20Jupyter%20Notebook%20Version.ipynb) <br />
[Open Version 1.1](https://github.com/gruquilla/FinAPy/blob/main/Jupyter%20Notebook%20files/FinAPy%20V1.1%20Jupyter%20Notebook%20Version.ipynb)
<br />
Check requirements: <br />
[requirements.txt](https://github.com/gruquilla/FinAPy/blob/main/requirements.txt)
<br />
[Download Ollama to have access to the local LLM features](https://ollama.com/download)
<br />
Information: Ollama and the local LLM features require a discrete GPU to run properly, with at least 8GB of VRAM. Your computer should also have at least 16GB of RAM. If you are below these specifications, you can still run the script but without these AI features.
<br />

## Table of contents
- [What is FinAPy?](#what-is-finapy)
- [How to run the project?](#how-to-run-the-project?)
- [Video demonstration](#video-demonstration)
- [Quick preview](#quick-preview)
- [The project in detail](#the-project-in-detail)
- [Potential practical applications](#potential-practical-applications)
- [Limitations and planned developments](#limitations-and-planned-developments)
  
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

## How to run the project?
- Step 1: Download the latest notebook.
- Step 2: install the libraries required (listed on the requirements.txt file at the top of this README)
- Step 3 (optional- for AI features only): Download Ollama (Link at the top of the Readme)
- Step 4 (optional- for AI features only): Install gpt-oss-20b on Ollama: Open Ollama, select as "model" "gpt-oss-20b" and send a random prompt to start downloading the model
- Step 5: Open the notebook in Python (keep Ollama open on the side) and run the script.

## Video demonstration
Work in progress - To be published soon.

## Quick preview
Example using the ticker of Netflix (NFLX):
Example of the market data analysis component:
![alt text](https://github.com/gruquilla/FinAPy/blob/main/media/previewmarket.jpg "Market data visualisation graph")

Example of the Operational ratios component:
![alt text](https://github.com/gruquilla/FinAPy/blob/main/media/ratiospreview.jpg "Ratio analysis")

The same kind of analysis is provided with the different ratio components (Profitability/ assets/ liquidity/ solvency (level of debt + coverage)).

## The project in detail
Here is how FinAPy works:
* **Prologue: Ticker input collection and essential functions and data:** <br />
  _In this part, the program gets in input a ticker from the user, and asks wether or not he wants to enable the AI analysis. Then, it generates a short summary about the company fetching information from Yahoo Finance, so the user has something to read while the next step proceeds. It also fetches the main financial metrics and computes additional ones._ <br /> <br />
* **Step 1: Events and news fetching:** <br />
  _This part fetches stock events from Yahoo Finance and news from Google RSS feed. It also generates a sentiment analysis about the articles fetched using FinBERT._ <br /> <br />
* **Step 2: Forecast using Machine Learning LSTM:**  <br />
  _This part creates a baseline scenario from a LSTM forecast. The forecast covers 60 days and is trained from 100 last values of close/ high/low prices. It is a quantiative model only. An optimistic and pessimistic scenario are then created by tweaking the main baseline to give a window of prediction. They do not integrate macroeconomic factors, specific metric variations nor Monte Carlo simulations for the moment._ <br /> <br />
* **Step 3: Market data restitution:** <br />
  _This part is dedicated to restitute graphically the previously computed data. It also computes CFA classical metrics (histogram of returns, skewness, kurtosis) and their explanation. The part concludes with an Ollama AI commentary of the analysis._ <br /> <br />
* **Step 4: Financial statement analysis:** <br />
  _This part is dedicated to the generation of the main ratios from the financial statements of the last 3 years of the company. Each part concludes with an Ollama AI commentary on the ratios. The analysis includes an overview of the variation, and highlights in color wether the change is positive or negative. Each ratio is commented so you can understand what they represent/ how they are calculated. The ratios include:_ <br />
    * **Profitability ratios:** Profit margin, ROA, ROCE, ROE,...
    * **Asset related ratios:** Asset turnover, working capital.
    * **Liquidity ratios:** Current ratio, quick ratio, cash ratio.
    * **Solvency ratios:** debt to assets, debt to capital, financial leverage, coverage ratios,...
    * **Operational ratios (cashflow related):** CFI/ CFF/ CFO ratios, cash return on assets,... 
    * **Bankrupcy and financial health scores**: Altman Z-score/ Ohlson O-score.<br /> <br />
* **Appendix: Financial statements:** <br />
  _A summary of the financial statements scaled for better readability in case you want to push the manual analysis further._ <br /> <br />

## Potential practical applications
As is: For educational and research purposes only:
  * Visualisation of ratios and metrics of any stock for students, researchers,...
  * CFA program candidates looking to visualise real-life cases ratios and data as learnt from the curriculum (skewness, kurtosis,... and other theoretical concepts professional models don't always explicitly show).
  * Financial ratio visualisation and analysis (informative only).
  * LSTM forecast preview
  * ... <br />

Once refined/ adapted/ modified, it might have the potential to implement:
  * In-house data analysis of financial information with a high level of confidentiality (the advantage of using Ollama locally). Main requirements: usage-adpated APIs for financial data and news/ hardware to run Ollama.
  * Qualitative analysis integration, with the graphical visualisation of news and their impact over time.
  * Weak signal additional detection layer with the AI analysis. It is not as reliable as a real human experienced analyst, but can sometimes highlight aspects to help deepen one's analysis.
  *  ... <br />
## Limitations and planned developments
My work focuses at the moment on improving the following aspects:
* Improving the sentiment analysis of the news with Finbert, as the accuracy is sometimes off when articles talk about several companies.
* Integrating sector/industry averages of the ratios, and potentially Q1/Q3 to help assess the financial situation of a company within its industry.
* Adding metrics to assess the relevancy of the LSTM forecast.
* Certain providers don't issue the financial statements in the same format as others on Yahoo Finance, which leads to the ratio not being calculated at all in certain rare cases.
* I'm planning the integration of elements of this script to a portfolio analysis project with multi-factor optimisation.
<br />
<br />
Code: © G.RUQUILLA - CC BY-NC-SA <br />
This license enables reusers to distribute, remix, adapt, and build upon the material in any medium or format for noncommercial purposes only, and only so long as attribution is given to the creator. If you remix, adapt, or build upon the material, you must license the modified material under identical terms. Data is fetched from Yahoo Finance through Yfinance and Google RSS feeds through pygooglenews. The information provided is for general informational and educational purposes only and does not constitute investment advice, financial advice, trading advice, or any other form of advice.
