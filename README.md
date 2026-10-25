# Stock Price Predictor 📈

A multi-input Recurrent Neural Network that predicts daily stock price percentage changes by combining real-time financial data with sentiment analysis of Reddit discussions.

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.14%2B-orange)](https://www.tensorflow.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

> **Note:** This repository holds the project's documentation (this README and [ARCHITECTURE.md](ARCHITECTURE.md)). The implementation, `StockPredictorFinal.py`, isn't included yet.

## 🎯 Overview

This project fuses two distinct data sources—structured financial time series and unstructured social media text—to predict stock market movements. By analyzing investor sentiment on Reddit alongside traditional price data, it tests whether crowd psychology helps predict short-term price changes.

**Key Features:**
- 🔄 Real-time data collection from Reddit and Yahoo Finance APIs
- 💬 VADER sentiment analysis optimized for social media
- 🧠 Multi-input LSTM architecture with three separate branches
- 📊 Date-synchronized fusion of social and financial data
- 📈 Predicts percentage changes rather than absolute prices
- 🎯 Configurable stocks and subreddits

## 🚀 Inspiration

Inspired by the film *Limitless* (2011), where Bradley Cooper's character analyzes not just financial data but rumors and social sentiment to make successful trades. This project asks: **What if we could build an AI that thinks like that?**

> "The market is moved by people, and people talk before they trade."

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────┐
│                   Input Layer (3 Branches)              │
├──────────────┬──────────────┬──────────────────────────┤
│ Post Titles  │  Post Bodies │  Numerical Features      │
│              │              │  (score, ratio, etc.)    │
├──────────────┼──────────────┼──────────────────────────┤
│ Embedding    │  Embedding   │  Direct Input            │
│ (100D)       │  (100D)      │                          │
├──────────────┼──────────────┤                          │
│ LSTM         │  LSTM        │                          │
│ (64 units)   │  (64 units)  │                          │
├──────────────┼──────────────┤                          │
│ GlobalMax    │  GlobalMax   │                          │
│ Pooling      │  Pooling     │                          │
└──────────────┴──────────────┴──────────────────────────┘
                        │
                        ▼
                 ┌──────────────┐
                 │ Concatenate  │
                 └──────────────┘
                        │
                        ▼
                 ┌──────────────┐
                 │ Dense (32)   │
                 │ + ReLU       │
                 └──────────────┘
                        │
                        ▼
                 ┌──────────────┐
                 │ Dropout(0.2) │
                 └──────────────┘
                        │
                        ▼
                 ┌──────────────┐
                 │ Dense (1)    │
                 │ Linear       │
                 └──────────────┘
                        │
                        ▼
              Price Change Prediction
```

## 📊 How It Works

### 1. Data Collection
- **Financial Data**: Yahoo Finance API retrieves OHLCV (Open, High, Low, Close, Volume) data
- **Social Data**: Reddit API (PRAW) scrapes posts from r/stocks, r/options, r/investing
- **Synchronization**: Each Reddit post is matched with the stock's price change on that same day

### 2. Sentiment Analysis
- **VADER** (Valence Aware Dictionary and sEntiment Reasoner) analyzes post titles and bodies
- Designed specifically for social media, handles slang, emojis, and informal language
- Separate sentiment scores for titles (often sensational) and bodies (more detailed)

### 3. Feature Processing
- **Text**: Tokenization, stop word removal, sequence padding
- **Numerical**: StandardScaler normalization for karma, comments, ratios
- **Temporal**: Unix timestamp conversion for date-based patterns

### 4. Neural Network Training
- **Architecture**: Multi-input RNN with separate LSTM branches
- **Loss**: Mean Squared Error (regression task)
- **Optimizer**: Adam (learning_rate=0.001)
- **Output**: Predicted percentage change for next trading day

## 🛠️ Installation

### Prerequisites
- Python 3.8 or higher
- pip package manager

### Setup

1. **Clone the repository**
```bash
git clone https://github.com/yourusername/stock-price-predictor.git
cd stock-price-predictor
```

2. **Install dependencies**
```bash
pip install -r requirements.txt
```

3. **Download NLTK data**
```python
python -c "import nltk; nltk.download('vader_lexicon'); nltk.download('stopwords'); nltk.download('punkt')"
```

4. **Configure Reddit API credentials**
   - Visit [reddit.com/prefs/apps](https://www.reddit.com/prefs/apps)
   - Create a new application (script type)
   - Copy your credentials to `StockPredictorFinal.py`:
   ```python
   reddit = praw.Reddit(
       client_id='YOUR_CLIENT_ID',
       client_secret='YOUR_CLIENT_SECRET',
       user_agent='YOUR_USER_AGENT'
   )
   ```

## 🎮 Usage

### Basic Usage

```python
python StockPredictorFinal.py
```

### Configuration

Modify these variables in the script to customize:

```python
# Stock symbols to analyze
symbols = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META', 'NFLX', 'TSLA', 'NVDA']

# Subreddits to scrape
subreddits = ['stocks', 'options', 'investing']

# Posts per stock per subreddit
limit = 100

# Training parameters
epochs = 30
batch_size = 32
```

### Output

The script generates two CSV files:
- `predictions3.csv` - Training set predictions
- `predictions4.csv` - Validation set predictions

Each file contains:
- Post title and date
- Stock symbol
- Predicted percentage change
- Actual percentage change

## 📈 Performance

### Multi-Stock Portfolio (10 stocks, 10 posts each)
- **Training MSE**: 0.3866
- **Validation MSE**: 0.1785

### Single-Stock Deep Dive (AAPL, 100 posts)
- **Training MSE**: 0.0602
- **Validation MSE**: 0.0024

### How Much to Trust These Numbers

Validation error well *below* training error (25× lower in the AAPL run) is a warning sign, not evidence of a good model:

- **Leakage through shared targets.** The data is split randomly by post (`train_test_split`, 80/20). Every post about the same stock on the same day has the same target, so the validation set contains posts whose answers the model has effectively seen in training. This is strongest in the single-stock run, where 100 posts cover few trading days.
- **No baseline.** The errors aren't compared to a trivial predictor (such as always predicting 0% change), so they don't show whether sentiment adds signal.
- **Same-day timing.** Posts are matched to the price change of the day they were written, so some may be reacting to the move rather than predicting it.

A fair test would split by date, report a naive baseline alongside the model, and use only posts written before the market opens. Per-feature importance hasn't been measured; an ablation under a date-based split would be the way to do it.

## 🧪 Technical Stack

**Data Collection:**
- `praw` - Reddit API wrapper
- `yfinance` - Yahoo Finance data

**NLP & Sentiment:**
- `nltk` - Natural language processing
- VADER lexicon - Social media sentiment analysis

**Machine Learning:**
- `tensorflow` / `keras` - Neural network framework
- `scikit-learn` - Preprocessing and evaluation

**Data Processing:**
- `pandas` - Data manipulation
- `numpy` - Numerical operations

## 📚 Project Structure

The original project also had `StockPredictorFinal.py` (the implementation) and generated `predictions3.csv` / `predictions4.csv` files, which aren't in this repository.

```
Predicting-Stock-Price-Changes-Using-Reddit-Sentiment/
├── README.md          # This file
├── ARCHITECTURE.md    # Detailed architecture guide
├── requirements.txt   # Python dependencies
├── LICENSE            # MIT License
└── .gitignore
```

## 🔬 Research Foundation

This project builds on established research:

- **Systematic Review**: 69 papers showing RNNs/LSTMs outperform traditional ML for time series ([ScienceDirect](https://www.sciencedirect.com/science/article/pii/S2590291124000615))
- **LSTM Stock Prediction**: Netflix 3-year model achieving 0.168 MSE ([ProjectPro](https://www.projectpro.io/article/stock-price-prediction-using-machine-learning-project/571))
- **VADER Sentiment**: Social media-optimized sentiment analysis ([GitHub](https://github.com/cjhutto/vaderSentiment))

### Novel Contributions
- Multi-input architecture separating titles and bodies
- Date-synchronized fusion of Reddit and financial data
- Dual sentiment analysis (title + body separately)
- Training across multiple stocks rather than one

## ⚠️ Limitations & Disclaimers

**This is an educational project and should NOT be used for actual trading decisions.**

**Known Limitations:**
- Limited dataset size due to API rate limits
- Random train/validation split leaks shared targets (see Performance)
- No baseline comparison yet
- No handling of after-hours trading or pre-market
- Sentiment analysis accuracy varies (sarcasm, context)
- Market efficiency limits predictability

**Disclaimers:**
- Past performance does not guarantee future results
- Stock market prediction is inherently uncertain
- This project is for learning purposes only
- Always consult financial professionals for investment advice

## 🚧 Future Improvements

- [ ] Add Twitter/X data integration
- [ ] Implement news article scraping
- [ ] Include company fundamentals (earnings, balance sheets)
- [ ] Add technical indicators (RSI, MACD, moving averages)
- [ ] Implement attention mechanisms / Transformer architecture
- [ ] Create ensemble models for robust predictions
- [ ] Build data caching system for faster iterations
- [ ] Deploy as real-time web service
- [ ] Add backtesting framework for strategy simulation
- [ ] Implement SHAP/LIME for model explainability
- [ ] Create interactive web dashboard

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- Inspired by the film *Limitless* (2011)
- VADER sentiment analysis by C.J. Hutto
- Reddit API community (PRAW)
- Yahoo Finance for free financial data
- TensorFlow/Keras development team
- Academic research community in financial ML

## 📞 Contact & Support

- **Portfolio:** [your-website.com](https://morganwhite13.github.io/)
- **Email:** morgan13white@icloud.com
- **LinkedIn:** [Your LinkedIn](https://www.linkedin.com/in/morgan-white-95b245237/)

---

**⚡ Remember**: The stock market is influenced by countless factors. This model captures social sentiment signals, but should never be your sole basis for investment decisions. Always do your own research and consider consulting financial advisors.
