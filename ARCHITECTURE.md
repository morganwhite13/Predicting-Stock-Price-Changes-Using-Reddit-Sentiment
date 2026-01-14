# Architecture Documentation

## System Overview

The Stock Price Predictor is a multi-stage machine learning pipeline that combines natural language processing with financial time series analysis. This document provides technical details about the architecture, design decisions, and data flow.

## High-Level Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                     Data Collection Layer                    │
├──────────────────────────┬──────────────────────────────────┤
│   Reddit API (PRAW)      │   Yahoo Finance API (yfinance)   │
│   - Post titles          │   - OHLCV data                   │
│   - Post bodies          │   - Historical prices            │
│   - Metadata             │   - Volume                       │
└──────────────┬───────────┴──────────────┬───────────────────┘
               │                          │
               ▼                          ▼
┌─────────────────────────────────────────────────────────────┐
│                   Data Synchronization                       │
│            Match posts to same-day stock data                │
│         Calculate percentage change: (Close-Open)/Open       │
└──────────────────────────┬──────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────┐
│                  Preprocessing Pipeline                      │
├─────────────────┬─────────────────┬─────────────────────────┤
│ Text Processing │   Sentiment     │   Feature Scaling       │
│ - Tokenization  │   - VADER       │   - StandardScaler      │
│ - Stop words    │   - Title score │   - Normalization       │
│ - Padding       │   - Body score  │                         │
└────────┬────────┴────────┬────────┴────────┬────────────────┘
         │                 │                 │
         ▼                 ▼                 ▼
┌─────────────────────────────────────────────────────────────┐
│                    Neural Network Model                      │
│                                                              │
│   Input Branch 1      Input Branch 2      Input Branch 3    │
│   (Post Titles)       (Post Bodies)       (Features)        │
│        │                   │                   │            │
│   Embedding            Embedding            Direct          │
│        │                   │                   │            │
│   LSTM (64)            LSTM (64)               │            │
│        │                   │                   │            │
│   GlobalMaxPool        GlobalMaxPool           │            │
│        │                   │                   │            │
│        └───────────┬───────┘───────────────────┘            │
│                    │                                         │
│               Concatenate                                    │
│                    │                                         │
│              Dense(32, ReLU)                                 │
│                    │                                         │
│              Dropout(0.2)                                    │
│                    │                                         │
│              Dense(1, Linear)                                │
│                    │                                         │
│           Percentage Change Prediction                       │
└─────────────────────────────────────────────────────────────┘
```

## Component Details

### 1. Data Collection Layer

#### Reddit API Integration (PRAW)

**Purpose**: Scrape stock-related discussions from financial subreddits

**Key Functions**:
```python
def get_reddit_posts(subreddits, symbols, limit=10):
    """
    Retrieves Reddit posts discussing specified stocks.
    
    Args:
        subreddits: List of subreddit names (e.g., ['stocks', 'investing'])
        symbols: List of stock tickers (e.g., ['AAPL', 'TSLA'])
        limit: Maximum posts per stock per subreddit
        
    Returns:
        DataFrame with columns:
        - title: Post title text
        - body: Post body text
        - score: Karma points
        - ratio: Upvote ratio (0-1)
        - num_comments: Comment count
        - date: Unix timestamp
        - dateConverted: Datetime object
        - symbol: Stock ticker
        - symbolIndex: Integer encoding of symbol
        - Percentage Change: Stock's daily % change
    """
```

**API Rate Limits**:
- 60 requests per minute
- Sequential processing to avoid hitting limits
- Built-in delays between requests

**Design Decision**: Why Reddit over Twitter?
- Twitter API severely restricted in free tier
- Reddit has generous rate limits
- Financial subreddits are high-quality sources
- PRAW library is well-maintained

#### Yahoo Finance Integration (yfinance)

**Purpose**: Retrieve historical stock price data

**Key Functions**:
```python
def get_stock_data(symbol):
    """
    Downloads complete historical data for a stock.
    
    Args:
        symbol: Stock ticker (e.g., 'AAPL')
        
    Returns:
        DataFrame with columns:
        - Open: Opening price
        - High: Daily high
        - Low: Daily low
        - Close: Closing price
        - Adj Close: Adjusted closing price
        - Volume: Trading volume
        - Symbol: Stock ticker
    """
```

**Data Coverage**:
- All available historical data (company IPO to present)
- Daily granularity
- Adjusted for splits and dividends

**Design Decision**: Why Yahoo Finance?
- Free API with no key required
- Reliable historical data
- Wide coverage (all major exchanges)
- Well-maintained Python library

### 2. Data Synchronization

**Challenge**: Reddit posts have timestamps, stock data has dates. Need to match them.

**Solution**:
```python
post_date = datetime.fromtimestamp(post.created_utc).strftime('%Y-%m-%d')

for index, row in stock_data.iterrows():
    if index.strftime('%Y-%m-%d') == post_date:
        percentage_change = ((row['Close'] - row['Open']) / row['Open']) * 100
```

**Edge Cases Handled**:
- Weekend posts (no trading) → percentage_change = 0
- After-hours posts → matched to next trading day
- Holidays → skipped (no market data)

**Design Decision**: Why percentage change?
- More meaningful than absolute price change
- Comparable across different stocks
- Normalized metric for prediction
- Standard in financial analysis

### 3. Preprocessing Pipeline

#### Text Preprocessing

**NLTK-Based Cleaning**:
```python
def preprocess_text(text):
    """
    Cleans and normalizes text for neural network input.
    
    Steps:
    1. Tokenization - Split into words
    2. Lowercase conversion - "Apple" → "apple"
    3. Alphanumeric filtering - Remove punctuation
    4. Stop word removal - Remove "the", "and", etc.
    5. Rejoin - Create clean string
    
    Example:
        Input:  "I'm buying $AAPL because the iPhone is AMAZING!!!"
        Output: "buying aapl iphone amazing"
    """
```

**Why this approach?**
- Reduces vocabulary size (faster training)
- Removes noise (punctuation doesn't help prediction)
- Retains semantic meaning
- Standard NLP practice

**Tokenization**:
```python
tokenizer_title = Tokenizer()
tokenizer_title.fit_on_texts(reddit_data['processed_title'])
X_title = tokenizer_title.texts_to_sequences(reddit_data['processed_title'])
```

**Vocabulary Management**:
- Separate vocabularies for titles and bodies
- Combined vocab for embedding layer
- Unknown words mapped to special token

**Sequence Padding**:
```python
max_len_title = max(len(seq) for seq in X_title)
X_title_pad = [seq + [0] * (max_len_title - len(seq)) for seq in X_title]
```

**Why dynamic padding?**
- Adapts to dataset (no hardcoded lengths)
- Wastes less memory than fixed large padding
- LSTM ignores padding zeros

#### Sentiment Analysis

**VADER Implementation**:
```python
from nltk.sentiment import SentimentIntensityAnalyzer

sia = SentimentIntensityAnalyzer()

def calculate_sentiment(text):
    """
    Calculates sentiment using VADER (Valence Aware Dictionary 
    and sEntiment Reasoner).
    
    Returns:
        float: Compound score from -1 (very negative) to +1 (very positive)
        
    Examples:
        "TSLA to the moon! 🚀🚀🚀" → +0.87
        "Company reports disappointing earnings" → -0.52
        "Holding shares, unsure about future" → +0.12
    """
    return sia.polarity_scores(text)['compound']
```

**Why VADER?**
- Designed for social media (handles slang, emojis, caps)
- Understands intensifiers ("VERY good" vs "good")
- Handles negations ("not good" vs "good")
- No training required (lexicon-based)

**Dual Sentiment Scores**:
- Title sentiment: Often sensationalized
- Body sentiment: More detailed analysis
- Model learns which to trust more

**Design Decision**: Why not fine-tuned BERT?
- VADER is faster (no GPU needed)
- Good enough for social media
- Lexicon-based is interpretable
- BERT would be overkill for this dataset size

#### Feature Scaling

**StandardScaler Application**:
```python
scaler = StandardScaler()
features = ['score', 'ratio', 'num_comments', 
            'Title Sentiment', 'Body Sentiment', 
            'date', 'symbolIndex']
reddit_data[features] = scaler.fit_transform(reddit_data[features])
```

**Why StandardScaler?**
- Karma scores: range 1-10,000+
- Sentiment: range -1 to +1
- Without scaling: karma dominates
- After scaling: all features contribute equally

**Formula**:
```
scaled_value = (value - mean) / standard_deviation
```

**Result**: All features centered at 0 with std=1

### 4. Neural Network Architecture

#### Multi-Input Model Design

**Keras Functional API**:
```python
# Define inputs
input_title = Input(shape=(max_len_title,))
input_body = Input(shape=(max_len_body,))
input_features = Input(shape=(num_features,))

# Process separately
title_branch = process_title(input_title)
body_branch = process_body(input_body)

# Merge
merged = Concatenate()([title_branch, body_branch, input_features])

# Prediction head
output = prediction_layers(merged)

# Build model
model = Model(inputs=[input_title, input_body, input_features], 
              outputs=output)
```

**Why multi-input?**
- Different data types need different processing
- Text requires embeddings, numbers don't
- Separate branches learn independently
- Late fusion combines learned representations

#### Embedding Layers

**Purpose**: Convert word indices to dense vectors

**Configuration**:
```python
Embedding(vocab_size=50000, output_dim=100, input_length=max_len)
```

**Parameters**:
- `vocab_size`: Number of unique words + 1 (for padding)
- `output_dim`: Embedding dimension (100D vectors)
- `input_length`: Maximum sequence length

**Why 100D embeddings?**
- Balances expressiveness and computational cost
- Smaller than word2vec (300D) but sufficient
- Trained from scratch on our financial corpus

**Example**:
```
Word "bullish" (index 247) → [0.23, -0.15, 0.67, ..., 0.44] (100 values)
```

#### LSTM Layers

**Architecture**:
```python
LSTM(units=64, return_sequences=True)
```

**Parameters**:
- `units=64`: Number of LSTM cells (memory capacity)
- `return_sequences=True`: Output sequence for pooling

**LSTM Cell Structure**:
```
┌─────────────────────────────────┐
│          LSTM Cell              │
│                                 │
│  ┌──────┐  ┌──────┐  ┌──────┐  │
│  │Forget│  │ Input│  │Output│  │
│  │ Gate │  │ Gate │  │ Gate │  │
│  └───┬──┘  └───┬──┘  └───┬──┘  │
│      │         │         │      │
│      └────┬────┴────┬────┘      │
│           │         │           │
│      Cell State  Hidden State   │
└─────────────────────────────────┘
```

**Why LSTM over simple RNN?**
- Solves vanishing gradient problem
- Remembers long-term dependencies
- Selective memory (forget gate)
- Proven for NLP tasks

**Sequence Processing Example**:
```
Input sequence: ["stock", "price", "rising", "fast"]
                    ↓       ↓        ↓        ↓
LSTM outputs:    [h1]    [h2]     [h3]     [h4]
                 (64D)   (64D)    (64D)    (64D)
```

#### Global Max Pooling

**Purpose**: Extract most important feature from sequence

**Operation**:
```python
GlobalMaxPooling1D()
# Input:  [h1, h2, h3, h4] → shape (seq_len, 64)
# Output: [max(h1...h4)]   → shape (64,)
```

**Why max pooling over average?**
- One word can determine sentiment
- "AMAZING" has high activation
- Max captures that peak
- Average would dilute signal

#### Concatenation Layer

**Purpose**: Merge all three input branches

```python
Concatenate()([title_pooled, body_pooled, features])
# Output shape: (64 + 64 + 7) = (135,)
```

**Merged representation**:
- 64D from title LSTM
- 64D from body LSTM  
- 7D from numerical features
- Total: 135D feature vector

#### Dense Layers

**Prediction Head**:
```python
Dense(32, activation='relu')    # Hidden layer
Dropout(0.2)                     # Regularization
Dense(1, activation='linear')    # Output layer
```

**ReLU Activation**:
```
f(x) = max(0, x)
# Positive values pass through
# Negative values become zero
```

**Why ReLU?**
- Avoids vanishing gradients
- Fast to compute
- Works well in practice
- Standard for hidden layers

**Linear Output**:
```
f(x) = x
# No transformation
# Can output any real number (needed for % change)
```

**Why linear for output?**
- Regression task (not classification)
- Need to predict continuous values
- Can be positive or negative

#### Dropout Regularization

**Purpose**: Prevent overfitting

**Mechanism**:
```python
Dropout(0.2)  # Randomly drops 20% of neurons during training
```

**During Training**:
```
Before:  [0.5, 0.3, 0.8, 0.2, 0.9]
After:   [0.0, 0.3, 0.0, 0.2, 0.9]  # Random 20% zeroed
```

**During Inference**: No dropout (use all neurons)

**Why 0.2 (20%)?**
- Empirically good balance
- Too high: underfitting
- Too low: overfitting
- Standard starting point

### 5. Training Configuration

#### Optimizer: Adam

**Configuration**:
```python
Adam(learning_rate=0.001)
```

**Adam Algorithm**:
- Adaptive learning rates per parameter
- Momentum for faster convergence
- Bias correction for early training

**Why Adam over SGD?**
- Works well with little tuning
- Handles sparse gradients (text data)
- Faster convergence
- Industry standard

#### Loss Function: MSE

**Mean Squared Error**:
```python
loss = mean((y_true - y_pred)²)
```

**Why MSE?**
- Regression task (continuous output)
- Penalizes large errors more (quadratic)
- Differentiable (needed for backprop)
- Standard for regression

**Alternative considered**: MAE (Mean Absolute Error)
- Less sensitive to outliers
- But MSE emphasizes accuracy on large movements (what we want)

#### Training Process

**Data Split**:
```python
train_test_split(X, y, test_size=0.2, random_state=42)
# 80% training, 20% validation
```

**Training Loop**:
```python
model.fit(
    [X_train_title, X_train_body, X_train_features],
    y_train,
    validation_data=([X_val_title, X_val_body, X_val_features], y_val),
    epochs=30,
    batch_size=32
)
```

**Hyperparameters**:
- `epochs=30`: Full passes through dataset
- `batch_size=32`: Samples per gradient update

**Why these values?**
- 30 epochs: Balances training time and convergence
- Batch size 32: Standard, fits in memory

## Data Flow Diagram

```
Reddit Post: "AAPL earnings beat! 🚀"
    │
    ├─> Title: "AAPL earnings beat! 🚀"
    │       └─> Preprocess: "aapl earnings beat"
    │           └─> Tokenize: [245, 1823, 932]
    │               └─> Pad: [245, 1823, 932, 0, 0, ...]
    │                   └─> Embed: [[0.2,...], [0.5,...], ...]
    │                       └─> LSTM: [h1, h2, h3, ...]
    │                           └─> MaxPool: [0.87, -0.23, ...]
    │
    ├─> Body: "Revenue up 15%, EPS beat..."
    │       └─> Similar pipeline
    │           └─> MaxPool: [0.65, 0.34, ...]
    │
    └─> Features: [score=245, ratio=0.92, sentiment=0.87, ...]
            └─> Scale: [1.2, 0.8, 2.1, ...]
            
[0.87,...] + [0.65,...] + [1.2,...] = [135D vector]
    │
    └─> Dense(32): [0.45, -0.12, ..., 0.88]
        └─> Dropout: [0.45, 0.00, ..., 0.88]
            └─> Dense(1): 3.2
            
Prediction: +3.2% price change
```

## Performance Characteristics

### Time Complexity

**Data Collection**:
- Reddit API: O(n × m × p) where n=stocks, m=subreddits, p=posts
- Yahoo Finance: O(n × d) where d=days
- Bottleneck: API rate limits

**Preprocessing**:
- Tokenization: O(n × w) where w=average words
- Padding: O(n × max_len)
- Scaling: O(n × f) where f=features

**Training**:
- Forward pass: O(b × L × H²) where b=batch, L=length, H=hidden units
- Backward pass: Same as forward
- Per epoch: O(samples / batch_size) iterations

### Space Complexity

**Model Parameters**:
- Embeddings: vocab_size × 100 ≈ 5M parameters
- LSTM: 4 × (64 × 64 + 64 × 100) ≈ 42K parameters each
- Dense: 135 × 32 + 32 × 1 ≈ 4K parameters
- **Total**: ~5.1M parameters

**Memory Usage**:
- Model: ~20MB (float32)
- Data: ~100MB (for 1000 posts)
- Training: ~500MB (gradients, optimizer state)

## Design Decisions & Tradeoffs

### Why Not Use Transformer?

**Considered**: BERT, GPT for text encoding

**Decided against because**:
- Dataset too small to fine-tune effectively
- LSTM sufficient for short texts (posts)
- Training time would be 10x longer
- Interpretability harder

**Future consideration**: If dataset grows to 100K+ posts

### Why Not Technical Indicators?

**Considered**: Adding RSI, MACD, moving averages

**Decided against because**:
- Focus on social sentiment signal
- Technical indicators already well-studied
- Combining both would dilute research focus
- Can be added in future iterations

### Why Not Attention Mechanism?

**Considered**: Attention over LSTM outputs

**Decided against because**:
- GlobalMaxPooling simpler and faster
- Attention adds complexity
- Dataset size doesn't justify it
- Marginal benefit for this task

**Future consideration**: If accuracy plateaus

## Extensibility Points

### Easy Extensions

1. **Add stocks**: Append to `symbols` list
2. **Add subreddits**: Append to `subreddits` list
3. **Change model size**: Modify LSTM units, Dense units
4. **Adjust training**: Change epochs, batch_size, learning_rate

### Moderate Extensions

1. **Add features**: Include technical indicators
2. **Alternative sentiment**: Replace VADER with BERT
3. **Ensemble methods**: Combine multiple models
4. **Data caching**: Save/load preprocessed data

### Complex Extensions

1. **Multi-day prediction**: Predict next 3-5 days
2. **Attention mechanisms**: Add attention layers
3. **Real-time deployment**: Build API endpoint
4. **Explainability**: Integrate SHAP/LIME

## Testing & Validation

### Unit Testing (Future Work)

```python
def test_preprocess_text():
    input_text = "I'm buying $AAPL!!!"
    expected = "buying aapl"
    assert preprocess_text(input_text) == expected

def test_sentiment_analysis():
    positive_text = "Amazing stock! 🚀"
    negative_text = "Terrible earnings report"
    assert calculate_sentiment(positive_text) > 0.5
    assert calculate_sentiment(negative_text) < -0.3
```

### Integration Testing

- Verify Reddit API connection
- Verify Yahoo Finance data retrieval
- Verify date synchronization
- Verify model can train without errors

### Performance Testing

- Measure data collection time
- Profile memory usage during training
- Benchmark prediction latency
- Test with varying dataset sizes

## Monitoring & Logging (Future Work)

### Proposed Logging

```python
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

logger.info(f"Collecting data for {len(symbols)} stocks")
logger.info(f"Model training started with {len(X_train)} samples")
logger.warning(f"Validation loss increasing - possible overfitting")
```

### Metrics to Track

- Training loss per epoch
- Validation loss per epoch
- Prediction accuracy (MAE, RMSE)
- API call counts and failures
- Processing time per stage

## References

- LSTM Architecture: [Hochreiter & Schmidhuber, 1997](http://www.bioinf.jku.at/publications/older/2604.pdf)
- VADER Sentiment: [Hutto & Gilbert, 2014](https://github.com/cjhutto/vaderSentiment)
- Adam Optimizer: [Kingma & Ba, 2014](https://arxiv.org/abs/1412.6980)
- Dropout: [Srivastava et al., 2014](http://jmlr.org/papers/v15/srivastava14a.html)
