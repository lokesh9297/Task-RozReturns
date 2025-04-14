# Regime Detection via Unsupervised Learning from Order Book and Volume Data

## 🚀 Objective
This project aims to **segment the financial market into distinct behavioral regimes** using real-time **order book** and **trade volume** data. These regimes are characterized along three key axes:

- **Trending vs Mean-Reverting**
- **Volatile vs Stable**
- **Liquid vs Illiquid**

The segmentation is achieved using **unsupervised learning**, with no predefined labels, making use of raw market microstructure data.

---

## 📊 Data Description

Two primary datasets were used:

- **Order Book Snapshots (`depth20_1000ms`)**  
  Captures the top 20 bid and ask levels (price and quantity) at 1-second intervals.

- **Aggregated Trades (`aggTrade`)**  
  Contains individual trade events, including time, price, quantity, and market maker side.

Both datasets were parsed from `.txt` files and aligned on timestamps for analysis.

---

## 🔍 Step-by-Step Workflow

### 1. Feature Engineering
Hand-crafted features were extracted to quantify key aspects of market behavior.

**Liquidity & Depth Features**
- Bid-Ask Spread: `ask_price_1 - bid_price_1`
- Microprice: `(bid_price_1 × ask_qty_1 + ask_price_1 × bid_qty_1) / (bid_qty_1 + ask_qty_1)`
- Order Book Imbalance: `(bid_qty_1 - ask_qty_1) / (bid_qty_1 + ask_qty_1)`
- Cumulative Depth: `sum(bid_qty_1 to bid_qty_20)`, `sum(ask_qty_1 to ask_qty_20)`
- Sloped Depth: Measures size decay across levels

**Price Action & Volatility**
- Mid-Price: `(bid_price_1 + ask_price_1) / 2`
- Log Returns: `log(mid_price_t / mid_price_t-1)`
- Rolling Volatility (10s, 30s)

**Volume-Based Features**
- Volume Imbalance (Buy vs Sell)
- Cumulative Volume (10s, 30s)
- VWAP Shift over short windows
- Trade Wipe Level: Depth levels cleared in a short window

---

### 2. Data Normalization
- Standardized features using **z-score**
- **PCA** was applied to reduce dimensionality (10 components)

---

### 3. Clustering
Unsupervised algorithms were applied to cluster the feature space:

| Method   | Silhouette Score ↑ | DB Index ↓ |
|----------|---------------------|-------------|
| KMeans   | 0.236               | 1.495       |
| HDBSCAN  | 0.387               | 0.819       |

- **KMeans** (n=4): Interpretable and simple
- **HDBSCAN**: More robust, detects noise and irregular clusters

---

### 4. Regime Labeling & Insights
Clusters were analyzed and labeled based on typical characteristics:

| Cluster | Characteristics                              | Regime Description                     |
|---------|----------------------------------------------|----------------------------------------|
| 0       | High spread, high volatility, low liquidity  | Volatile & Illiquid                    |
| 1       | Low spread, low volatility, high liquidity   | Stable & Liquid                        |
| 2       | Mean-reverting returns, moderate spread      | Mean Reverting                         |
| 3       | Trending price, decent liquidity & stability | Trending & Liquid & Stable            |

---

### 5. Visualizations

- 📈 **Regime Evolution** over time  
- 💹 **Mid-Price Overlay** with regime labels  
- 🌐 **UMAP Projection** of PCA features  
- 🔁 **Regime Transition Matrix** showing switching probabilities  

---

### 6. Regime Transition Insights
Analyzed how likely the market is to switch between regimes, e.g.:

- What’s the probability that a **"Trending & Liquid"** state follows **"Volatile & Illiquid"**?

This provides insights for trading strategies or market monitoring.

---

## 🤖 Tech Stack
- Python (NumPy, Pandas, Scikit-learn, HDBSCAN, UMAP, Matplotlib, Seaborn)
- Jupyter Notebook for development and experimentation

---

## 🧠 Conclusion
This project shows that **unsupervised learning** on **market microstructure data** can uncover **meaningful behavioral regimes**, offering insights into market dynamics without needing labeled data.  
**HDBSCAN** provided the most reliable clusters, while **KMeans** helped label and interpret the underlying regimes.
"""
