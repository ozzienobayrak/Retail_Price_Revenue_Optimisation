# Retail_Price_Revenue_Optimisation

# Retail Price Optimization with Machine Learning

A machine learning system for **demand forecasting and revenue-maximizing price simulation** in retail environments.

This project builds a **LightGBM demand prediction model** and deploys an **interactive price simulator** that recommends optimal prices under different inventory scenarios.

The system allows users to explore how **price changes impact demand, revenue, and inventory feasibility**.

---

# Business Problem

Retailers constantly face the challenge of setting prices that maximize revenue while accounting for:

- changing demand
- inventory limitations
- heterogeneous product behavior
- store-level variation

Traditional approaches often rely on static pricing rules or simple elasticity estimates.

This project builds a **data-driven pricing simulator** that enables dynamic price exploration using machine learning demand forecasts.

---

# Project Overview

The system consists of three main components.

## 1️⃣ Demand Forecasting Model

A **LightGBM regression model** predicts product demand using:

- lagged sales features
- rolling demand statistics
- price features
- calendar variables
- categorical store and product identifiers

This model captures **non-linear demand responses to price and time dynamics**.

---

## 2️⃣ Price Simulation Engine

Using the trained demand model, the simulator evaluates multiple candidate prices.

For each price:

1. Demand is predicted
2. Inventory constraints are applied
3. Revenue is calculated

The system then selects the **revenue-maximizing price–sales combination**.

---

## 3️⃣ Interactive Pricing Tool (Streamlit)

The final application allows users to:

- select a **product**
- choose a **store**
- choose a **pricing horizon** (weekly or monthly)
- set an **inventory scenario**

The tool then simulates pricing outcomes and returns:

- optimal price recommendation
- predicted sales
- predicted revenue
- price–revenue curves

---

# Example Output

The simulator produces curves showing how price impacts revenue and sales.

**Revenue curve**

Price ↑ → Demand ↓ → Revenue changes non-linearly.

The optimal price corresponds to the **maximum of the revenue curve**.

---

# Model Performance

| Model | Relative RMSE (Revenue) |
|------|------|
| OLS baseline | 0.455 |
| LightGBM | **0.177** |

The LightGBM model significantly improves revenue prediction accuracy compared to a simple linear benchmark.

---

# Inventory Scenarios

Because inventory data is unavailable in the dataset, the simulator approximates stock constraints using **recent demand baselines**.

Inventory scenarios are defined as:

| Scenario | Inventory Level |
|------|------|
Low | 80% of recent demand |
Medium | 100% of recent demand |
High | 120% of recent demand |

This allows the simulator to evaluate pricing under different supply conditions.

---

# Repository Structure

```
retail-price-optimization
│
├── data
│   ├── simulator_df.csv
│   ├── inventory_base_week.csv
│   └── inventory_base_month.csv
│
├── models
│   └── lightgbm_model_base.joblib
│
├── notebooks
│   └── PriceSimulator.ipynb
│
├── app
│   └── streamlit_app.py
│
├── src
│   ├── simulation.py
│   ├── inventory.py
│   └── plotting.py
│
├── requirements.txt
└── README.md
```

---

# Technologies Used

- Python
- LightGBM
- Pandas
- NumPy
- Matplotlib
- Scikit-learn
- Streamlit

---

# Key Features

- demand forecasting with machine learning
- price simulation engine
- revenue optimization
- inventory-constrained demand modeling
- weekly and monthly revenue evaluation
- interactive pricing tool

---

# Running the Project

## Install dependencies

```
pip install -r requirements.txt
```

## Run the Streamlit app

```
streamlit run app/streamlit_app.py
```

---

# Future Improvements

Possible extensions include:

- cross-price elasticity modeling
- competitor price integration
- promotion and discount simulation
- reinforcement learning pricing strategies
- store-level revenue optimization

---

# Dataset

This project uses the **Walmart Sales Forecasting dataset**, which contains historical sales data across products, stores, and time.

---

# Author

Özlem Albayrak  
Data Scientist / Economist

---

# Project Goal

The goal of this project is to demonstrate how **machine learning and simulation techniques can support real-world pricing decisions in retail environments**.