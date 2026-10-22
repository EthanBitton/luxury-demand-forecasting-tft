# Weekly Demand Forecasting for a Luxury Group with Temporal Fusion Transformers

> CentraleSupélec · AI track · Client: **Iris by Argon & Co** (data & AI team of a consulting firm), on behalf of a global luxury house · Sep 2023 – Jan 2024 · Team of 5

**TL;DR**: We forecast weekly sales per product and per country for a luxury group. The aim was to test whether an interpretable deep-learning model (Temporal Fusion Transformer) could outperform the client's decision-tree approach at weekly granularity. On the European scope, forecast accuracy (1 − WAPE) went from **~55% for a naive baseline to 85–90%**. The model outputs quantile forecasts, and a Power BI dashboard presents them to business users.

> Code and data are not published: the client's sales data is confidential.

---

## Business problem

Luxury houses must avoid two costly failures at once:
- **stock-outs**, because customers expect immediate availability;
- **over-production** of items that will not sell.

The client's existing tree-based model worked at a monthly level. It struggled in three situations: weekly granularity, forecasting several horizons at once, and explaining *why* a forecast was made.

**Question:** can a Temporal Fusion Transformer forecast weekly demand per SKU × country accurately, and explain its predictions?

## Data

- **~7.5M rows** of weekly sales over ~3 years: SKU × country × week, with quantity, revenue and product family
- **Product master** with 50+ attributes per SKU: target gender, product family, seasonality, aesthetic line, etc.
- The data was messy: infinite and negative prices, very sparse or abruptly discontinued SKUs, and missing weeks

## Approach

```mermaid
flowchart LR
    A["Raw weekly sales<br/>7.5M rows"] --> B["Cleaning & filtering<br/>sparse SKUs, invalid prices"]
    B --> C["Zero-filling missing weeks<br/>from each SKU's launch date per country"]
    C --> D["Feature engineering<br/>cyclical week, price, product attributes"]
    D --> E["Temporal Fusion Transformer<br/>PyTorch Forecasting & Darts"]
    E --> F["Quantile forecasts<br/>P10 / P50 / P90"]
    F --> G["Power BI dashboard"]
```

1. **Data cleaning.** We set filtering rules on history length and regularity: at least 104 weekly observations and at least 33 sales per year. Without them, sparse SKUs pulled all forecasts towards zero. Missing weeks were zero-filled from each SKU's launch date in each country, a choice validated with the client. A generic pandas fill was not enough, because every SKU × country pair has its own launch date.
2. **Metric choice.** We used **WAPE instead of MAPE**. MAPE explodes on low-volume weeks, which are frequent in luxury. For example, predicting 2 units when 1 sold counts as a 100% error. We defined accuracy as 1 − WAPE.
3. **Modeling.** We used a Temporal Fusion Transformer ([Lim et al., 2020](https://arxiv.org/abs/1912.09363)) with a quantile loss, implemented first in PyTorch Forecasting and then in Darts for stability. Runs were tracked in TensorBoard.
4. **Iterations.** Each step was measured against the baseline:
   - longer history and stricter filtering;
   - cyclical encoding of the week of the year, |sin(2π·week/104)|, so the model sees that late December is close to early January;
   - more attention heads and more countries;
   - price as a covariate, plus static product covariates;
   - a regional training scope.
5. **Interpretability.** Attention weights over time and variable-selection weights show what drives each forecast. The main static drivers were the SKU, its sales scale, product family, target gender and segment.

## Results

| Stage | Accuracy (1 − WAPE) |
|---|---|
| Naive baseline (last observed value) | ~55% |
| First TFT (default hyper-parameters, ~6 months of history) | 50–60% |
| + longer history, stricter filtering | 60–68% |
| + cyclical week, more countries and attention heads | ~60% (but now captures sales peaks instead of a flat line) |
| + price and product covariates, single country (US) | 67–72% |
| **Final models, European scope (PyTorch Forecasting & Darts)** | **85–90%** |

**Key insights**
- **Price was by far the most impactful feature.** Adding it produced the largest accuracy jump in both implementations.
- **A culturally homogeneous region beats a global model.** Training on Europe captures shared seasonality, such as Christmas, better than a worldwide model. Single-country models were even more accurate but less general.
- **Domain-driven features mattered more than tuning.** Default settings gave smooth, unusable forecasts. We considered hyper-parameter search with Optuna, but dropped it because of compute cost and overfitting risk on our setup.

## Limitations & what I would do differently

- **Evaluation protocol.** The final checkpoint was chosen on forecast accuracy rather than on the loss. Today I would use a strict time-based train / validation / test split, with a rolling-origin backtest, so the reported accuracy is not optimistic.
- **Benchmark.** The head-to-head comparison with the client's tree model was planned but not completed. I would add it, along with a gradient-boosting baseline such as LightGBM, on the same backtest.
- **Tuning.** Proper hyper-parameter search (Optuna) would only make sense once a clean validation set is in place.

## My contribution

- Data exploration, preprocessing and data augmentation (filtering rules, zero-filling pipeline)
- Hyper-parameter experiments and exploration of a daily-frequency model
- **Power BI dashboard**: actual vs. predicted sales with quantile bands, filterable by country, product and year
- Market-trend analysis, client report and final presentation

## Stack

`Python` · `pandas` · `PyTorch` · `PyTorch Lightning` · `PyTorch Forecasting` · `Darts` · `TensorBoard` · `Power BI` · `Git`

## References

- B. Lim, S. Ö. Arık, N. Loeff, T. Pfister. *Temporal Fusion Transformers for Interpretable Multi-horizon Time Series Forecasting*, 2020.
