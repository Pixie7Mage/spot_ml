# Spotter Machine Learning Engineer Assessment: Loom Video Script

**Duration:** 2 minutes 30 seconds – 2 minutes 50 seconds  
**Speaker:** Machine Learning Engineer Candidate  
**Objective:** Concise, high-signal walkthrough of data findings, data quality resolutions, validation strategy, modeling decisions, and codebase architecture.

---

## Visual Preparation Checklist Before Hitting Record
1. **Screen 1 (Left Half):** Open VS Code / IDE with `run_pipeline.py`, `src/feature_engineering.py`, and `src/train.py`.
2. **Screen 2 (Right Half):** Open `Spotter_ML_Assessment_Report.pdf` displaying **Table 1** (Performance Benchmarks) and **Figure 1** (`candidate_december.png`).
3. **Terminal:** Show terminal output verifying that `score.py` passed with 0 errors.

---

## Script & Timestamp Walkthrough

### 0:00 – 0:30 | Introduction & Key Data Findings
> **[On Camera + Show Report Title & Executive Summary]**  
> *"Hi everyone, I’m excited to present my solution for the Spotter Freight Rate Prediction Assessment.*  
> 
> *Our goal is to accurately predict spot freight rates across 12,000 forward evaluation loads in November and December 2025, and generate daily predicted rates for a fixed benchmark lane in December.*  
> 
> *When exploring the 48,000 development loads, three key findings stood out:*  
> 1. *First, `distance * quote_signal` serves as a natural baseline for unadjusted lane pricing, but leaves significant error unmodeled (baseline MAE is $246).*  
> 2. *Second, freight rates and the macroeconomic `market_index` exhibit a pronounced 7-day cyclicality—troughing on Sundays and peaking on Wednesdays and Thursdays when spot capacity tightens.*  
> 3. *Third, trailer equipment types—specifically Reefer and Flatbed—command systematic rate premiums over standard Dry Vans due to temperature control and specialized handling."*

---

### 0:30 – 1:00 | Data Quality Issues Identified & Resolutions
> **[Switch to Screen showing Section 2 of PDF Report / `src/data_cleaning.py`]**  
> *"During data auditing, I identified and resolved three critical data quality issues:*  
> - *First, **Negative Cargo Weights**: 292 rows in train and 145 in validation had negative weights (such as -36,000 lbs). In freight, cargo weight cannot be negative. This was an erroneous sign flip, which I cleanly resolved using `abs(weight)`.*  
> - *Second, **Missing Weights**: Missing values were imputed using equipment-stratified medians to preserve trailer weight physics.*  
> - *Third, **Missing Market Index**: Because `market_index` reflects daily macroeconomic market conditions, I imputed missing values using the cross-lane daily mean for that specific date.*  
> - *Finally, I audited origin and destination nodes and confirmed a 1:1 deterministic mapping to GPS centroids, allowing us to accurately compute spatial routing features."*

---

### 1:00 – 1:40 | Validation Strategy & Train/Test Split Rationale
> **[Highlight Section 3 & Table 1 in PDF Report]**  
> *"Next, the validation strategy. In freight market pricing, standard random K-fold cross-validation is fundamentally flawed because it causes temporal lookahead leakage—predicting past rates using future market signals.*  
> 
> *To mirror real-world deployment where our model must predict forward quarters (November–December), I established an **Out-of-Time Temporal Split**:*  
> - *Training on Months 1 through 8 (January to August, ~38,500 loads).*  
> - *Validating on Months 9 and 10 (September and October, ~9,500 loads).*  
> 
> *This allowed us to evaluate true forward-in-time generalization.*  
> 
> *For our final production model, we used a **5-fold cross-validated bagged ensemble** across all 48,000 labeled development loads. Bagging across the 5 folds eliminates model variance and ensures maximum stability on unseen test loads."*

---

### 1:40 – 2:15 | Model Reasoning, Feature Engineering & Results
> **[Show `src/feature_engineering.py` and `src/train.py`]**  
> *"For feature engineering, we translated logistics domain principles into high-impact features:*  
> - *Route circuity by comparing highway distance to straight-line Haversine distance.*  
> - *Freight physics via ton-miles `(weight / 2000 * distance)` and weight-per-mile.*  
> - *Pricing signals combining distance, quote signal, and daily market index.*  
> - *Cyclical $\sin/\cos$ calendar encodings for weekly and annual seasonality.*  
> 
> *For model selection, while MSE loss was overly sensitive to rate outliers, formulating **Gradient Boosted Decision Trees (LightGBM and CatBoost) with an L1 MAE objective** delivered superior results.*  
> 
> *As shown in Table 1, our blended ensemble:*  
> - *Slashed MAE from **$246.02** down to **$117.76**—a **50.1% error reduction** over baseline.*  
> - *Reduced MAPE to **4.99%**.*  
> - *Achieved an **R² of 0.8256** on out-of-time evaluation."*

---

### 2:15 – 2:45 | Code Walkthrough & December Prediction Chart
> **[Show Figure 1 in Report (`candidate_december.png`) and Terminal with `score.py` output]**  
> *"Looking at Figure 1, here is our December prediction chart produced by `score.py` for Lexington to Fort Wayne (Dry Van, 360 miles, 32,000 lbs).*  
> 
> *The curve captures realistic logistics dynamics:*  
> - *The weekly peak-to-trough cycle with Sunday dips (~$780) and mid-week surges (~$803).*  
> - *A steady upward drift into late December, reflecting holiday capacity tightening between Christmas and New Year.*  
> - *The mean rate of $827 ($2.30/mile) strictly aligns with historical Dry Van pricing on this lane.*  
> 
> *Our solution is fully modular in `src/`, with clean separation of data cleaning, feature engineering, training, and inference. Running `python run_pipeline.py` executes the entire pipeline and validates cleanly against `score.py` with zero errors.*  
> 
> *Thank you, and I look forward to discussing the system further!"*

---

## Presenter Tips
- **Pacing:** Speak at a natural, steady pace. Avoid rushing; pause briefly when transitioning between slides.
- **Tone:** Confident, structured, and focused on domain mechanics (logistics, seasonality, and validation integrity).
