Heart Disease Dataset: Data Engineering and Analysis
1. Dataset Overview 🗃️
The dataset used is a structured CSV file containing patient heart health indicators such as:
- Age, sex, chest pain type, resting blood pressure (trestbps)
- Cholesterol level (chol), fasting blood sugar, ECG results
- Maximum heart rate achieved (thalach), exercise-induced angina (exang)
- ST depression (oldpeak), and target (presence of heart disease)
2. ECT: Extraction, Cleaning, Transformation ⚙️
Data Cleaning 🧼
- Removed unnecessary/duplicate entries.
- Checked and handled missing/null values.
- Converted categorical values to human-readable labels (e.g., sex: 0 → Female, 1 → Male).
Data Transformation 🔁
- Binning and categorization:
  - age_group (Young, Middle-aged, Senior)
  - age_decade (20s, 30s, ..., 70s)
  - cholesterol levels (Desirable, Borderline High, High)
  - blood pressure levels (Normal, Elevated, Hypertension Stages)
  - oldpeak_category (low, moderate, high)
- Type conversions for binary classification (e.g., high_cholesterol, high_bp).
3. Feature Engineering 🧠
New features engineered to enrich analysis and modeling:
- age_group: Categorical age bins
- age_decade: Decade-wise age
- chol_per_age: Cholesterol normalized by age
- risk_score: Weighted combination of risk factors
- exercise_recovery: Recovery based on thalach and exang
- age_oldpeak: Interaction of age and oldpeak
- high_cholesterol, high_bp: Flags for elevated conditions
- heart_rate_reserve: 220 - age - thalach
- max_hr_percent: thalach / (220 - age)
- hr_efficiency: heart_rate_reserve / (oldpeak + 1)
- combined_risk_flag: True if multiple high-risk conditions
- sex_str: Human-readable sex label
4. Statistical Summary & Charts 📊
- Descriptive statistics (mean, median, mode, 25%, 50%, 70%, max) were exported.
- Charts were created to visualize distributions and compare metrics.
5. Visualizations 📈
- Summary Bar Chart (mean, median, mode, percentiles)
- Pie Charts:
  - Sex distribution
  - Age group distribution
- Optional: Correlation heatmap and Streamlit dashboard
6. Output Directory 🧪
All outputs are stored in the dw/ directory:
- heart_engineered_features_extended_v2.csv
- heart_summary_stats_extended.csv
- pie_sex_distribution.png
- pie_age_group_distribution.png
- plot_mean_median_mode_v2.png
7. Technologies Used 🚀
- Python 3
- pandas, numpy
- matplotlib, seaborn
- os (file handling)
8. Next Steps 💡
- Deploy interactive Streamlit dashboard
- Push data to cloud DW (e.g., Snowflake, BigQuery)
- Train ML models with engineered features
