import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Set up output directory
output_dir = "dw"
os.makedirs(output_dir, exist_ok=True)

print("📦 Loading dataset...")
df = pd.read_csv("heart_cleaned_transformed.csv")
print(f"✅ Loaded: {df.shape[0]} rows, {df.shape[1]} columns\n")

# Feature Engineering
print("🔧 Performing feature engineering...")
df['age_group'] = pd.cut(df['age'], bins=[0, 39, 59, 120], labels=['Young', 'Middle-aged', 'Senior'])
df['age_decade'] = pd.cut(df['age'], bins=np.arange(20, 90, 10), right=False, labels=['20s', '30s', '40s', '50s', '60s', '70s'])
df['chol_per_age'] = df['chol'] / df['age']
df['risk_score'] = (
    df['age'] * 0.02 +
    df['trestbps'] * 0.015 +
    df['chol'] * 0.01 +
    df['thalach'] * -0.01 +
    df['oldpeak'] * 0.03 +
    df['sex'] * 0.05 +
    df['exang'] * 0.05
)
df['exercise_recovery'] = df['thalach'] / (df['exang'] + 1)
df['age_oldpeak'] = df['age'] * df['oldpeak']
df['high_cholesterol'] = (df['chol'] > 240).astype(int)
df['high_bp'] = (df['trestbps'] > 130).astype(int)
df['heart_rate_reserve'] = 220 - df['age'] - df['thalach']
df['max_hr_percent'] = df['thalach'] / (220 - df['age'])
df['hr_efficiency'] = df['heart_rate_reserve'] / (df['oldpeak'] + 1)
df['oldpeak_category'] = pd.cut(df['oldpeak'], bins=[-0.1, 1, 2, df['oldpeak'].max()], labels=['low', 'moderate', 'high'])
df['chol_category'] = pd.cut(df['chol'], bins=[0, 200, 240, df['chol'].max()], labels=['Desirable', 'Borderline High', 'High'])
df['bp_category'] = pd.cut(df['trestbps'], bins=[0, 120, 129, 139, 200], labels=['Normal', 'Elevated', 'Hypertension Stage 1', 'Hypertension Stage 2'])
df['combined_risk_flag'] = ((df['high_cholesterol'] + df['high_bp'] + df['exang']) >= 2).astype(int)
df['sex_str'] = df['sex'].map({0: 'Female', 1: 'Male'})

# Save engineered features
engineered_file = os.path.join(output_dir, "heart_engineered_features_extended_v2.csv")
df.to_csv(engineered_file, index=False)
print(f"✅ Saved: {os.path.basename(engineered_file)}")

# Summary Statistics
print("\n📊 Generating summary statistics...")
summary = df.describe(percentiles=[.25, .5, .7, 1.0]).T
summary['median'] = df.median(numeric_only=True)
summary['mode'] = df.mode(numeric_only=True).iloc[0]

summary_file = os.path.join(output_dir, "heart_summary_stats_extended.csv")
summary.to_csv(summary_file)
print(f"✅ Saved: {os.path.basename(summary_file)}")

# Show a portion of summary in terminal
print("\n📋 Sample Summary Statistics (first 6 rows):")
print(summary[['mean', 'median', 'mode', '25%', '50%', '70%', 'max']].head(6))
print("")

# Summary Bar Plot
print("📈 Creating summary plot...")
features = ['age', 'trestbps', 'chol', 'thalach', 'oldpeak', 'risk_score']
stats = {
    'mean': df[features].mean(),
    'median': df[features].median(),
    'mode': df[features].mode().iloc[0],
    '25%': df[features].quantile(0.25),
    '70%': df[features].quantile(0.70),
    'max': df[features].max()
}

x = np.arange(len(features))
width = 0.13
plt.figure(figsize=(14, 6))
for i, (label, values) in enumerate(stats.items()):
    plt.bar(x + (i - 3) * width, values, width, label=label)

plt.xticks(x, features)
plt.ylabel("Value")
plt.title("Summary Statistics")
plt.legend()
plt.tight_layout()
plot_file = os.path.join(output_dir, "plot_mean_median_mode_v2.png")
plt.savefig(plot_file)
plt.show()
print(f"✅ Saved: {os.path.basename(plot_file)}")

# Pie Charts
print("\n🥧 Creating pie charts...")
plt.figure()
df['sex_str'].value_counts().plot.pie(autopct='%1.1f%%', title='Sex Distribution')
plt.ylabel("")
pie_sex_file = os.path.join(output_dir, "pie_sex_distribution.png")
plt.savefig(pie_sex_file)
plt.show()
print(f"✅ Saved: {os.path.basename(pie_sex_file)}")

plt.figure()
df['age_group'].value_counts().plot.pie(autopct='%1.1f%%', title='Age Group Distribution')
plt.ylabel("")
pie_age_file = os.path.join(output_dir, "pie_age_group_distribution.png")
plt.savefig(pie_age_file)
plt.show()
print(f"✅ Saved: {os.path.basename(pie_age_file)}")

print("\n✅ All tasks completed. Outputs are saved in the dw folder.")
