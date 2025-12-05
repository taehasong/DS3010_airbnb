# Airbnb Price Prediction: EDA Checkpoint
# Author: Team 3
# Due: December 2, 2025
# FINAL VERSION - Improved Visualizations

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
import warnings
warnings.filterwarnings('ignore')

# Set style for better visualizations
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (16, 12)
plt.rcParams['font.size'] = 9

# LOAD & INSPECT DATA

df = pd.read_csv('Airbnb_Data.csv')

print("="*70)
print("DATASET OVERVIEW")
print("="*70)
print(f"Shape: {df.shape[0]} rows, {df.shape[1]} columns")
print(f"\nColumn Names:\n{df.columns.tolist()}")

# DESCRIPTIVE STATISTICS & MISSING DATA

print("\n" + "="*70)
print("DESCRIPTIVE STATISTICS (NUMERICAL FEATURES)")
print("="*70)
print(df.describe().round(2))

print("\n" + "="*70)
print("MISSING DATA ANALYSIS")
print("="*70)
missing = pd.DataFrame({
    'Feature': df.columns,
    'Missing Count': df.isnull().sum(),
    'Missing %': (df.isnull().sum() / len(df) * 100).round(2)
}).sort_values('Missing %', ascending=False)
print(missing[missing['Missing %'] > 0])

# SECTION 3: DATA CLEANING

print("\n" + "="*70)
print("DATA CLEANING STEPS")
print("="*70)

print(f"Before cleaning: {len(df)} rows")

# Handle missing values
print("\nStep 1: Handle Missing Values")
numeric_cols = df.select_dtypes(include=[np.number]).columns
for col in numeric_cols:
    missing_count = df[col].isnull().sum()
    if missing_count > 0:
        median_val = df[col].median()
        df[col].fillna(median_val, inplace=True)
        print(f"  - {col}: {missing_count} values imputed with median {median_val:.2f}")

for col in df.select_dtypes(include=['object']).columns:
    missing_count = df[col].isnull().sum()
    if missing_count > 0:
        df[col].fillna('Unknown', inplace=True)
        print(f"  - {col}: {missing_count} values filled with 'Unknown'")

# Remove outliers
print("\nStep 2: Remove Price Outliers")
if 'log_price' in df.columns:
    q1 = df['log_price'].quantile(0.01)
    q99 = df['log_price'].quantile(0.99)
    initial_rows = len(df)
    df = df[(df['log_price'] >= q1) & (df['log_price'] <= q99)]
    rows_removed = initial_rows - len(df)
    print(f"  - Removed {rows_removed} rows ({rows_removed/initial_rows*100:.1f}%)")

# Feature engineering
print("\nStep 3: Feature Engineering")
if 'amenities' in df.columns:
    df['amenities_count'] = df['amenities'].apply(lambda x: len(str(x).split(',')) if pd.notna(x) else 0)
    print(f"  - Created amenities_count feature")

if 'log_price' in df.columns and 'price' not in df.columns:
    df['price'] = np.expm1(df['log_price'])
    print(f"  - Created price column from log_price")

print(f"\nAfter cleaning: {len(df)} rows")

# SECTION 4: EXPLORATORY VISUALIZATIONS

print("\n" + "="*70)
print("CREATING VISUALIZATIONS FOR CHECKPOINT")
print("="*70)

fig = plt.figure(figsize=(16, 12))
price_col = 'log_price' if 'log_price' in df.columns else 'price'

# VIZ 1: Price Distribution
ax1 = plt.subplot(2, 3, 1)
ax1.hist(df[price_col], bins=50, color='steelblue', alpha=0.7, edgecolor='black')
ax1.set_xlabel(f'{price_col.title()} ($)')
ax1.set_ylabel('Number of Listings')
ax1.set_title(f'{price_col.title()} Distribution', fontweight='bold')
ax1.axvline(df[price_col].median(), color='red', linestyle='--', linewidth=2, label=f'Median: {df[price_col].median():.2f}')
ax1.legend()

# VIZ 2: Price by Bedrooms (Box Plot)
ax2 = plt.subplot(2, 3, 2)
if 'bedrooms' in df.columns:
    try:
        df_plot = df.copy()
        df_plot['bedrooms_binned'] = pd.cut(df['bedrooms'], bins=[0, 1, 2, 3, 4, 100], 
                                             labels=['Studio', '1BR', '2BR', '3BR', '4+BR'])
        sns.boxplot(data=df_plot, x='bedrooms_binned', y=price_col, ax=ax2, palette='Set2')
        ax2.set_xlabel('Number of Bedrooms')
        ax2.set_ylabel(f'{price_col.title()}')
        ax2.set_title('Price by Bedrooms', fontweight='bold')
    except Exception as e:
        ax2.text(0.5, 0.5, f'Could not create boxplot', ha='center', va='center')
else:
    ax2.text(0.5, 0.5, 'bedrooms not available', ha='center', va='center')

# VIZ 3: Feature Correlation Heatmap 
ax3 = plt.subplot(2, 3, 3)
try:
    numerical_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    # Select only key numerical features
    key_cols = ['log_price', 'bedrooms', 'bathrooms', 'accommodates', 'amenities_count', 
                'number_of_reviews', 'review_scores_rating']
    key_cols = [col for col in key_cols if col in df.columns]
    
    if len(key_cols) > 1:
        corr_matrix = df[key_cols].corr()
        sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='coolwarm', center=0, 
                   ax=ax3, cbar_kws={'label': 'Correlation'}, square=True, 
                   annot_kws={'size': 8})
        ax3.set_title('Correlation Heatmap (Key Features)', fontweight='bold')
        plt.setp(ax3.get_xticklabels(), rotation=45, ha='right', fontsize=8)
        plt.setp(ax3.get_yticklabels(), rotation=0, fontsize=8)
    else:
        ax3.text(0.5, 0.5, 'Not enough numerical features', ha='center', va='center')
except Exception as e:
    ax3.text(0.5, 0.5, f'Could not create heatmap', ha='center', va='center')

# VIZ 4: Missing Data Pattern
ax4 = plt.subplot(2, 3, 4)
missing_data = missing[missing['Missing %'] > 0].sort_values('Missing %')
if len(missing_data) > 0:
    ax4.barh(missing_data['Feature'].head(10), missing_data['Missing %'].head(10), color='coral')
    ax4.set_xlabel('% Missing')
    ax4.set_title('Missing Data Pattern', fontweight='bold')
else:
    ax4.text(0.5, 0.5, 'No missing data', ha='center', va='center')

# VIZ 5: Price by Top Neighborhoods
ax5 = plt.subplot(2, 3, 5)
if 'neighbourhood' in df.columns:
    try:
        top_neighborhoods = df['neighbourhood'].value_counts().head(8).index
        df_top = df[df['neighbourhood'].isin(top_neighborhoods)]
        neigh_prices = df_top.groupby('neighbourhood')[price_col].median().sort_values(ascending=False)
        neigh_prices.plot(kind='barh', ax=ax5, color='skyblue', edgecolor='black')
        ax5.set_xlabel(f'Median {price_col.title()}')
        ax5.set_title('Median Price by Top 8 Neighborhoods', fontweight='bold')
    except Exception as e:
        ax5.text(0.5, 0.5, f'Could not plot neighborhoods', ha='center', va='center')
else:
    ax5.text(0.5, 0.5, 'Neighborhood data not available', ha='center', va='center')

# VIZ 6: Amenities Count vs Price (Scatter)
ax6 = plt.subplot(2, 3, 6)
if 'amenities_count' in df.columns:
    try:
        ax6.scatter(df['amenities_count'], df[price_col], alpha=0.4, s=20, color='green')
        # Add trend line
        valid_mask = df[['amenities_count', price_col]].notna().all(axis=1)
        if valid_mask.sum() > 1:
            z = np.polyfit(df.loc[valid_mask, 'amenities_count'], 
                          df.loc[valid_mask, price_col], 1)
            p = np.poly1d(z)
            x_line = sorted(df['amenities_count'].dropna().unique())
            ax6.plot(x_line, p(x_line), "r--", linewidth=2, label=f'Trend: +{z[0]:.3f}/amenity')
        ax6.set_xlabel('Number of Amenities')
        ax6.set_ylabel(f'{price_col.title()}')
        ax6.set_title('Price vs Amenities Count', fontweight='bold')
        ax6.legend()
    except Exception as e:
        ax6.text(0.5, 0.5, f'Could not create scatter', ha='center', va='center')
else:
    ax6.text(0.5, 0.5, 'Amenities not available', ha='center', va='center')

plt.tight_layout()
plt.savefig('airbnb_eda_visualizations.png', dpi=300, bbox_inches='tight')
print("Saved visualization: airbnb_eda_visualizations.png")
plt.show()

# SECTION 5: KEY INSIGHTS SUMMARY

print("\n" + "="*70)
print("KEY INSIGHTS FROM EDA")
print("="*70)

print(f"\n1. PRICE DISTRIBUTION ({price_col}):")
print(f"   - Mean: {df[price_col].mean():.2f}")
print(f"   - Median: {df[price_col].median():.2f}")
print(f"   - Std Dev: {df[price_col].std():.2f}")
print(f"   - Range: {df[price_col].min():.2f} - {df[price_col].max():.2f}")

if 'bedrooms' in df.columns:
    print(f"\n2. PRICE BY BEDROOMS:")
    for br in sorted(df['bedrooms'].unique())[:6]:
        if pd.notna(br):
            subset = df[df['bedrooms'] == br]
            print(f"   - {int(br)} bedrooms: Median {subset[price_col].median():.2f} (n={len(subset)})")

if 'amenities_count' in df.columns:
    print(f"\n3. AMENITIES IMPACT:")
    for i in range(0, int(df['amenities_count'].max()), 5):
        subset = df[(df['amenities_count'] >= i) & (df['amenities_count'] < i+5)]
        if len(subset) > 0:
            print(f"   - {i}-{i+4} amenities: Median {subset[price_col].median():.2f} (n={len(subset)})")

print(f"\n4. TOP CORRELATIONS WITH {price_col.upper()}:")
numerical_features = df.select_dtypes(include=[np.number]).columns.tolist()
if price_col in numerical_features:
    try:
        correlations = df[numerical_features].corr()[price_col].sort_values(ascending=False)
        for feature, corr in correlations.head(8).items():
            if feature != price_col:
                print(f"   - {feature}: {corr:.3f}")
    except Exception as e:
        print(f"   - Could not calculate: {e}")

if 'neighbourhood' in df.columns:
    print(f"\n5. PRICE BY NEIGHBOURHOOD (Top 5):")
    top_5 = df.groupby('neighbourhood')[price_col].median().sort_values(ascending=False).head(5)
    for neigh, price in top_5.items():
        print(f"   - {neigh}: {price:.2f}")

print(f"\n6. MISSING DATA SUMMARY:")
print(missing[missing['Missing %'] > 0].to_string())

# SAVE CLEANED DATA

df.to_csv('airbnb_cleaned.csv', index=False)
print(f"\nCleaned dataset saved to airbnb_cleaned.csv ({len(df)} rows)")

print("\n" + "="*70)
print("EDA COMPLETE - Ready for PDF Checkpoint")
print("="*70)