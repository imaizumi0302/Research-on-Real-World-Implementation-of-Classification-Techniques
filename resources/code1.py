import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight

# Load the dataset
df = pd.read_csv('default_credit_card_clients.csv')

# Display basic information
print(f"Dataset shape: {df.shape}")
print(f"\nFirst few rows:")
print(df.head())

print(f"\nData types:")
print(df.dtypes)

print(f"\nBasic statistics:")
print(df.describe())

# Check target variable distribution
print(f"\nTarget variable (Y) distribution:")
print(df['Y'].value_counts())
print(f"Default rate: {df['Y'].mean():.1%}")

# Check for missing values
print("Missing values per column:")
missing_values = df.isnull().sum()
print(missing_values[missing_values > 0])

# Calculate percentage of missing values
missing_percentage = (df.isnull().sum() / len(df)) * 100
print(f"\nPercentage of missing values:")
print(missing_percentage[missing_percentage > 0])

# Handle missing values - numerical columns
numerical_cols = df.select_dtypes(include=[np.number]).columns
for col in numerical_cols:
    if df[col].isnull().sum() > 0:
        df[col].fillna(df[col].median(), inplace=True)
        print(f"Filled {col} with median: {df[col].median()}")

# Handle missing values - categorical columns
categorical_cols = ['SEX', 'EDUCATION', 'MARRIAGE']
for col in categorical_cols:
    if col in df.columns and df[col].isnull().sum() > 0:
        df[col].fillna(df[col].mode()[0], inplace=True)
        print(f"Filled {col} with mode: {df[col].mode()[0]}")

# Verify no missing values remain
print(f"\nRemaining missing values: {df.isnull().sum().sum()}")


# Function to detect outliers using IQR method
def detect_outliers_iqr(data, column):
    Q1 = data[column].quantile(0.25)
    Q3 = data[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    outliers = data[(data[column] < lower_bound) | (data[column] > upper_bound)]
    return len(outliers), lower_bound, upper_bound

# Check outliers in key columns
numerical_features = ['LIMIT_BAL', 'AGE', 'BILL_AMT1', 'BILL_AMT2', 
                      'BILL_AMT3', 'BILL_AMT4', 'BILL_AMT5', 'BILL_AMT6',
                      'PAY_AMT1', 'PAY_AMT2', 'PAY_AMT3', 'PAY_AMT4', 
                      'PAY_AMT5', 'PAY_AMT6']

print("Outlier detection:")
for col in numerical_features:
    n_outliers, lower, upper = detect_outliers_iqr(df, col)
    print(f"{col}: {n_outliers} outliers ({n_outliers/len(df)*100:.2f}%)")

# Cap outliers at 1st and 99th percentile for bill and payment amounts
bill_payment_cols = ['BILL_AMT1', 'BILL_AMT2', 'BILL_AMT3', 'BILL_AMT4', 
                     'BILL_AMT5', 'BILL_AMT6', 'PAY_AMT1', 'PAY_AMT2', 
                     'PAY_AMT3', 'PAY_AMT4', 'PAY_AMT5', 'PAY_AMT6']

for col in bill_payment_cols:
    lower_bound = df[col].quantile(0.01)
    upper_bound = df[col].quantile(0.99)
    df[col] = df[col].clip(lower=lower_bound, upper=upper_bound)
    print(f"Capped {col} at [{lower_bound:.2f}, {upper_bound:.2f}]")

# Remove clearly erroneous values (e.g., invalid ages)
print(f"\nRecords before age filtering: {len(df)}")
df = df[(df['AGE'] >= 18) & (df['AGE'] <= 100)]
print(f"Records after age filtering: {len(df)}")





# Check current distribution of categorical variables
print("EDUCATION distribution (before cleaning):")
print(df['EDUCATION'].value_counts().sort_index())

print("\nMARRIAGE distribution (before cleaning):")
print(df['MARRIAGE'].value_counts().sort_index())

print("\nSEX distribution:")
print(df['SEX'].value_counts().sort_index())

# Clean EDUCATION: recode 0, 5, 6 as 4 (others)
df['EDUCATION'] = df['EDUCATION'].replace({0: 4, 5: 4, 6: 4})
print("\nEDUCATION distribution (after cleaning):")
print(df['EDUCATION'].value_counts().sort_index())

# Clean MARRIAGE: recode 0 as 3 (others)
df['MARRIAGE'] = df['MARRIAGE'].replace({0: 3})
print("\nMARRIAGE distribution (after cleaning):")
print(df['MARRIAGE'].value_counts().sort_index())

# Verify SEX values
print(f"\nSEX unique values: {sorted(df['SEX'].unique())}")
assert df['SEX'].isin([1, 2]).all(), "Invalid SEX values found"
print("✓ SEX values are valid (1 or 2)")

# Validate payment status variables
pay_cols = ['PAY_0', 'PAY_2', 'PAY_3', 'PAY_4', 'PAY_5', 'PAY_6']
print("\nPayment status ranges:")
for col in pay_cols:
    print(f"{col}: min={df[col].min()}, max={df[col].max()}, unique values={len(df[col].unique())}")





# Create Gender-Marriage combined feature
def create_gender_marriage_category(row):
    """
    Creates combined gender-marriage category
    1: Married man
    2: Single man
    3: Divorced man
    4: Married woman
    5: Single woman
    6: Divorced woman
    """
    sex = row['SEX']
    marriage = row['MARRIAGE']
    
    if sex == 1:  # Male
        if marriage == 1:
            return 1  # Married man
        elif marriage == 2:
            return 2  # Single man
        else:  # marriage == 3 (others/divorced)
            return 3  # Divorced man
    else:  # Female (sex == 2)
        if marriage == 1:
            return 4  # Married woman
        elif marriage == 2:
            return 5  # Single woman
        else:  # marriage == 3 (others/divorced)
            return 6  # Divorced woman

df['GENDER_MARRIAGE'] = df.apply(create_gender_marriage_category, axis=1)

print("Gender-Marriage category distribution:")
print(df['GENDER_MARRIAGE'].value_counts().sort_index())
print(f"\nCategory labels:")
print("1: Married man")
print("2: Single man")
print("3: Divorced man")
print("4: Married woman")
print("5: Single woman")
print("6: Divorced woman")





# Count divorced women before exclusion
divorced_women_count = (df['GENDER_MARRIAGE'] == 6).sum()
print(f"\nDivorced women records: {divorced_women_count}")
print(f"Dataset size before exclusion: {len(df)}")

# Exclude divorced women (category 6)
df = df[df['GENDER_MARRIAGE'] != 6].copy()

print(f"Dataset size after exclusion: {len(df)}")
print(f"Records removed: {divorced_women_count}")

print("\nRemaining Gender-Marriage categories:")
print(df['GENDER_MARRIAGE'].value_counts().sort_index())






# NOTE: These additional features are NOT mentioned in the paper
# They are optional enhancements you can experiment with

# Average bill amount over 6 months
bill_cols = ['BILL_AMT1', 'BILL_AMT2', 'BILL_AMT3', 
             'BILL_AMT4', 'BILL_AMT5', 'BILL_AMT6']
df['AVG_BILL_AMT'] = df[bill_cols].mean(axis=1)

# Average payment amount over 6 months
pay_amt_cols = ['PAY_AMT1', 'PAY_AMT2', 'PAY_AMT3', 
                'PAY_AMT4', 'PAY_AMT5', 'PAY_AMT6']
df['AVG_PAY_AMT'] = df[pay_amt_cols].mean(axis=1)

# Payment to bill ratio
df['PAY_TO_BILL_RATIO'] = df['AVG_PAY_AMT'] / (df['AVG_BILL_AMT'] + 1)

# Total number of months with delayed payment
pay_status_cols = ['PAY_0', 'PAY_2', 'PAY_3', 'PAY_4', 'PAY_5', 'PAY_6']
df['TOTAL_DELAYED_MONTHS'] = (df[pay_status_cols] > 0).sum(axis=1)

# Maximum payment delay across all months
df['MAX_PAYMENT_DELAY'] = df[pay_status_cols].max(axis=1)

print("\nOptional engineered features created:")
print(f"AVG_BILL_AMT: {df['AVG_BILL_AMT'].describe()}")
print(f"AVG_PAY_AMT: {df['AVG_PAY_AMT'].describe()}")
print(f"PAY_TO_BILL_RATIO: {df['PAY_TO_BILL_RATIO'].describe()}")
print(f"TOTAL_DELAYED_MONTHS: {df['TOTAL_DELAYED_MONTHS'].describe()}")
print(f"MAX_PAYMENT_DELAY: {df['MAX_PAYMENT_DELAY'].describe()}")






# Remove ID column if it exists
if 'ID' in df.columns:
    df = df.drop('ID', axis=1)
    print("ID column removed")

# Define target variable
# Note: Paper says 0=default, 1=not default, but this might be reversed in some datasets
# Verify your dataset's encoding
target_col = 'Y'  # or 'default_payment' depending on your column name

print(f"\nTarget variable: {target_col}")
print("Distribution:")
print(df[target_col].value_counts())
print(f"Default rate: {df[target_col].mean():.1%}")

# Separate features and target
y = df[target_col].copy()

# Define feature columns (all except target)
# Option 1: Use all original features + GENDER_MARRIAGE (as per paper)
feature_cols = ['LIMIT_BAL', 'SEX', 'EDUCATION', 'MARRIAGE', 'AGE',
                'PAY_0', 'PAY_2', 'PAY_3', 'PAY_4', 'PAY_5', 'PAY_6',
                'BILL_AMT1', 'BILL_AMT2', 'BILL_AMT3', 'BILL_AMT4', 
                'BILL_AMT5', 'BILL_AMT6',
                'PAY_AMT1', 'PAY_AMT2', 'PAY_AMT3', 'PAY_AMT4', 
                'PAY_AMT5', 'PAY_AMT6',
                'GENDER_MARRIAGE']

# Option 2: Include optional engineered features (NOT in paper)
# feature_cols = feature_cols + ['AVG_BILL_AMT', 'AVG_PAY_AMT', 
#                                 'PAY_TO_BILL_RATIO', 'TOTAL_DELAYED_MONTHS',
#                                 'MAX_PAYMENT_DELAY']

X = df[feature_cols].copy()

print(f"\nFeature matrix shape: {X.shape}")
print(f"Target vector shape: {y.shape}")
print(f"\nFeatures being used: {list(X.columns)}")

# Verify no data leakage
assert target_col not in X.columns, "Target variable found in features!"
print("\n✓ No data leakage detected")

# Check for missing values
print(f"\nMissing values in features: {X.isnull().sum().sum()}")
print(f"Missing values in target: {y.isnull().sum()}")






from sklearn.model_selection import train_test_split

# Train-test split with stratification
# Paper doesn't specify ratio, but 70-30 or 80-20 are common
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.3,        # 30% for testing (70-30 split)
    random_state=42,      # For reproducibility
    stratify=y            # Maintain class distribution in both sets
)

print("=" * 50)
print("TRAIN-TEST SPLIT RESULTS")
print("=" * 50)

print(f"\nTraining set size: {X_train.shape[0]} ({X_train.shape[0]/len(X)*100:.1f}%)")
print(f"Test set size: {X_test.shape[0]} ({X_test.shape[0]/len(X)*100:.1f}%)")

print(f"\nTraining set default rate: {y_train.mean():.3%}")
print(f"Test set default rate: {y_test.mean():.3%}")

# Verify stratification worked
print("\n" + "-" * 50)
print("Class Distribution Comparison")
print("-" * 50)
print("\nOriginal dataset:")
print(y.value_counts(normalize=True).sort_index())
print("\nTraining set:")
print(y_train.value_counts(normalize=True).sort_index())
print("\nTest set:")
print(y_test.value_counts(normalize=True).sort_index())

# Check if distributions are similar (difference should be < 1%)
train_default_rate = y_train.mean()
test_default_rate = y_test.mean()
overall_default_rate = y.mean()

diff_train = abs(train_default_rate - overall_default_rate)
diff_test = abs(test_default_rate - overall_default_rate)

if diff_train < 0.01 and diff_test < 0.01:
    print("\n✓ Stratification successful - distributions are similar")
else:
    print("\n⚠ Warning: Distributions differ more than expected")








from sklearn.utils.class_weight import compute_class_weight

# Calculate class weights
classes = np.unique(y_train)
class_weights = compute_class_weight(
    class_weight='balanced',
    classes=classes,
    y=y_train
)

# Create dictionary for class weights
class_weight_dict = dict(zip(classes, class_weights))

print("=" * 50)
print("CLASS WEIGHTS")
print("=" * 50)
print(f"\nClass weights: {class_weight_dict}")

print(f"\nClass 0 (non-default) weight: {class_weight_dict[0]:.4f}")
print(f"Class 1 (default) weight: {class_weight_dict[1]:.4f}")

print(f"\nInterpretation:")
print(f"- Minority class (default) has weight {class_weight_dict[1]:.2f}x higher than 1.0")
print(f"- This compensates for the {y_train.mean():.1%} default rate imbalance")

# Calculate what the weights mean
n_samples = len(y_train)
n_classes = len(classes)
print(f"\nFormula: n_samples / (n_classes * n_samples_for_class)")
for cls in classes:
    n_cls = (y_train == cls).sum()
    weight = n_samples / (n_classes * n_cls)
    print(f"Class {cls}: {n_samples} / ({n_classes} * {n_cls}) = {weight:.4f}")






def verify_data_quality(X_train, X_test, y_train, y_test):
    """
    Comprehensive verification of processed data
    """
    print("=" * 50)
    print("DATA QUALITY VERIFICATION")
    print("=" * 50)
    
    checks_passed = True
    
    # Check 1: No missing values in features
    train_missing = X_train.isnull().sum().sum()
    test_missing = X_test.isnull().sum().sum()
    if train_missing > 0 or test_missing > 0:
        print(f"\n❌ Missing values found - Train: {train_missing}, Test: {test_missing}")
        checks_passed = False
    else:
        print("\n✓ No missing values in features")
    
    # Check 2: No missing targets
    if y_train.isnull().sum() > 0 or y_test.isnull().sum() > 0:
        print("❌ Missing values found in target")
        checks_passed = False
    else:
        print("✓ No missing values in target")
    
    # Check 3: Consistent shapes
    if len(X_train) != len(y_train):
        print(f"❌ Inconsistent shapes - X_train: {len(X_train)}, y_train: {len(y_train)}")
        checks_passed = False
    elif len(X_test) != len(y_test):
        print(f"❌ Inconsistent shapes - X_test: {len(X_test)}, y_test: {len(y_test)}")
        checks_passed = False
    else:
        print("✓ Consistent shapes between X and y")
    
    # Check 4: Same features in train and test
    if not X_train.columns.equals(X_test.columns):
        print("❌ Different features in train and test sets")
        checks_passed = False
    else:
        print(f"✓ Same {len(X_train.columns)} features in train and test sets")
    
    # Check 5: Binary target
    train_unique = set(y_train.unique())
    test_unique = set(y_test.unique())
    if not train_unique.issubset({0, 1}) or not test_unique.issubset({0, 1}):
        print(f"❌ Target is not binary - Train: {train_unique}, Test: {test_unique}")
        checks_passed = False
    else:
        print("✓ Target is binary (0, 1)")
    
    # Check 6: Reasonable class balance
    train_default_rate = y_train.mean()
    test_default_rate = y_test.mean()
    if abs(train_default_rate - test_default_rate) > 0.05:
        print(f"⚠️  Warning: Significant difference in default rates")
        print(f"   Train: {train_default_rate:.1%}, Test: {test_default_rate:.1%}")
    else:
        print(f"✓ Consistent default rates (Train: {train_default_rate:.1%}, Test: {test_default_rate:.1%})")
    
    # Check 7: No data leakage
    if 'Y' in X_train.columns or 'default_payment' in X_train.columns:
        print("❌ Target variable found in features (data leakage)")
        checks_passed = False
    else:
        print("✓ No data leakage detected")
    
    # Check 8: Numerical data types
    non_numeric_train = X_train.select_dtypes(exclude=[np.number]).columns
    non_numeric_test = X_test.select_dtypes(exclude=[np.number]).columns
    if len(non_numeric_train) > 0 or len(non_numeric_test) > 0:
        print(f"❌ Non-numerical features found: {list(non_numeric_train)}")
        checks_passed = False
    else:
        print("✓ All features are numerical")
    
    print("\n" + "=" * 50)
    if checks_passed:
        print("✓✓✓ ALL CHECKS PASSED - READY FOR MODEL TRAINING ✓✓✓")
    else:
        print("❌❌❌ SOME CHECKS FAILED - PLEASE REVIEW ❌❌❌")
    print("=" * 50)
    
    return checks_passed

# Run verification
all_checks_passed = verify_data_quality(X_train, X_test, y_train, y_test)

# Save processed data (optional)
print("\n" + "=" * 50)
print("SAVING PROCESSED DATA")
print("=" * 50)

X_train.to_csv('X_train.csv', index=False)
X_test.to_csv('X_test.csv', index=False)
y_train.to_csv('y_train.csv', index=False)
y_test.to_csv('y_test.csv', index=False)

print("\n✓ Processed data saved:")
print("  - X_train.csv")
print("  - X_test.csv")
print("  - y_train.csv")
print("  - y_test.csv")

# Save class weights for later use
import json
with open('class_weights.json', 'w') as f:
    # Convert numpy types to Python types for JSON serialization
    class_weight_dict_json = {int(k): float(v) for k, v in class_weight_dict.items()}
    json.dump(class_weight_dict_json, f, indent=4)

print("  - class_weights.json")

# Create and save summary
summary = {
    'total_records': len(df),
    'training_records': len(X_train),
    'test_records': len(X_test),
    'num_features': X_train.shape[1],
    'features': list(X_train.columns),
    'overall_default_rate': float(y.mean()),
    'train_default_rate': float(y_train.mean()),
    'test_default_rate': float(y_test.mean()),
    'class_weights': class_weight_dict_json,
    'train_test_split_ratio': f"{len(X_train)}/{len(X_test)} ({100*len(X_train)/len(df):.1f}%/{100*len(X_test)/len(df):.1f}%)"
}

with open('preprocessing_summary.json', 'w') as f:
    json.dump(summary, f, indent=4)

print("  - preprocessing_summary.json")

print("\n" + "=" * 50)
print("PREPROCESSING SUMMARY")
print("=" * 50)
for key, value in summary.items():
    if key != 'features':  # Skip printing all feature names
        print(f"{key}: {value}")