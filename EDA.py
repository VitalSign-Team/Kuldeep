import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# =====================================================
# VITALSIGN - HEALTHCARE READMISSION PROJECT
# EXPLORATORY DATA ANALYSIS
# =====================================================

CSV_PATH = r"C:\Users\ASUS\PycharmProjects\MainPro1\diabetes+130-us+hospitals+for+years+1999-2008\diabetic_data.csv"

OUTPUT_DIR = "static/eda_output"
os.makedirs(OUTPUT_DIR, exist_ok=True)

sns.set_style("whitegrid")

pd.set_option("display.max_columns", None)
pd.set_option("display.width", 200)


# =====================================================
# 1. LOAD DATA
# =====================================================

df = pd.read_csv(CSV_PATH)

print("=" * 70)
print("VITALSIGN - DIABETES HEALTHCARE DATASET")
print("=" * 70)

print("\nDataset Shape:")
print(df.shape)

print("\nTotal Rows:", df.shape[0])
print("Total Columns:", df.shape[1])

print("\nFirst 5 Rows:")
print(df.head())


# =====================================================
# 2. BASIC INFORMATION
# =====================================================

print("\n" + "=" * 70)
print("BASIC INFORMATION")
print("=" * 70)

print("\nData Types:")
print(df.dtypes)

print("\nDataset Information:")
df.info()

print("\nNumeric Statistics:")
print(df.describe())


# =====================================================
# 3. MISSING VALUES
# =====================================================

print("\n" + "=" * 70)
print("MISSING VALUE ANALYSIS")
print("=" * 70)

# Dataset uses ? for many missing categorical values
df_clean = df.replace("?", np.nan)

missing_count = df_clean.isnull().sum()

missing_percent = (
    missing_count / len(df_clean)
) * 100

missing_table = pd.DataFrame({
    "Missing Count": missing_count,
    "Missing Percentage": missing_percent
})

missing_table = missing_table[
    missing_table["Missing Count"] > 0
].sort_values(
    by="Missing Percentage",
    ascending=False
)

print(missing_table)

if not missing_table.empty:

    plt.figure(figsize=(12, 6))

    sns.barplot(
        x=missing_table["Missing Percentage"],
        y=missing_table.index
    )

    plt.title("Missing Values by Column")
    plt.xlabel("Missing Percentage (%)")
    plt.ylabel("Columns")

    plt.tight_layout()

    plt.savefig(
        f"{OUTPUT_DIR}/missing_values.png",
        dpi=150
    )

    plt.show()


# =====================================================
# 4. DUPLICATE ROWS
# =====================================================

print("\n" + "=" * 70)
print("DUPLICATE ANALYSIS")
print("=" * 70)

duplicate_count = df.duplicated().sum()

print("Duplicate Rows:", duplicate_count)


# =====================================================
# 5. TARGET VARIABLES
# =====================================================

print("\n" + "=" * 70)
print("TARGET VARIABLE ANALYSIS")
print("=" * 70)

print("\nReadmission Distribution:")
print(df["readmitted"].value_counts())

print("\nLength of Stay Distribution:")
print(
    df["time_in_hospital"]
    .value_counts()
    .sort_index()
)


# Readmission Distribution

plt.figure(figsize=(8, 5))

sns.countplot(
    data=df,
    x="readmitted"
)

plt.title("Readmission Distribution")
plt.xlabel("Readmission Status")
plt.ylabel("Number of Patients")

plt.tight_layout()

plt.savefig(
    f"{OUTPUT_DIR}/readmission_distribution.png",
    dpi=150
)

plt.show()


# Length of Stay

plt.figure(figsize=(10, 5))

sns.countplot(
    data=df,
    x="time_in_hospital"
)

plt.title("Length of Hospital Stay Distribution")
plt.xlabel("Days in Hospital")
plt.ylabel("Number of Encounters")

plt.tight_layout()

plt.savefig(
    f"{OUTPUT_DIR}/length_of_stay.png",
    dpi=150
)

plt.show()


# =====================================================
# 6. NUMERIC FEATURES
# =====================================================

numeric_features = [

    "time_in_hospital",
    "num_lab_procedures",
    "num_procedures",
    "num_medications",
    "number_outpatient",
    "number_emergency",
    "number_inpatient",
    "number_diagnoses"

]

print("\n" + "=" * 70)
print("NUMERIC FEATURE ANALYSIS")
print("=" * 70)

for column in numeric_features:

    plt.figure(figsize=(8, 5))

    sns.histplot(
        df[column],
        kde=True
    )

    plt.title(
        f"Distribution of {column}"
    )

    plt.xlabel(column)
    plt.ylabel("Frequency")

    plt.tight_layout()
    plt.show()


# =====================================================
# 7. OUTLIER DETECTION
# =====================================================

print("\n" + "=" * 70)
print("OUTLIER ANALYSIS")
print("=" * 70)

for column in numeric_features:

    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)

    IQR = Q3 - Q1

    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR

    outliers = df[
        (df[column] < lower_bound) |
        (df[column] > upper_bound)
    ]

    print(
        f"{column}: "
        f"{len(outliers)} outliers"
    )

    plt.figure(figsize=(8, 4))

    sns.boxplot(
        x=df[column]
    )

    plt.title(
        f"Boxplot - {column}"
    )

    plt.tight_layout()
    plt.show()


# =====================================================
# 8. CORRELATION ANALYSIS
# =====================================================

print("\n" + "=" * 70)
print("CORRELATION ANALYSIS")
print("=" * 70)

corr = df[numeric_features].corr()

print(corr)

plt.figure(figsize=(11, 8))

sns.heatmap(
    corr,
    annot=True,
    cmap="coolwarm",
    fmt=".2f"
)

plt.title("Correlation Heatmap")

plt.tight_layout()

plt.savefig(
    f"{OUTPUT_DIR}/correlation_heatmap.png",
    dpi=150
)

plt.show()


# =====================================================
# 9. RELATIONSHIP ANALYSIS
# =====================================================

plt.figure(figsize=(8, 5))

sns.scatterplot(
    data=df,
    x="num_medications",
    y="time_in_hospital",
    alpha=0.3
)

plt.title(
    "Number of Medications vs Length of Stay"
)

plt.tight_layout()
plt.show()


plt.figure(figsize=(8, 5))

sns.scatterplot(
    data=df,
    x="num_lab_procedures",
    y="time_in_hospital",
    alpha=0.3
)

plt.title(
    "Lab Procedures vs Length of Stay"
)

plt.tight_layout()
plt.show()


# =====================================================
# 10. CATEGORICAL ANALYSIS
# =====================================================

categorical_features = [

    "gender",
    "race",
    "age",
    "weight",
    "max_glu_serum",
    "A1Cresult",
    "insulin",
    "diabetesMed"

]

for column in categorical_features:

    if column in df.columns:

        plt.figure(figsize=(10, 5))

        sns.countplot(
            data=df,
            x=column,
            order=df[column]
            .value_counts()
            .index
        )

        plt.title(
            f"{column} Distribution"
        )

        plt.xticks(
            rotation=45
        )

        plt.tight_layout()

        plt.show()


# =====================================================
# 11. AGE VS READMISSION
# =====================================================

plt.figure(figsize=(12, 6))

sns.countplot(
    data=df,
    x="age",
    hue="readmitted"
)

plt.title(
    "Age Group vs Readmission"
)

plt.xticks(rotation=45)

plt.tight_layout()

plt.show()


# =====================================================
# 12. GENDER VS READMISSION
# =====================================================

plt.figure(figsize=(8, 5))

sns.countplot(
    data=df,
    x="gender",
    hue="readmitted"
)

plt.title(
    "Gender vs Readmission"
)

plt.tight_layout()

plt.show()


# =====================================================
# 13. LENGTH OF STAY VS READMISSION
# =====================================================

plt.figure(figsize=(12, 6))

sns.countplot(
    data=df,
    x="time_in_hospital",
    hue="readmitted"
)

plt.title(
    "Length of Stay vs Readmission"
)

plt.tight_layout()

plt.show()


# =====================================================
# FINAL SUMMARY
# =====================================================

print("\n" + "=" * 70)
print("VITALSIGN EDA SUMMARY")
print("=" * 70)

print(
    "\nTotal Encounters:",
    len(df)
)

print(
    "Total Features:",
    df.shape[1]
)

print(
    "\nAverage Length of Stay:",
    round(
        df["time_in_hospital"].mean(),
        2
    ),
    "days"
)

print(
    "Median Length of Stay:",
    df["time_in_hospital"].median(),
    "days"
)

print(
    "\nReadmission Distribution:"
)

print(
    df["readmitted"]
    .value_counts()
)

print(
    "\nMost Common Age Group:",
    df["age"]
    .value_counts()
    .idxmax()
)

print(
    "Most Common Gender:",
    df["gender"]
    .value_counts()
    .idxmax()
)

print("\nEDA COMPLETED SUCCESSFULLY!")