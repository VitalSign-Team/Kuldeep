import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

file_path = r"/placement_preprocessed.csv"

df = pd.read_csv(file_path)

feature = "CodingTestScore"
print("Original stats:")

Q1 = df[feature].quantile(0.25)
Q3 = df[feature].quantile(0.75)

IQR = Q3 - Q1

lower_fence = Q1 - 1.5 * IQR
upper_fence = Q3 - 1.5 * IQR

print("\nQ1 =", Q1)
print("Q3 =", Q3)
print("IQR =", IQR)
print("Lower Fence =", lower_fence)
print("Upper Fence =", upper_fence)

outliers =df[
    (df[feature] > lower_fence) |
    (df[feature] < upper_fence)
]
print("\nNumber of outlier =", len(outliers))

df["CodingTestScore_Clipped"] = df[feature].clip(
    lower =lower_fence,
    upper =upper_fence
)
print("\nMinimum BEFORE clipping:")
print(df[feature].min())

print("\nMinimum AFTER clipping:")
print(df["CodingTestScore_Clipped"].min())

scaler = MinMaxScaler()

df["CodingTestScore_Scaled"] = scaler.fit_transform(df[["CodingTestScore_Clipping"]])