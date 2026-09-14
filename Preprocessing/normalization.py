import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

file_path = r"/placement_preprocessed.csv"

df = pd.read_csv(file_path)
print(df)

num_cols = [
    "CGPA",
    "AttendancePercent",
    "AptitudeTestScore",
    "CodingTestScore",

]
train_df, test_df = train_test_split(
df,
test_size = 0.2,
random_state= 42,
stratify = df["PlacementStatus"]
)
print("/nTraining Data before scaling:")
print(train_df[num_cols].head())

print("/nTesting Data before scaling:")
print(test_df[num_cols].head())

scaler = MinMaxScaler()

train_df[num_cols] =scaler.fit_transform(train_df[num_cols])

test_df[num_cols] = scaler.transform(test_df[num_cols])
print("\nScaled Training Data")
print(train_df[num_cols].head())
print("\nScaled Testing Data")
print(test_df[num_cols].head())

