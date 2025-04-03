import pandas as pd
from sklearn.preprocessing import StandardScaler

# Load the dataset
df = pd.read_csv('data/global_food_wastage_dataset.csv')

# Identify numeric and categorical columns
numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
categorical_cols = df.select_dtypes(include=['object']).columns.tolist()

# Feature Engineering
df['Year Difference'] = df['Year'] - 2019
df['Economic Loss per Capita'] = df['Economic Loss (Million $)'] / df['Population (Million)']
df['Total Waste per Household'] = df['Total Waste (Tons)'] * (df['Household Waste (%)'] / 100)
df['Average Economic Loss per Ton'] = df['Economic Loss (Million $)'] / df['Total Waste (Tons)']

# One-Hot Encoding for categorical columns
df = pd.get_dummies(df, columns=['Country', 'Food Category'])

# Normalization for numeric columns
scaler = StandardScaler()
df[['Total Waste (Tons)', 'Economic Loss (Million $)', 'Avg Waste per Capita (Kg)']] = scaler.fit_transform(
    df[['Total Waste (Tons)', 'Economic Loss (Million $)', 'Avg Waste per Capita (Kg)']]
)

# Save the processed data
df.to_csv('data/processed_global_food_wastage_dataset.csv', index=False)