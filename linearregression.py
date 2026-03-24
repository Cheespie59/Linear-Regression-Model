import matplotlib.pyplot as plt
from scipy import stats
x = [1,2,3,4,5,6,7,8,9,10]
y = [2,4,6,8,10,12,14,16,18,20]
slope, intercept, r, p, std_err = stats.linregress(x, y)
def myfunc(x):
  return slope * x + intercept
mymodel = list(map(myfunc, x))
plt.scatter(x, y)
plt.plot(x, mymodel)
plt.xlabel("x")
plt.ylabel("y")
plt.show()
print("Regression Coefficients: ", slope, " and ", intercept)
print("p-value= ",p)
print("R2 Score= ",r)
print("Standard Error= ",std_err)
print("Equation of line:")
print("y=",round(slope,2),"x","+",round(intercept,2),"+",round(std_err,0))
print("For x=11, y=",myfunc(11))

import pandas as pd
df=pd.read_csv("Housing(1).csv",index_col=None)
print(df)

df.info()

print(df.describe())

import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt

# Reload the original DataFrame to ensure a clean state for processing
df = pd.read_csv("Housing(1).csv",index_col=None)

# Convert 'yes'/'no' columns to numerical (1/0)
binary_cols = ['mainroad', 'guestroom', 'basement', 'hotwaterheating', 'airconditioning', 'prefarea']
for col in binary_cols:
    df[col] = df[col].apply(lambda x: 1 if x == 'yes' else 0)

# Handle 'furnishingstatus' column using one-hot encoding
df = pd.get_dummies(df, columns=['furnishingstatus'], drop_first=True)

# Define the exact columns you want for the correlation matrix
selected_columns = ['price', 'area', 'bedrooms', 'bathrooms', 'stories', 'parking']
df_selected = df[selected_columns]

# Generate correlation matrix for *only* the selected columns
corr_matrix_selected = df_selected.corr()

# Print the columns to confirm what is being used for the heatmap
print("Columns included in the correlation heatmap:", corr_matrix_selected.columns.tolist())

# Create heatmap using Seaborn with the specified attributes
plt.figure(figsize=(10, 8)) # Adjust figure size for fewer columns
sns.heatmap(data=corr_matrix_selected,
            annot=True,
            cmap='RdBu_r', # Using a diverging color palette
            fmt=".2f",
            linewidths=0.25,
            linecolor='white',
            cbar=True,
            cbar_kws={'label': 'Correlation Coefficient'})
plt.title("Correlation Matrix Heatmap (Selected Columns)")
plt.tight_layout() # Adjust layout to prevent labels from being cut off
plt.show()