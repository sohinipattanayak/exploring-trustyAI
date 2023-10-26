import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

# Load the synthetic dataset
data = pd.read_csv("candidates.csv")

# Assume "Hired" column is the target and rest are features
X = data.drop(columns=["Candidate Name", "Hired"])
y = data["Hired"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Train a logistic regression model
clf = LogisticRegression(max_iter=1000).fit(X_train, y_train)
