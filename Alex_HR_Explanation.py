import numpy as np
from trustyai.model import Model
from trustyai.explainers import LimeExplainer

# Define weights for the model. In a real-world scenario, these would come from training data.
weights = np.random.uniform(low=-5, high=5, size=7)
print(f"Weights for Features: {weights}")

# Simple linear model representing the recruitment decision process.
def recruitment_model(x):
    return np.dot(x, weights)

model = Model(recruitment_model)

# Sigmoid function to squash values between 0 and 1.
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# Sample data for a candidate.
candidate_data = np.random.rand(1, 7)
raw_hiring_score = model(candidate_data)
predicted_hiring_score_percentage = sigmoid(raw_hiring_score) * 100

lime_explainer = LimeExplainer(samples=1000, normalise_weights=False)
lime_explanation = lime_explainer.explain(
    inputs=candidate_data,
    outputs=raw_hiring_score,
    model=model)

print(lime_explanation.as_dataframe())

print("Summary of the explanation:")
if raw_hiring_score > 0:
  print("The candidate is likely to be selected.")
else:
  print("The candidate is unlikely to be selected.")

print(f"Predicted Hiring Score Percentage: {predicted_hiring_score_percentage[0]:.2f}%")

print("Feature weights:")
for feature, weight in zip(["Years of Experience", "Technical Skill Certifications",
                            "Duration at the Last Job", "Number of Projects Completed",
                            "GitHub Repositories and Contributions", "Education Level",
                            "Name Bias (American-sounding=1, Non-American sounding=0)",
                            "College Attended (Pedigree=1, Non-Pedigree=0)"], weights):
  print(f"{feature}: {weight:.2f}")
