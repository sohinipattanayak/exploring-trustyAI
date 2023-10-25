import numpy as np
from trustyai.model import Model
from trustyai.explainers import LimeExplainer

# Create a simple linear model with 5 features
weights = np.random.uniform(low=-5, high=5, size=5)
print(f"Weights: {weights}")

def linear_model(x):
    return np.dot(x, weights)

# Wrap the linear model in the TrustyAI Model class
model = Model(linear_model)

# Establish a random data point for explanation
model_input = np.random.rand(1, 5)
model_output = model(model_input)

# Initialize LIME explainer
lime_explainer = LimeExplainer(samples=1000, normalise_weights=False)

# Produce explanations using LIME
lime_explanation = lime_explainer.explain(
    inputs=model_input,
    outputs=model_output,
    model=model)

# Display the explanations
print(lime_explanation.as_dataframe())