import numpy as np
import matplotlib.pyplot as plt
from trustyai.model import Model
from trustyai.explainers import LimeExplainer

# Define 10 essential features for HR resume screening
features = [
    "Years of Experience",
    "Education Level",
    "Skill Match",
    "Culture Fit",
    "Referral Status",
    "Certifications",
    "Internships Completed",
    "Languages Known",
    "Tech Stack Proficiency",
    "Leadership Experience"
]

# Random weights for our hypothetical model.
weights = np.random.uniform(low=-5, high=5, size=10)
print(f"Weights for Features: {weights}")

# Simple linear model representing the resume screening process
def resume_screening_model(x):
    return np.dot(x, weights)

model = Model(resume_screening_model)

# Sample data for an applicant (random for this example)
applicant_data = np.random.rand(1, 10)
predicted_resume_screening_score = model(applicant_data)

lime_explainer = LimeExplainer(samples=1000, normalise_weights=False)
lime_explanation = lime_explainer.explain(
    inputs=applicant_data,
    outputs=predicted_resume_screening_score,
    model=model)

print(lime_explanation.as_dataframe())

# ... [rest of the code above this remains unchanged]

# Access the dataframe for the output
output_df = lime_explanation.as_dataframe()["output-0"]

# Initialize saliencies to near-zero for all features
all_saliencies = {feature: -0.01 for feature in features}

# Update saliencies with LIME output for the explained features
for index, row in output_df.iterrows():
    all_saliencies[features[int(row['Feature'].split('-')[1])]] = row['Saliency']

# Sorting features based on saliency values
sorted_features = sorted(all_saliencies.items(), key=lambda item: item[1], reverse=True)

# Top 3 positive and negative features
top_positive_features = sorted_features[:3]
top_negative_features = sorted_features[-3:]

# Neutral features: Features with saliency near zero
neutral_threshold = 0.1
neutral_features = [feature for feature, saliency in all_saliencies.items() if abs(saliency) < neutral_threshold]

# Insights from the visualizations
print("\nInsights:")
print("-" * 40)

print("\nTop Positive Features Influencing the Decision:")
for feature, saliency in top_positive_features:
    print(f"  - {feature}: {saliency:.3f}")

print("\nTop Negative Features Influencing the Decision:")
for feature, saliency in top_negative_features:
    print(f"  - {feature}: {saliency:.3f}")

if neutral_features:
    print("\nFeatures with Neutral Impact:")
    for feature in neutral_features:
        print(f"  - {feature}")
else:
    print("\nNo features with neutral impact!")

y_axis = list(all_saliencies.values())
x_axis = list(all_saliencies.keys())
colors = []

# Coloring based on saliency values
for value in y_axis:
    if value < 0:
        colors.append("red")
    else:
        colors.append("green")

fig = plt.figure(figsize=(15, 7))
ax = fig.add_subplot(111)
ax.set_facecolor("#f2f2f2")
ax.bar(x_axis, height=y_axis, color=colors)
plt.title('LIME: Feature Impact for Resume Screening')
plt.xticks(rotation=45, ha='right', fontsize=10)
ax.axhline(0, color="black")  # x-axis line
plt.tight_layout()  # Adjust layout for better visualization
plt.show()

