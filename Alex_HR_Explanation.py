import pandas as pd

data = {
    "Candidate Name": ["Alice", "Bob", "Charlie", "David", "Eva", "Frank", "Grace", "Henry", "Ivy", "Jack"],
    "Education Score": [8, 7, 9, 6, 8, 5, 9, 6, 8, 6],
    "Years of Experience": [5, 3, 6, 3, 5, 2, 6, 3, 6, 2],
    "Number of Projects": [4, 3, 5, 2, 5, 2, 6, 2, 5, 3],
    "Communication Score": [7, 6, 8, 6, 7, 5, 8, 5, 7, 6],
    "Technical Score": [8, 7, 9, 7, 8, 6, 9, 5, 7, 6],
    "Leadership Score": [6, 5, 7, 5, 7, 4, 8, 4, 7, 4],
    "College Pedigree": [1, 0, 1, 0, 1, 0, 1, 0, 1, 0],
    "Name Bias": [1, 1, 0, 1, 0, 1, 0, 1, 0, 1],
    "Hired": [1, 0, 1, 0, 1, 0, 1, 0, 1, 0]
}

df = pd.DataFrame(data)
df.to_csv("candidates.csv", index=False)
