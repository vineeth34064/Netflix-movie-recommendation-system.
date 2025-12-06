# Netflix-movie-recommendation-system using apriori algorithm.
“A machine learning–based movie recommendation system that analyzes user preferences and suggests personalized Netflix-style movie recommendations using collaborative filtering and content-based techniques.”

This notebook demonstrates how to build a basic movie recommendation system using the Apriori algorithm. The goal is to discover association rules between movies that are frequently watched together, which can then be used to suggest movies to users.

## Project Overview

The project involves the following key steps:

1.  **Data Loading**: Loading a dataset of movie transactions (e.g., movies watched by users) from a CSV file.
2.  **Data Preprocessing**: Converting the transactional data into a format suitable for the Apriori algorithm.
3.  **Applying Apriori Algorithm**: Utilizing the `apyori` library to find frequent itemsets and generate association rules based on specified support, confidence, and lift thresholds.
4.  **Result Visualization**: Displaying the generated association rules in a structured format (Pandas DataFrame) and identifying the most significant rules.

## Data Source

The dataset `netflix.csv` is assumed to contain rows where each row represents a transaction (e.g., a user's movie watching history), and columns contain the names of movies watched in that transaction. NaN values indicate fewer movies watched in a particular transaction.

## Setup and Installation

To run this notebook, you need to install the `apyori` library:

```bash
pip install apyori
```

## Key Steps in the Notebook

### 1. Importing Libraries

Essential libraries like `numpy`, `pandas`, `matplotlib.pyplot`, and `apyori` are imported for data handling, analysis, and visualization.

### 2. Loading the Dataset

The `netflix.csv` file is loaded into a Pandas DataFrame:

```python
dataset = pd.read_csv('netflix.csv')
```

### 3. Data Transformation for Apriori

The dataset is transformed into a list of lists, where each inner list represents a transaction and contains the movies watched in that transaction. NaN values are converted to strings and then implicitly handled by the Apriori algorithm (or removed, depending on the implementation details and how `apyori` treats string 'nan').

```python
transactions = dataset.iloc[:7466, :20].astype(str).values.tolist()
```

### 4. Applying the Apriori Algorithm

The `apriori` function is used to generate association rules with the following parameters:

-   `min_support = 0.003`: The minimum support threshold for an itemset to be considered frequent.
-   `min_confidence = 0.2`: The minimum confidence threshold for a rule.
-   `min_lift = 3`: The minimum lift threshold for a rule, indicating a strong positive correlation between items.
-   `min_len = 2`: The minimum number of items in a rule.

```python
from apyori import apriori
netflix_rules = apriori(transactions=transactions, min_support=0.003, min_confidence=0.2, min_lift=3, min_len=2)
```

### 5. Visualizing and Interpreting Results

The generated rules, which are initially a generator object, are converted into a list and then parsed into a Pandas DataFrame for easier inspection. A custom `inspect` function extracts the left-hand side (movie1), right-hand side (movie2), and support for each rule.

```python
def inspect(results):
  lhs = [tuple(result[2][0][0])[0] for result in results]
  rhs = [tuple(result[2][0][1])[0] for result in results]
  supports = [result[1] for result in results]
  return list(zip(lhs,rhs,supports))

resultdata = pd.DataFrame(inspect(list(netflix_rules)), columns=['movie1', 'movie2', 'support'])
```

Finally, the top 30 rules by support are displayed to show the strongest associations:

```python
resultdata.nlargest(n=30, columns='support')
```

## Conclusion

This notebook provides a foundational example of how association rule mining can be applied to build a simple movie recommendation system. The resulting rules can be used to recommend movies to users based on their viewing history or preferences.
