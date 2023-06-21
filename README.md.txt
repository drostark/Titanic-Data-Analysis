# Titanic Data Analysis

This data analysis explores various aspects of the Titanic dataset to gain insights into the passengers' demographics, their cabins, cities of embarkation, family status, and factors influencing survival rates.

## Dataset Information
- Dataset Link: [Titanic Dataset](https://www.kaggle.com/c/titanic)

### Part 1: Column Names Clarification for `train.csv`
- `Survived` column: 0 indicates not survived, 1 indicates survived
- `Pclass`: Class of the passenger (1, 2, 3)
- `SibSp`: Number of siblings on board (0 indicates none, 1 indicates at least one)
- `Parch`: Number of parents or children aboard
- `Embarked`: Port of embarkation (C = Cherbourg, Q = Queenstown, S = Southampton)

Upon using the `.info()` function, it was found that the `Cabin` column has only 284 entries out of 891 total passengers, indicating missing information for the majority of passengers.

## Questions to be Answered

1. Who were the passengers on the Titanic? (Ages, Gender, Class, etc.)
2. What deck were the passengers on, and how does that relate to their class?
3. Where did the passengers come from?
4. Who was alone and who was with family?
5. What factors helped someone survive the sinking? (broader question)

## Data Preparation

The datasets used for analysis were obtained from Kaggle.com as part of a challenge task.

### Part 2: Additional Information

- `FacetGrid` enables making multiple plots on one figure.
- A KDE plot comparing male versus female passengers was created.
- Note: `shade=True` is not available in seaborn anymore, so `fill=True` is used instead.
- The KDE plot for Male vs. Female vs. Children extends beyond 16 years due to the bandwidth.

## Analysis Findings

### Answering Question 2: What deck were the passengers on, and how does that relate to their class?

To determine the deck for each passenger:
- NaN values in the `Cabin` column were dropped.
- The first letter from the `Cabin` column was extracted to identify the deck (A, B, C, D, E, F).

### Answering Question 3: Where did the passengers come from?

A categorical plot (`catplot`) was used to visualize the distribution of passengers across different cities of embarkation (`Embarked`), categorized by passenger class (`Pclass`).

- The majority of passengers embarked from Southampton (`S`).
- Queenstown (`Q`) had mostly third-class passengers, indicating a potential difference in financial situations.
- Cherbourg (`C`) had a significant number of passengers who embarked in the first class.

### Answering Question 4: Who was alone and who was with family?

The definition of "alone" was established as follows:
- If both `SibSp` (number of siblings) and `Parch` (number of parents or children) are 0, the passenger was considered completely alone.

### Answering Question 5: What factors helped someone survive the sinking?

- A new column, `Survivor`, was created to identify whether a passenger survived (yes) or not (no).
- It was observed that passenger class and gender played significant roles in survival chances.
- Being male dramatically decreased the chances of survival.
- Older females had better survival chances than older males.

## Additional Questions

### 1. Did the deck have an effect on the passengers' survival rate? Did this answer match up with your intuition?

To investigate the effect of the deck on survival rate, the analysis considered the relationship between deck and class.

### 2. Did having a family member increase the odds of surviving the crash?

The analysis explored whether having a family member increased the odds of surviving the crash.