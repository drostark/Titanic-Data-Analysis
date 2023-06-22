# Titanic Data Analysis

This project analyzes the Titanic dataset to gain insights into the passengers on board the Titanic and factors that influenced their survival. The dataset used for this analysis is available on Kaggle at [Titanic Dataset](https://www.kaggle.com/c/titanic/data).

## Dataset Information

The dataset contains information about the passengers on the Titanic. Here are some key columns and their descriptions from the `train.csv` file:

- Survived: 0 indicates not survived, 1 indicates survived.
- Pclass: Class of the passenger (1, 2, or 3).
- Sex: Gender of the passenger.
- Age: Age of the passenger.
- SibSp: Number of siblings/spouses aboard.
- Parch: Number of parents/children aboard.
- Embarked: Port of embarkation (C = Cherbourg, Q = Queenstown, S = Southampton).

Please note that the "Cabin" column has only 284 non-null entries out of 891 total passengers.

## Project Overview

The project aims to answer the following questions about the Titanic dataset:

1. Who were the passengers on the Titanic? (Ages, Gender, Class, etc.)
2. What deck were the passengers on, and how does that relate to their class?
3. Where did the passengers come from?
4. Who was alone, and who was with family?
5. What factors helped someone survive the sinking?
6. Did the deck have an effect on the passengers' survival rate?
7. Did having a family member increase the odds of surviving the crash?

## Project Steps

**Part 1: Data Preparation**

1. Load the dataset from the provided CSV file.
2. Check for missing information using the `info()` method.
   - Note that the "Cabin" column has missing data.

**Part 2: Exploratory Data Analysis**

Perform exploratory data analysis to answer the questions and gain insights into the dataset. Use various plots, including bar plots, count plots, histograms, and kernel density estimation (KDE) plots.

**Answers to Questions**

1. Who were the passengers on the Titanic? (Ages, Gender, Class, etc.)

   - Visualize the distribution of passengers by sex (male vs. female).
   - Analyze the distribution of passengers by gender and class.
   - Determine the distribution of passengers by age.

   ![Distribution of Passengers by Sex](https://www.dropbox.com/sh/70exukhhno032te/AABZ3wi72MZKGTsG8fzZFbS8a?dl=0)
   
   ![Distribution of Passengers by Gender and Class](plot2.png)
   
   ![Distribution of Passengers by Age](plot3.png)

2. What deck were the passengers on, and how does that relate to their class?

   - Extract the deck information from the "Cabin" column.
   - Visualize the distribution of passengers by cabin deck and class.

   ![Distribution of Passengers by Deck and Class](plot4.png)

3. Where did the passengers come from?

   - Analyze the distribution of passengers by the port of embarkation (Cherbourg, Queenstown, Southampton) and class.

   ![Distribution of Passengers by Port of Embarkation and Class](plot5.png)

4. Who was alone, and who was with family?

   - Define the "Alone" column based on the number of siblings/spouses and parents/children aboard.
   - Visualize the count of passengers who were alone versus with family.

   ![Count of Passengers Alone vs. With Family](plot6.png)

5. What factors helped someone survive the sinking?

   - Create a "Survivor" column mapping the survival status (0 = No, 1 = Yes).
   - Visualize the count of survivors.
   - Analyze the survival rate based on passenger class, gender, and age.

   ![Count of Survivors](plot7.png)
   
   ![Survival Rate by Passenger Class](plot8.png)
   
   ![Survival Rate by Gender](plot9.png)
   
   ![Survival Rate by Age](plot10.png)

6. Did the deck have an effect on the passengers' survival rate?

   - To determine the effect of the deck on the passengers' survival rate, analyze the survival rate based on the deck they were on.

   ![Survival Count by Deck and Class](plot11.png)

7. Did having a family member increase the odds of surviving the crash?

   - Analyze the survival rate based on the presence of family members (siblings/spouses and parents/children).

   ![Survival Rate by Presence of Family](plot12.png)
