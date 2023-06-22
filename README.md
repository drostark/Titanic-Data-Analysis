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
![Distribution of Passengers by Sex](https://github.com/drostark/Titanic-Data-Analysis/assets/52506085/6872f639-59c8-4c92-a9e4-5fda5e6de422)
   - Determine the distribution of passengers by age.
![Distribution of Passengers by Age](https://github.com/drostark/Titanic-Data-Analysis/assets/52506085/9500b072-80ac-460c-8ee6-50c9db80438a)
   - To provide a more comprehensive analysis of the distribution of passengers by gender, class, and age, a new column called 'person' was created. Using a binary method, individuals with an age greater than 16 were classified as adults, while those aged 16 or younger were classified as children.
![Distribution of Passengers by Gender and Class](https://github.com/drostark/Titanic-Data-Analysis/assets/52506085/e934ec74-cb56-4d5d-8d0d-08d6ca07d879)
   - To visualize the age distribution among different passenger classes, we created a Kernel Density Estimation (KDE) plot. This plot provides a smooth estimate of the probability density function for each passenger class, allowing us to observe any variations in age distribution.
![KDE Age Distribution in Different Passenger Classes](https://github.com/drostark/Titanic-Data-Analysis/assets/52506085/183f1b48-b359-405e-972d-7ca1b421511c)
The KDE plot displays the density of age distribution for each passenger class, helping us understand the age demographics within different classes.
   - I also examined the gender and age distribution across different age groups using another KDE plot. This plot allows us to visualize the density of gender and age within specific age intervals.
![KDE Gender and Age Distribution in Different Age Groups](https://github.com/drostark/Titanic-Data-Analysis/assets/52506085/81e39e7e-493c-49db-b75e-3575ccd26dcf)
This KDE plot provides insights into the distribution of gender and age across different age groups, aiding our understanding of the composition of passengers within specific age ranges.

2. What deck were the passengers on, and how does that relate to their class?

   - Extract the deck information from the "Cabin" column.
   - Visualize the distribution of passengers by cabin deck and class.

   ![Distribution of Passengers by Deck and Class](https://github.com/drostark/Titanic-Data-Analysis/assets/52506085/8723ad1d-56bd-48e6-860b-0080f4f8a03f)


3. Where did the passengers come from?

   - Analyze the distribution of passengers by the port of embarkation (Cherbourg, Queenstown, Southampton) and class.

   ![Distribution of Passengers by Port of Embarkation and Class](https://github.com/drostark/Titanic-Data-Analysis/assets/52506085/7d4265d6-042c-4e09-9cdb-2b9809ad5c89)


4. Who was alone, and who was with family?

   - Determine if passengers were traveling alone or with family by creating a new column called "Alone" based on the number of siblings/spouses and parents/children aboard.
   - Visualize the count of passengers who were alone versus with family using a bar plot to compare the number of individuals in each category.

   ![Count of Passengers Alone vs. with Family](https://github.com/drostark/Titanic-Data-Analysis/assets/52506085/b9aaf087-23b6-48e0-a9dd-03e3619228ca)

5. What factors helped someone survive the sinking?

   - Create a "Survivor" column mapping the survival status (0 = No, 1 = Yes).
   - Visualize the count of survivors.
   ![Count of Survivors](https://github.com/drostark/Titanic-Data-Analysis/assets/52506085/08f21733-23b5-4c7f-8e9d-fd0f752a289a)
   - Analyze the survival rate based on passenger class, gender, and age.
   ![Survival Rate by Passenger Class](https://github.com/drostark/Titanic-Data-Analysis/assets/52506085/3e9f873b-15b6-4f70-918b-04daf096857f)
   
   ![Survival Rate by Sex and Age](https://github.com/drostark/Titanic-Data-Analysis/assets/52506085/1fff2295-a869-4f9a-a380-10715931e934)
   
   ![Survival Rate by Age and Passenger Class](https://github.com/drostark/Titanic-Data-Analysis/assets/52506085/c8dd08ee-fb37-42cd-abd3-f6ba9e081702)

6. Did the deck have an effect on the passengers' survival rate?

   - To determine the effect of the deck on the passengers' survival rate, analyze the survival rate based on the deck they were on.

   ![Survival Count by Deck and Class](https://github.com/drostark/Titanic-Data-Analysis/assets/52506085/0e0fc83c-8351-4664-9587-3d8113173aa8)

7. Did having a family member increase the odds of surviving the crash?

   - Analyze the survival rate based on the presence of family members (siblings/spouses and parents/children).

   ![Survival Rate by Presence of Family](https://github.com/drostark/Titanic-Data-Analysis/assets/52506085/947047c0-ca9c-4d31-99e9-d8a1a11ccc0c)
