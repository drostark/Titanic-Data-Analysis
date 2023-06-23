# Titanic Data Analysis

This project analyzes the Titanic dataset to gain insights into the passengers on board the Titanic and factors that influenced their survival. The dataset used for this analysis is available on Kaggle at [Titanic Dataset](https://www.kaggle.com/c/titanic/data).

## Dataset Information

The dataset contains information about the passengers on the Titanic. Here are some key columns and their descriptions from the `train.csv` file:

- Survived: 0 indicates not survived, 1 indicates survived.
- Pclass: Class of the passenger (1, 2, or 3).
- SibSp: Number of siblings/spouses aboard.
- Parch: Number of parents/children aboard.
- Ticket: Ticket number
- Cabin: Cabin Number
- Embarked: Port of embarkation (C = Cherbourg, Q = Queenstown, S = Southampton).

## Project Overview

In this project, utilizing the Google data analysis process, which includes the Ask, Prepare, Process, Analyze, Share, and Act phases, our objective is to address the following inquiries pertaining to the Titanic dataset:

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
```python
titanic_df = pd.read_csv("input/train.csv")
```
2. General Overview of the train.csv Dataset 
```python
titanic_df.head(10)
```
```
   PassengerId  Survived  Pclass                                               Name     Sex   Age  SibSp  Parch            Ticket     Fare Cabin Embarked
0            1         0       3                            Braund, Mr. Owen Harris    male  22.0      1      0         A/5 21171   7.2500   NaN        S
1            2         1       1  Cumings, Mrs. John Bradley (Florence Briggs Th...  female  38.0      1      0          PC 17599  71.2833   C85        C
2            3         1       3                             Heikkinen, Miss. Laina  female  26.0      0      0  STON/O2. 3101282   7.9250   NaN        S
3            4         1       1       Futrelle, Mrs. Jacques Heath (Lily May Peel)  female  35.0      1      0            113803  53.1000  C123        S
4            5         0       3                           Allen, Mr. William Henry    male  35.0      0      0            373450   8.0500   NaN        S
```

```python
titanic_df.describe()
```
```
       PassengerId    Survived      Pclass         Age       SibSp       Parch        Fare
count   891.000000  891.000000  891.000000  714.000000  891.000000  891.000000  891.000000
mean    446.000000    0.383838    2.308642   29.699118    0.523008    0.381594   32.204208
std     257.353842    0.486592    0.836071   14.526497    1.102743    0.806057   49.693429
min       1.000000    0.000000    1.000000    0.420000    0.000000    0.000000    0.000000
25%     223.500000    0.000000    2.000000   20.125000    0.000000    0.000000    7.910400
50%     446.000000    0.000000    3.000000   28.000000    0.000000    0.000000   14.454200
75%     668.500000    1.000000    3.000000   38.000000    1.000000    0.000000   31.000000
max     891.000000    1.000000    3.000000   80.000000    8.000000    6.000000  512.329200
```
4. Check for missing information using the `info()` and `isnull()` methods.
```python
titanic_df.info()
```
```
 1   Survived     891 non-null    int64
 2   Pclass       891 non-null    int64
 3   Name         891 non-null    object
 4   Sex          891 non-null    object
 5   Age          714 non-null    float64
 6   SibSp        891 non-null    int64
 7   Parch        891 non-null    int64
 8   Ticket       891 non-null    object
 9   Fare         891 non-null    float64
 10  Cabin        204 non-null    object
 11  Embarked     889 non-null    object
dtypes: float64(2), int64(5), object(5)
memory usage: 83.7+ KB
```
- The training dataset consists of a total of 891 rows and 12 columns. Please note that the "Cabin" column has only 204 non-null entries out of 891 total passengers.
```python
titanic_df.isnull().sum()
```
```
PassengerId      0
Survived         0
Pclass           0
Name             0
Sex              0
Age            177
SibSp            0
Parch            0
Ticket           0
Fare             0
Cabin          687
Embarked         2
dtype: int64
```
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

3. What deck were the passengers on, and how does that relate to their class?

   - Extract the deck information from the "Cabin" column.
   - Visualize the distribution of passengers by cabin deck and class.

      ![Distribution of Passengers by Deck and Class](https://github.com/drostark/Titanic-Data-Analysis/assets/52506085/8723ad1d-56bd-48e6-860b-0080f4f8a03f)


4. Where did the passengers come from?

   - Analyze the distribution of passengers by the port of embarkation (Cherbourg, Queenstown, Southampton) and class.

      ![Distribution of Passengers by Port of Embarkation and Class](https://github.com/drostark/Titanic-Data-Analysis/assets/52506085/7d4265d6-042c-4e09-9cdb-2b9809ad5c89)


5. Who was alone, and who was with family?

   - Determine if passengers were traveling alone or with family by creating a new column called "Alone" based on the number of siblings/spouses and parents/children aboard.
   - Visualize the count of passengers who were alone versus with family using a bar plot to compare the number of individuals in each category.

      ![Count of Passengers Alone vs. with Family](https://github.com/drostark/Titanic-Data-Analysis/assets/52506085/b9aaf087-23b6-48e0-a9dd-03e3619228ca)

6. What factors helped someone survive the sinking?

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
