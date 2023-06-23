# Titanic Data Analysis

This project analyzes the Titanic dataset to gain insights into the passengers on board the Titanic and the factors that influenced their survival. The dataset used for this analysis is available on Kaggle at [Titanic Dataset](https://www.kaggle.com/c/titanic/data).

## Dataset Information

The dataset contains information about the passengers on the Titanic. Here are some key columns and their descriptions from the `train.csv` file:

- Survived: 0 indicates not survived, 1 indicates survived.
- Pclass: Class of the passenger (1, 2, or 3).
- SibSp: Number of siblings/spouses aboard.
- Parch: Number of parents/children aboard.
- Ticket: Ticket number
- Cabin: Cabin Number
- Embarked: Port of embarkation (C = Cherbourg, Q = Queenstown, S = Southampton).

## Project Overview and Steps

In this project, utilizing the Google data analysis process, which includes the `Ask`, `Prepare`, `Process`, `Analyze`, and `Share` phases. Our objective is to address the following inquiries pertaining to the Titanic dataset:

**Part 1: Ask**
--------------------
1. Who were the passengers on the Titanic? (Ages, Gender, Class, etc.)
2. What deck were the passengers on, and how does that relate to their class?
3. Where did the passengers come from?
4. Who was alone, and who was with family?
5. What factors helped someone survive the sinking?
6. Did the deck have an effect on the passengers' survival rate?
7. Did having a family member increase the odds of surviving the crash?

**Part 2: Data Preparation**
--------------------
1. Imports
```python
import pandas as pd
from pandas import Series, DataFrame
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
```
2. Adjusting the plot title and incorporating interactive plotting capabilities.
```python
# Interactive plotting
plt.ion()
# Set the global title offset
title_offset = 0.2
plt.rcParams['axes.titlepad'] = title_offset
```
3. Load the dataset from the provided CSV file.
```python
titanic_df = pd.read_csv("input/train.csv")
```
4. General Overview of the train.csv Dataset 
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
5. Check for missing information using the `info()` and `isnull()` methods.
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
**Part 3: Data Process**
--------------------
1. To answer the first question, a new column called `person` was created. Using a binary method, individuals with an age greater than 16 were classified as adults, while those aged 16 or younger were classified as children.
```python
def male_female_child(passenger):
    age, sex = passenger
    if age < 16:
        return 'child'
    else:
        return sex


titanic_df['person'] = titanic_df[['Age', 'Sex']].apply(male_female_child, axis=1)
titanic_df[0:10]
```
```
   PassengerId  Survived  Pclass                                               Name     Sex   Age  SibSp  Parch            Ticket     Fare Cabin Embarked  person
0            1         0       3                            Braund, Mr. Owen Harris    male  22.0      1      0         A/5 21171   7.2500   NaN        S    male
1            2         1       1  Cumings, Mrs. John Bradley (Florence Briggs Th...  female  38.0      1      0          PC 17599  71.2833   C85        C  female
2            3         1       3                             Heikkinen, Miss. Laina  female  26.0      0      0  STON/O2. 3101282   7.9250   NaN        S  female
3            4         1       1       Futrelle, Mrs. Jacques Heath (Lily May Peel)  female  35.0      1      0            113803  53.1000  C123        S  female
4            5         0       3                           Allen, Mr. William Henry    male  35.0      0      0            373450   8.0500   NaN        S    male
5            6         0       3                                   Moran, Mr. James    male   NaN      0      0            330877   8.4583   NaN        Q    male
6            7         0       1                            McCarthy, Mr. Timothy J    male  54.0      0      0             17463  51.8625   E46        S    male
7            8         0       3                     Palsson, Master. Gosta Leonard    male   2.0      3      1            349909  21.0750   NaN        S   child
8            9         1       3  Johnson, Mrs. Oscar W (Elisabeth Vilhelmina Berg)  female  27.0      0      2            347742  11.1333   NaN        S  female
9           10         1       2                Nasser, Mrs. Nicholas (Adele Achem)  female  14.0      1      0            237736  30.0708   NaN        C   child
```
2. In order to address the second question, it was necessary to determine the deck by extracting the first letter from the 'Cabin' column. It is important to note that the letter 'T' unexpectedly appeared in the 'Cabin' category. However, to ensure a more accurate representation of the decks, this category was dropped from the analysis.
```python
deck = titanic_df['Cabin'].dropna()

# Grabbing only the first letter from the deck column 'A','B','C','D','E','F'
levels = []
for level in deck:
    if level[0] == 'T':
        levels.append(level[1:])
    else:
        levels.append(level[0])

cabin_df = DataFrame(levels, columns=['Cabin'])
cabin_df['Cabin'] = cabin_df['Cabin'].astype('category')
cabin_df = cabin_df[cabin_df['Cabin'] != '']
```
```
    Cabin
0       C
1       C
2       E
3       G
4       C
..    ...
199     D
200     B
201     C
202     B
203     C
```
3. To provide a clearer explanation for question number 3, a mapping dictionary was utilized to accurately identify the city names in the plot. This allowed for a more precise representation and interpretation of the data related to the embarked passengers' cities.
```python
city_mapping = {'C': 'Cherbourg', 'Q': 'Queenstown', 'S': 'Southampton'}
```
4. To address question number 4, it was necessary to establish the criteria for determining whether a passenger was traveling alone. This was achieved by considering the information from the `SibSp` column, which indicates the number of siblings/spouses aboard, and the `Parch` column, which indicates the number of parents/children aboard. By incorporating the information from these two columns, a new column named `Alone` was created to classify passengers as either being with or without family. This allowed for a more comprehensive understanding of whether individuals were traveling alone or accompanied by family members.
```python
titanic_df['Alone'] = titanic_df.SibSp + titanic_df.Parch
titanic_df['Alone'].loc[titanic_df['Alone']> 0] = 'With Family'
titanic_df['Alone'].loc[titanic_df['Alone'] == 0] = 'Alone'
```
```
0      With Family
1      With Family
2            Alone
3      With Family
4            Alone
          ...
886          Alone
887          Alone
888    With Family
889          Alone
890          Alone
```
5. To address question number 5, the `Survivor` column was introduced. It contains categorical values indicating whether a passenger survived or not. The values 'No' and 'Yes' are assigned to represent passengers who did not survive and those who did survive, respectively. This mapping provides a clear distinction and aids in analyzing the factors that influenced passenger survival during the Titanic incident.
```python
titanic_df['Survivor'] = titanic_df.Survived.map({0:'No',1:'Yes'})
```
```
0       No
1      Yes
2      Yes
3      Yes
4       No
      ...
886     No
887    Yes
888     No
889    Yes
890     No
Name: Survivor, Length: 891, dtype: object
```
6. In order to address question 6, I merged the new 'Cabin' column with the 'Survived' column by concatenating them. Any missing values in the resulting dataset were subsequently removed, ensuring a complete and reliable dataset for analysis.
```python
merged_df2 = pd.concat([titanic_df['Survived'], titanic_df['Alone']], axis=1)
merged_df2 = merged_df2.dropna()
```
```
    Cabin  Survived
0       C         0
1       C         1
2       E         1
3       G         1
4       C         0
..    ...       ...
199     D         0
200     B         0
201     C         0
202     B         0
203     C         0
```
7. By combining the 'Survived' and 'Alone' columns of the Titanic dataset, a new data frame is created. This data frame represents the relationship between passenger survival and their status of traveling alone or with family. The 'Alone' column is mapped to distinguish between passengers traveling with family and those traveling alone.
```python
merged_df2 = pd.concat([titanic_df['Survived'], titanic_df['Alone']], axis=1)
Alone_mapping = {'With Family': 'With Family', 'Alone': 'Without Family'}
```
```
     Survived        Alone
0           0  With Family
1           1  With Family
2           1        Alone
3           1  With Family
4           0        Alone
..        ...          ...
886         0        Alone
887         1        Alone
888         0  With Family
889         1        Alone
890         0        Alone
```

**Part 4: Exploratory Data Analysis**
--------------------
Perform exploratory data analysis to answer the questions and gain insights into the dataset. Use various plots, including bar plots, count plots, histograms, and kernel density estimation (KDE) plots.

**Answers to Questions** 
1. Who were the passengers on the Titanic? (Ages, Gender, Class, etc.)
```python
sns.set(style="whitegrid", palette="Paired")
sns.catplot(x='Sex',data=titanic_df, kind='count')
plt.title("Distribution of Passengers by Sex - Male VS. Female")
```
```python
male_count = titanic_df[titanic_df['Sex'] == 'male'].shape[0]
female_count = titanic_df[titanic_df['Sex'] == 'female'].shape[0]
print(f"Number of Males on board: {male_count}")
print(f"Number of Females on board: {female_count}")
```
```
Number of Males on board: 577
Number of Females on board: 314
```
   ![Distribution of Passengers by Sex](https://github.com/drostark/Titanic-Data-Analysis/blob/948ef4049fdb90edca24008ecb3d579683ab0d09/Images/01_distribution_of_passengers_by_sex.png)

   ```python
   titanic_df['Age'].hist(bins=70,edgecolor='white')
   plt.title("Distribution of Ages")
   plt.xlabel("Age")
   plt.ylabel("Count")
   average_age = round(titanic_df['Age'].mean(),1)
   plt.text(average_age + 1, 30, f"Average Age: {average_age}", fontsize=16, color='red')
   ```
   ![Distribution of Passengers by Age](https://github.com/drostark/Titanic-Data-Analysis/blob/948ef4049fdb90edca24008ecb3d579683ab0d09/Images/03_distribution_of_ages.png)

In accordance with the information provided in the earlier explanation within the Process section, an additional column was created        for this plot to distinguish individuals classified as children.
```python
sns.set(style="whitegrid", palette="Paired")
sns.catplot(x='Pclass',data=titanic_df, kind='count', hue='Sex')
plt.title("Distribution of Passengers - Genders by Classes")
```
```python
class_person_counts = titanic_df.groupby(['Pclass', 'person']).size().unstack(fill_value=0)
for pclass, row in class_person_counts.iterrows():
    male_count = row['male']
    female_count = row['female']
    child_count = row['child']
    print(f"Passenger Class {pclass}: Male: {male_count}, Female: {female_count}, Child: {child_count}")
```
```
Passenger Class 1: Male: 119, Female: 91, Child: 6
Passenger Class 2: Male: 99, Female: 66, Child: 19
Passenger Class 3: Male: 319, Female: 114, Child: 58
```
   ![Distribution of Passengers by Gender and Class](https://github.com/drostark/Titanic-Data-Analysis/blob/948ef4049fdb90edca24008ecb3d579683ab0d09/Images/02_distribution_of_genders_and_children_by_passenger_class.png)

Additionally, I analyzed the distribution of gender and age across various age groups by utilizing a KDE plot. This visualization provides insights into the density of gender and age within specific age intervals.
```python
fig = sns.FacetGrid(titanic_df,hue='person',aspect=4, palette="winter")
fig.map(sns.kdeplot,'Age',fill=True)
plt.title("Kernel Density Estimation (KDE) of Gender and Age Distribution in Different Age Groups")
oldest = titanic_df['Age'].max()
fig.set(xlim=(0,oldest))
fig.add_legend()
```
   ![KDE Gender and Age Distribution in Different Age Groups](https://github.com/drostark/Titanic-Data-Analysis/blob/ed5d94855e683dc77983e3646e7264b184336e50/Images/04_kernel_density_estimation_of_gender_and_age_distribution_in_different_age_groups.png)

In order to observe the age distribution across different passenger classes, a Kernel Density Estimation (KDE) plot was generated. This plot offers a smoothed estimate of the probability density function for each passenger class, enabling us to identify any discrepancies or patterns in the age distribution.
```python
fig = sns.FacetGrid(titanic_df, hue='Pclass', aspect=4, palette="winter")
fig.map(sns.kdeplot, 'Age', fill=True)
plt.title("Kernel Density Estimation (KDE) of Age Distribution in Different Passenger Classes")
fig.set(xlim=(0, oldest))
fig.add_legend(title='Passenger Class')
```
   ![KDE Age Distribution in Different Passenger Classes](https://github.com/drostark/Titanic-Data-Analysis/blob/948ef4049fdb90edca24008ecb3d579683ab0d09/Images/02_kernel_density_estimation_of_age_distribution_in_different_passenger_classes.png)

2. What deck were the passengers on, and how does that relate to their class?

```python
levels = []
for level in deck:
    if level[0] == 'T':
        levels.append(level[1:])
    else:
        levels.append(level[0])

cabin_df = DataFrame(levels, columns=['Cabin'])
cabin_df['Cabin'] = cabin_df['Cabin'].astype('category')
cabin_df = cabin_df[cabin_df['Cabin'] != '']
cabin_df = cabin_df.dropna()
cabin_df_columns = ['Cabin']
cabin_df['Cabin'] = cabin_df['Cabin'].cat.remove_categories('')


palette = sns.color_palette("winter_d", len(cabin_df['Cabin'].cat.categories))[::-1]
sns.catplot(x='Cabin', data=cabin_df, order=sorted(cabin_df['Cabin'].cat.categories), 
            palette=palette, kind='count')
plt.title("Passenger Distribution by Cabin Deck and Class")
```
```python
class_counts = cabin_df['Cabin'].value_counts().sort_index()
for cabin_class, count in class_counts.items():
    print(f"Passengers in Cabin Class {cabin_class}: {count}")
```
```
Passengers in Cabin Class A: 15
Passengers in Cabin Class B: 47
Passengers in Cabin Class C: 59
Passengers in Cabin Class D: 33
Passengers in Cabin Class E: 32
Passengers in Cabin Class F: 13
Passengers in Cabin Class G: 4
```
   ![Distribution of Passengers by Deck and Class](https://github.com/drostark/Titanic-Data-Analysis/blob/948ef4049fdb90edca24008ecb3d579683ab0d09/Images/02_kernel_density_estimation_of_age_distribution_in_different_passenger_classes.png)

3. Where did the passengers come from?

```python
city_mapping = {'C': 'Cherbourg', 'Q': 'Queenstown', 'S': 'Southampton'}
sns.catplot(x='Embarked', data=titanic_df, hue='Pclass', kind='count',palette='winter', order=['C', 'Q', 'S'])
plt.xticks(range(len(city_mapping)), city_mapping.values())
plt.title("Embarked Passengers Cities by Class")
```
```python
passenger_counts = titanic_df.groupby(['Embarked', 'Pclass']).size().unstack()
for city_code, city_name in city_mapping.items():
    total_count = passenger_counts.loc[city_code].sum()
    class_counts = ', '.join([f"Class {pclass}: {count}" for pclass, count in passenger_counts.loc[city_code].items()])
    print(f"Embarked from {city_name}: Total: {total_count}, Counts: {class_counts}")
```
```
Embarked from Cherbourg: Total: 168, Counts: Class 1: 85, Class 2: 17, Class 3: 66
Embarked from Queenstown: Total: 77, Counts: Class 1: 2, Class 2: 3, Class 3: 72
Embarked from Southampton: Total: 644, Counts: Class 1: 127, Class 2: 164, Class 3: 353
```
   ![Distribution of Passengers by Port of Embarkation and Class](https://github.com/drostark/Titanic-Data-Analysis/blob/948ef4049fdb90edca24008ecb3d579683ab0d09/Images/01_embarked_passengers_cities_by_class.png)


4. Who was alone, and who was with family?

```python
titanic_df['Alone'] = titanic_df.SibSp + titanic_df.Parch
titanic_df['Alone'].loc[titanic_df['Alone']> 0] = 'With Family'
titanic_df['Alone'].loc[titanic_df['Alone'] == 0] = 'Alone'
sns.catplot(x='Alone',data=titanic_df, palette='Blues', kind='count',order=['Alone','With Family'])
plt.title("Passenger Count: Alone vs. With Family")
```
```python
alone_counts = titanic_df['Alone'].value_counts()
for category, count in alone_counts.items():
    print(f"Passengers {category}: {count}")
```
```
Passengers Alone: 537
Passengers With Family: 354
```
   ![Count of Passengers Alone vs. with Family](https://github.com/drostark/Titanic-Data-Analysis/blob/948ef4049fdb90edca24008ecb3d579683ab0d09/Images/08_alone_vs_with_family.png)

5. What factors helped someone survive the sinking?
```python
titanic_df['Survivor'] = titanic_df.Survived.map({0:'No',1:'Yes'})
sns.catplot(x='Survivor',data=titanic_df, palette='Set1', kind='count')
plt.title("Passenger Survival Count")
```
```python
survivor_counts = titanic_df['Survivor'].value_counts()
print(f"Number of Survivors: {survivor_counts['Yes']}")
print(f"Number of Died: {survivor_counts['No']}")
```
```
Number of Survivors: 342
Number of Died: 549
```
   ![Count of Survivors](https://github.com/drostark/Titanic-Data-Analysis/blob/948ef4049fdb90edca24008ecb3d579683ab0d09/Images/08_survivor_yes_no.png)
     
```python
sns.lineplot(x='Pclass', y='Survived', data=titanic_df, palette='winter', marker='o')
plt.xticks(np.arange(1, 4, 1), ['1', '2', '3'])
plt.gca().set_yticklabels(['{:.0f}%'.format(x * 100) for x in plt.gca().get_yticks()])
plt.title('Survival Rate by Passenger Class')
```
```python
survival_rates = titanic_df.groupby('Pclass')['Survived'].mean()
for pclass, survival_rate in survival_rates.items():
    print(f"Survival Rate for Class {pclass}: {survival_rate:.2%}")
```
```
Survival Rate for Class 1: 62.96%
Survival Rate for Class 2: 47.28%
Survival Rate for Class 3: 24.24%
```
   ![Survival Rate by Passenger Class](https://github.com/drostark/Titanic-Data-Analysis/blob/948ef4049fdb90edca24008ecb3d579683ab0d09/Images/08_survival_rate_by_age_and_passenger_class.png)
```python
sns.lineplot(x='Pclass', y='Survived',hue='person', data=titanic_df, palette='winter', marker='o')
plt.xticks(np.arange(1, 4, 1), ['1', '2', '3'])
plt.gca().set_yticklabels(['{:.0f}%'.format(x * 100) for x in plt.gca().get_yticks()])
plt.title('Survival Rate by Passenger Class and Gender')
```
```python
for pclass in survival_rates.index.levels[0]:
    child_rate = survival_rates.get((pclass, 'child'), 0)
    female_rate = survival_rates.get((pclass, 'female'), 0)
    male_rate = survival_rates.get((pclass, 'male'), 0)
    print(f"Survival Rate for Class {pclass}: child: {child_rate:.2%}, female: {female_rate:.2%}, male: {male_rate:.2%}")
```
```
Survival Rate for Class 1: child: 83.33%, female: 97.80%, male: 35.29%
Survival Rate for Class 2: child: 100.00%, female: 90.91%, male: 8.08%
Survival Rate for Class 3: child: 43.10%, female: 49.12%, male: 11.91%
```
   ![Survival Rate by Passenger Class and gender](https://github.com/drostark/Titanic-Data-Analysis/blob/948ef4049fdb90edca24008ecb3d579683ab0d09/Images/08_survived_based_on_passenger_class_and_gender.png)

```python
sns.lmplot(x='Age',y='Survived',data=titanic_df,palette='winter')
plt.gca().set_yticklabels(['{:.0f}%'.format(x * 100) for x in plt.gca().get_yticks()])
plt.title('Survival Rate by Age')
```
```python
age_survival_rate = titanic_df.groupby('Age')['Survived'].mean()
age_decline_rate = age_survival_rate.pct_change() * 100
age_decline_rate = age_decline_rate.replace([np.inf, -np.inf], np.nan).dropna()
average_decline = age_decline_rate.mean()
print(f"Average Decline in Survival Rate: {average_decline:.2f}%")
```
```
Average Decline in Survival Rate: -5.80%
```
   ![Survival Rate by Age](https://github.com/drostark/Titanic-Data-Analysis/blob/948ef4049fdb90edca24008ecb3d579683ab0d09/Images/08_survival_rate_by_age.png)

```python
generations = [10,20,40,60,80]
sns.lmplot(x='Age',y='Survived', hue='Sex', data=titanic_df, palette='winter',x_bins=generations)
plt.gca().set_yticklabels(['{:.0f}%'.format(x * 100) for x in plt.gca().get_yticks()])
plt.title('Survival Rate by Age and Sex')
```
   ![Survival Rate by Sex and Age](https://github.com/drostark/Titanic-Data-Analysis/blob/948ef4049fdb90edca24008ecb3d579683ab0d09/Images/08_age_and_survival_rates_base_on_sex.png)
```python
sns.lmplot(x='Age',y='Survived', hue='Pclass', data=titanic_df, palette='winter',x_bins=generations)
plt.gca().set_yticklabels(['{:.0f}%'.format(x * 100) for x in plt.gca().get_yticks()])
plt.title('Survival Rate by Age and Passenger Class')
```
   ![Survival Rate by Age and Passenger Class](https://github.com/drostark/Titanic-Data-Analysis/blob/948ef4049fdb90edca24008ecb3d579683ab0d09/Images/08_survival_rate_by_age_and_passenger_class.png)

6. Did the deck have an effect on the passengers' survival rate?
```python
merged_df = pd.concat([cabin_df['Cabin'], titanic_df['Survived']], axis=1)
merged_df = merged_df.dropna()
sns.barplot(x='Cabin', y='Survived', data=merged_df, palette='winter')
plt.gca().set_yticklabels(['{:.0f}%'.format(x * 100) for x in plt.gca().get_yticks()])
plt.title('Survival Rate by Cabin')
```
   ![Survival Count by Deck and Class](https://github.com/drostark/Titanic-Data-Analysis/blob/948ef4049fdb90edca24008ecb3d579683ab0d09/Images/06_survival_rate_by_cabin.png)

7. Did having a family member increase the odds of surviving the crash?
```python
merged_df2 = pd.concat([titanic_df['Survived'], titanic_df['Alone']], axis=1)
Alone_mapping = {'With Family': 'With Family', 'Alone': 'Without Family'}
sns.barplot(x='Alone', y='Survived', data=merged_df2, palette='winter')
```
   ![Survival Rate by Presence of Family](https://github.com/drostark/Titanic-Data-Analysis/blob/948ef4049fdb90edca24008ecb3d579683ab0d09/Images/07_survival_rate_with_or_without_family.png)

**Part 5: Conclusion**
--------------------
Throughout this analysis, we have made significant progress, starting from data comprehension to delving deeper into the factors influencing survival rates in the Titanic crash.

Based on our analysis, the following insights have emerged:

1. The majority of passengers onboard were male, yet female passengers had a higher likelihood of surviving the crash.
2. Among female passengers, those in their 20s, traveling in passenger class 1 and staying in cabins A or B had the highest chances of survival.
3. Conversely, male passengers who were traveling alone had a lower probability of surviving the crash.

Preparing this analysis has been an enriching experience that has significantly contributed to my professional development. Writing the article has not only deepened my understanding of the data but also provided an opportunity to refine my analytical skills and storytelling abilities. By structuring my thoughts and effectively communicating the insights, I have honed my report-writing capabilities. This project has served as a reminder of the fundamental principles of analysis and has motivated me to pursue further projects and continue my growth as a data analyst.

Moreover, I would like to mention that this analysis is an ongoing endeavor. Currently, I am utilizing SciKit to construct an entropy decision tree, which aims to predict the last survivors based on the provided dataset. I am excited about the potential insights this approach can offer. __Thank you for your time, and have a wonderful day!__
