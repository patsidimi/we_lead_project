# -*- coding: utf-8 -*-

#εισάγω τις απαραίτητες βιβλιοθήκες
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib import colors
from scipy import stats
import warnings
warnings.filterwarnings("ignore")

# Φόρτωση δεδομένων
df = pd.read_excel("/content/mpg.data.xlsx")

print("\nΠληροφορίες για τις στήλες:\n")
print("-" * 50)
print(df.info())

#rename col
df=df.rename(columns={'displayments':'displacement'})

#drop unnamed empty columns
df=df.drop(columns=['Unnamed: 9'])
df=df.drop(columns=['Unnamed: 10'])
df=df.drop(columns=["Unnamed: 11"])
df=df.drop(columns=['Unnamed: 12'])

def basics_of_df(df):
  '''
   Εμφανίζει βασικές πληροφορίες και στατιστικά στοιχεία του DataFrame.

    Parameters
    ----------
    df (pandas.DataFrame): το DataFrame με τα δεδομένα
    ----------
    Η συνάρτηση εκτυπώνει τα εξής στοιχεία:

    1. Εμφανίζει τις διαστάσεις (γραμμές x στήλες) του DataFrame
    2. Παρουσιάζει πληροφορίες για κάθε στήλη (τύπος δεδομένων, μη-κενές τιμές)
    3. Δείχνει τις πρώτες 5 εγγραφές για γρήγορη επισκόπηση
    4. Υπολογίζει βασικά στατιστικά (mean, std, min, max, κλπ.) για αριθμητικές στήλες
    5. Ελέγχει για ελλιπείς τιμές (missing values) σε κάθε στήλη
    6. Εντοπίζει τυχόν διπλότυπες εγγραφές στο DataFrame

  '''
  #μέγεθος dataframe
  print("Διαστάσεις:", df.shape)
  print("-" * 80)
  # information about the dataset
  print("\nΠληροφορίες για τις στήλες:\n")
  print("-" * 80)
  print(df.info())

  #εμφανίζω τις 5 πρώτες γραμμές
  print("\nΠρώτες γραμμές:\n")
  print("-" *80)
  print(df.head())
  #παρουσιάζει βασικά στατιστικά των αριθμητικών χαρακτηριστικών
  print("\nΒασικά Στατιστικά:\n")
  print("-" * 80)
  print(df.describe())

  # Έλεγχος για missing values
  print("\nΈλεγχος για missing values:\n")
  print("-" * 80)
  print(df.isnull().sum())
  #check for duplicated
  print("\nΈλεγχος για διπλότυπα:\n")
  print("-" * 20)
  print(df.duplicated().sum())

basics_of_df(df)

# Ιστόγραμμα για την κατανομή του mpg
sns.histplot(df['mpg'], kde=True)
plt.title("Κατανομή του MPG")
plt.show()

# Boxplot για το mpg
sns.boxplot(x=df['mpg'])
plt.title("Boxplot του MPG")
plt.show()

# Bar plot για την στήλη 'cylinders'
sns.countplot(x='cylinders', data=df)
plt.title("Κατανομή των Cylinders")
plt.show()

# Bar plot για την στήλη 'origin'
sns.countplot(x='origin', data=df)
plt.title("Κατανομή της Origin")
plt.show()

# Bar plot για την στήλη 'model_year'
sns.countplot(x='model year', data=df)
plt.title("Κατανομή του Model Year")
plt.show()

# Συσχέτιση με MPG
numerical_df = df.select_dtypes(include=['number'])
correlations = numerical_df.corr()['mpg'].sort_values()
print("\nΣυσχέτιση με MPG:")
print(correlations)

#Heatmap
sns.heatmap(numerical_df.corr(), annot=True, cmap='Blues_r')
plt.title("Heatmap Συσχέτισης Μεταβλητών")
plt.show()

# Scatter plots για mpg vs. displacement, horsepower, weight, acceleration
sns.pairplot(df, vars=['mpg', 'displacement', 'horsepower', 'weight', 'acceleration'])
plt.show()

# Boxplots για αριθμητικές μεταβλητές
numerical_columns = ['displacement', 'horsepower', 'weight', 'acceleration']
for col in numerical_columns:
    sns.boxplot(x=df[col])
    plt.title(f"Boxplot για {col}")
    plt.show()

def find_outliers(df, column, verbose=True):
    """
    Εντοπίζει και αναλύει τα outliers σε μια στήλη χρησιμοποιώντας τη μέθοδο IQR.

    Parameters
    ----------
    df : pandas.DataFrame       Το DataFrame με τα δεδομένα
    column : str                Το όνομα της στήλης προς έλεγχο
    verbose : bool, optional    Αν True, εκτυπώνει αναλυτικές πληροφορίες (default: True)

    Returns
    -------
    pandas.DataFrame            DataFrame με τις καταχωρήσεις που περιέχουν outliers και επιπλέον στατιστικά

    Notes
    -----
    Χρησιμοποιεί τη μέθοδο Interquartile Range (IQR) για τον εντοπισμό outliers:
    - Lower bound = Q1 - 1.5 * IQR
    - Upper bound = Q3 + 1.5 * IQR


    """
    # Υπολογισμός βασικών στατιστικών
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    median = df[column].median()
    mean = df[column].mean()

    # Ορισμός ορίων για τα outliers
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR

    # Εύρεση των outliers
    outliers = df[
        (df[column] < lower_bound) |
        (df[column] > upper_bound)
    ].copy()  # Χρήση .copy() για αποφυγή warning

    # Προσθήκη επιπλέον πληροφοριών
    outliers = outliers.assign(
        is_lower_outlier = lambda x: x[column] < lower_bound,
        is_upper_outlier = lambda x: x[column] > upper_bound,
        lower_bound = lower_bound,
        upper_bound = upper_bound,
        distance_from_median = lambda x: abs(x[column] - median),
        z_score = lambda x: abs(x[column] - mean) / df[column].std()
    )

    if verbose:
        # Εκτύπωση αναλυτικών πληροφοριών
        print(f"\nΑνάλυση Outliers για τη στήλη '{column}':")
        print("-" * 50)
        print(f"Συνολικές εγγραφές: {len(df)}")
        print(f"Αριθμός outliers: {len(outliers)} ({(len(outliers)/len(df)*100):.2f}%)")
        print(f"Lower outliers: {sum(outliers['is_lower_outlier'])} ({(sum(outliers['is_lower_outlier'])/len(df)*100):.2f}%)")
        print(f"Upper outliers: {sum(outliers['is_upper_outlier'])} ({(sum(outliers['is_upper_outlier'])/len(df)*100):.2f}%)")
        print("\nΣτατιστικά στοιχεία:")
        print(f"Median: {median:.2f}")
        print(f"Mean: {mean:.2f}")
        print(f"Q1: {Q1:.2f}")
        print(f"Q3: {Q3:.2f}")
        print(f"IQR: {IQR:.2f}")
        print(f"Lower bound: {lower_bound:.2f}")
        print(f"Upper bound: {upper_bound:.2f}")

        if len(outliers) > 0:
            print("\nΣτατιστικά outliers:")
            print(f"Min outlier value: {outliers[column].min():.2f}")
            print(f"Max outlier value: {outliers[column].max():.2f}")
            print(f"Mean distance from median: {outliers['distance_from_median'].mean():.2f}")
            print(f"Mean z-score: {outliers['z_score'].mean():.2f}")

    return outliers.sort_values(by='z_score', ascending=False)

find_outliers(df, 'acceleration')

find_outliers(df, 'horsepower')

find_outliers(df, 'mpg')

# Scatter plot: mpg vs. weight
sns.scatterplot(x='weight', y='mpg', data=df)
plt.title("MPG vs. Weight")
plt.show()

# Scatter plot: mpg vs. Displacement
sns.scatterplot(x='displacement', y='mpg', data=df)
plt.title("MPG vs. Displacement")
plt.show()

# Scatter plot: mpg vs. Cylinders
sns.scatterplot(x='cylinders', y='mpg', data=df)
plt.title("MPG vs. Cylinders")
plt.show()

# Scatter plot: mpg vs. horsepower
sns.scatterplot(x='horsepower', y='mpg', data=df)
plt.title("MPG vs. Horsepower")
plt.show()

# Κατηγοριοποίηση με ιστορικά όρια <15 low, >15 & <25 medium , >25 high που αφορούν την δεκαετία 70-80
df["mpg_category"] = pd.cut(df["mpg"],
                           bins=[0, 15, 25, float('inf')],
                           labels=['low', 'medium', 'high'])
# Βασική ανάλυση
print("Κατανομή κατηγοριών:")
print(df["mpg_category"].value_counts())
print("\nΣτατιστικά ανά κατηγορία:")
print(df.groupby("mpg_category")["mpg"].describe())

# Bar plot για την κατανομή των κατηγοριών mpg
df['mpg_category'] = pd.cut(df['mpg'], bins=[0, 15, 25, float('inf')], labels=['low', 'medium', 'high'])
sns.countplot(x='mpg_category', data=df)
plt.title("Κατανομή των Κατηγοριών MPG")
plt.show()

#scatterplots
fig, axes = plt.subplots(2, 2, figsize=(15, 10))
sns.scatterplot(data=df, x='weight', y='mpg', ax=axes[0,0],hue='mpg_category')
sns.scatterplot(data=df, x='horsepower', y='mpg', ax=axes[0,1],hue='mpg_category')
sns.scatterplot(data=df, x='displacement', y='mpg', ax=axes[1,0],hue='mpg_category')
sns.scatterplot(data=df, x='acceleration', y='mpg', ax=axes[1,1],hue='mpg_category')
plt.tight_layout()
plt.show()

#πλήθος διαφορετικών τιμών στο χαρακτηριστικό weight
print("Το πλήθος διαφορετικών τιμών στο χαρακτηριστικό weight : ", df["weight"].nunique())

#Υπάρχουν 351 διαφορετικά βάρη, τα ομαδοποιώ σε ισοπληθείς ομάδες
#5 weight categories 20%
df["weight_category"] = pd.qcut(df["weight"], q=5, labels=["1", "2", "3","4","5"])

print(df["weight_category"].value_counts())

sns.countplot(x='weight_category', data=df)
plt.title("Κατανομή των Κατηγοριών βάρους")
plt.show()

# Stacked bar plot για MPG categories ανά weight category
colors = {'low': 'darkblue', 'medium': 'royalblue', 'high': 'lightsteelblue'}
plt.figure(figsize=(12, 6))
pd.crosstab(df['weight_category'], df['mpg_category']).plot(
    kind='bar',
    stacked=True,
    color=colors
)
plt.title('MPG Categories Distribution by Weight Category')
plt.xlabel('Weight Category')
plt.legend(title='MPG Category')
plt.xticks(rotation=45)
plt.grid(True, alpha=0.3)
plt.show()

plt.figure(figsize=(10, 6))
crosstab = pd.crosstab(df['mpg_category'], df['weight_category'])
sns.heatmap(crosstab, annot=True, fmt='d',  cmap='Blues_r')
plt.title('Heatmap: MPG Category vs Weight Category')
plt.xlabel('Weight Category')
plt.ylabel('MPG Category')
plt.show()

#αφαιρώ από την στήλη mpg τα στιγμιότυπα που είναι NaN
df=df.dropna(subset=['mpg'])

# Violin plot για MPG ανά κατηγορία βάρους
plt.figure(figsize=(10,6))
sns.violinplot(data=df, x='weight_category', y='mpg')
plt.title('Κατανομή MPG ανά Κατηγορία Βάρους')
plt.xticks(rotation=45)
plt.show()

"""Το violin plot δείχνει ότι τα αυτοκίνητα με μικρότερο βάρος τείνουν να έχουν υψηλότερο MPG, με μικρότερη διασπορά. Αντίθετα, τα βαρύτερα αυτοκίνητα έχουν χαμηλότερο MPG με μεγαλύτερη διασπορά. Η κατανομή στις μεσαίες κατηγορίες βάρους φαίνεται να είναι πιο συμμετρική.

"""

#Faceted histogram για MPG ανά weight_category
g = sns.FacetGrid(df, col="weight_category", col_wrap=3, height=4)
g.map(plt.hist, "mpg", bins=20, color='#1f77b4')
g.fig.suptitle('MPG Distribution by Weight Category', y=1.05)
plt.show()

# car name unique names, υπάρχουν πολλά διαφορετικά μοντέλο, δεν είναι χρήσιμο αυτό
print("Το πλήθος των διαφορετικών τιμών που παίρνει το χαρακτηριστικό car name: ", df["car name"].nunique())

#col "brand" from car name col, δημιουργώ νέο χαρακτηριστικό "μάρκα", τραβάω την πρώτη λέξη από κάθε car name
df["brand"] = df["car name"].str.extract("(^.*?)\s")

#εμφανίζει τις μοναδικές τιμές που παίρνει το χαρακτηριστικό brand
print("Οι διαφορετικές τιμές που παίρνει το χαρακτηριστικό car name: ",df["brand"].unique())

#κάποιες τιμές αφορούν την ίδια μάρκα αλλά είναι γραμμένες με διαφορετικό τρόπο, το διορθώνουμε:
df["brand"] = df["brand"].replace(to_replace="maxda",value="mazda")
df["brand"] = df["brand"].replace(to_replace="toyouta",value="toyota")
df["brand"] = df["brand"].replace(to_replace="mercedes-benz",value="mercedes")
df["brand"] = df["brand"].replace(to_replace=["chevroelt","chevy"],value="chevrolet")
df["brand"] = df["brand"].replace(to_replace=["volkswagen","vokswagen","vw"],value="VW")
df["brand"] = df["brand"].replace(to_replace="porcshce",value="porsche")
df["brand"] = df["brand"].replace(to_replace="capri",value="ford")
df["brand"] = df["brand"].replace(to_replace="datsun",value="Nissan")


#check for nulls
print ("\n Έλεγχος αν υπάρχουν NaN στην στήλη brand:\n ", df["brand"].isnull().sum())
print("\n Οι γραμμές που περιέχουν NaN  στην στήλη brand: \n", df[df["brand"].isnull()])
# τα δύο που έχουν κενό την μάρκα είναι subaru, το προσθέτω
df["brand"]=df["brand"].fillna("Subaru")


#capitalize brand
df["brand"]=df["brand"].str.capitalize()
#drop car name col
df.drop("car name",axis=1,inplace=True)

# Κατανομή μαρκών
plt.figure(figsize=(12,6))
sns.countplot(data=df, x='brand', order=df['brand'].value_counts().index)
plt.title('Συχνότητα Εμφάνισης Μαρκών')
plt.xticks(rotation=45)
plt.show()

# MPG ανά μάρκα
plt.figure(figsize=(12,6))
sns.boxplot(data=df, x='brand', y='mpg')
plt.title('Κατανομή MPG ανά Μάρκα')
plt.xticks(rotation=45)
plt.show()

# Average MPG ανά μάρκα
plt.figure(figsize=(20,8))
plt.title("Average mpg values of each brand",fontsize=25)
plt.xticks(rotation=45, horizontalalignment='right',fontweight='light',fontsize='x-large')


sns.barplot(x="brand",y="mpg",data=df,palette="pastel")

# Υπολογισμός μέσου MPG ανά μάρκα
mean_mpg_by_brand = df.groupby('brand')['mpg'].mean().sort_values()
top_5_brands = mean_mpg_by_brand.tail(5).index
bottom_5_brands = mean_mpg_by_brand.head(5).index

# Συνδυασμός των top και bottom 5 μαρκών
selected_brands = list(bottom_5_brands) + list(top_5_brands)

# Ανάλυση ανά μάρκα (Box Plot)
plt.figure(figsize=(12,6))
sns.boxplot(data=df[df['brand'].isin(selected_brands)],
            x='brand',
            y='mpg',
            order=selected_brands)
plt.title('Κατανομή MPG για τις 5 καλύτερες και 5 χειρότερες μάρκες')
plt.xticks(rotation=45)
plt.axhline(y=df['mpg'].mean(), color='r', linestyle='--', label='Μέσο MPG')
plt.legend()
plt.show()

# Σύνθετη Ανάλυση (Stacked Bar Plot)

plt.figure(figsize=(12,6))
crosstab = pd.crosstab(df['brand'], df['mpg_category'])
crosstab.loc[selected_brands].plot(kind='bar', stacked=True,color=colors)
plt.title('Κατανομή Κατηγοριών MPG για τις 5 καλύτερες και 5 χειρότερες μάρκες')
plt.xticks(rotation=45)
plt.legend(title='MPG Category')
plt.show()

# Προαιρετικά: Εμφάνιση των ακριβών τιμών
print("\nTop 5 μάρκες με το υψηλότερο μέσο MPG:")
print(mean_mpg_by_brand.tail().round(2))
print("\nBottom 5 μάρκες με το χαμηλότερο μέσο MPG:")
print(mean_mpg_by_brand.head().round(2))

import plotly.express as px

# Βρίσκουμε τις top 5 brands με βάση το μέσο MPG
top_5_brands = df.groupby('brand')['mpg'].mean().sort_values(ascending=False).head(5).index.tolist()

# Φιλτράρουμε το dataframe για να περιέχει μόνο αυτές τις brands
df_filtered = df[df['brand'].isin(top_5_brands)]

# Υπολογίζουμε τα ποσοστά για κάθε κατηγορία MPG
df_sunburst = df_filtered.groupby(['brand', 'mpg_category']).size().reset_index(name='count')
df_sunburst['percentage'] = df_sunburst.groupby('brand')['count'].transform(lambda x: x / x.sum() * 100)

fig = px.sunburst(df_sunburst,
                  path=['brand', 'mpg_category'],
                  values='percentage',
                  title='Hierarchical View of Top 5 Brands by MPG Category Distribution (%)')

fig.show()

# Εξέλιξη MPG στο χρόνο
sns.lineplot(data=df, x='model year', y='mpg')
plt.title('Εξέλιξη MPG στο Χρόνο')
plt.show()

# display model year against mpg with respect to origin
plt.figure(figsize=(10,5))
sns.pointplot(x='model year', y='mpg', hue='origin', data=df, ci=None,
              palette='Set1');
plt.title("Model year against mpg with respect to origin", fontsize = 20)
plt.xlabel("model year", fontsize = 15)
plt.ylabel("mpg", fontsize = 15)
plt.legend(title='origin', labels=['USA', 'EUROPE', 'ASIA'])
plt.show()

# Δημιουργία figure με δύο υποπλαίσια
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

# Box plot στο πρώτο υποπλαίσιο
sns.boxplot(data=df, x='origin', y='mpg', ax=ax1, palette='Set2')
ax1.set_title('Box Plot: MPG ανά Προέλευση')
ax1.set_xlabel('Προέλευση')
ax1.set_ylabel('Miles Per Gallon (MPG)')
ax1.set_xticklabels(['USA', 'EUROPE', 'ASIA'])

# Stacked bar στο δεύτερο υποπλαίσιο
crosstab = pd.crosstab(df['origin'], df['mpg_category'], normalize='index') * 100
crosstab.plot(kind='bar', stacked=True, ax=ax2,color=colors)
ax2.set_title('Κατανομή MPG Categories ανά Προέλευση')
ax2.set_xlabel('Προέλευση')
ax2.set_ylabel('Ποσοστό (%)')
ax2.legend(title='MPG Category')
ax2.tick_params(axis='x', rotation=0)
ax2.set_xticklabels(['USA', 'EUROPE', 'ASIA'])
plt.tight_layout()
plt.show()

# Προσθήκη στατιστικών πληροφοριών
print("\nΣτατιστικά στοιχεία MPG ανά προέλευση:")
print(df.groupby('origin')['mpg'].describe().round(2))

# Δημιουργούμε το crosstab
crosstab = pd.crosstab(df['origin'], df['weight_category'], normalize='index') * 100

# Με συγκεκριμένα χρώματα
colors = ['lightsteelblue',"cornflowerblue", 'royalblue', "blue",'darkblue']
# Δημιουργία του γραφήματος
plt.figure(figsize=(10, 6))
crosstab.plot(kind='bar', stacked=True,color=colors)


# Προσαρμογή των ετικετών στον άξονα x
plt.xticks([0, 1, 2], ['USA', 'EUROPE', 'ASIA'], rotation=0)

# Προσθήκη τίτλων και ετικετών
plt.title('Κατανομή Weight Categories ανά Προέλευση', fontsize=12)
plt.xlabel('Προέλευση', fontsize=10)
plt.ylabel('Ποσοστό (%)', fontsize=10)
plt.legend(title='Weight Category')


# Προσαρμογή του layout
plt.tight_layout()
plt.legend(title='Weight Category', bbox_to_anchor=(1, 0), loc='lower right')
# Εμφάνιση του γραφήματος
plt.show()

"""##**Συμπεράσματα από EDA**

Ελλιπείς τιμές:

8 στην στήλη mpg και
6 στην στήλη horsepower    


Outliers

6 στην στήλη  acceleration, 8 στην στήλη horsepower και 1 στην στήλη mpg

Χρειάστηκε να γίνει κατηγοριοποίηση σε κατηγορίες βάρους και κατηγοριοποίηση με βάση το mpg (Κατανομή κατηγοριών:mpg_category medium    171 high      158
low        69), η κατανομή είναι άνιση όμως βασίζεται σε πραγματικά δεδομένα.

Ανακτήσαμε τις μάρκες των αυτοκινήτων από τα μοντέλα

Φαίνεται να υπάρχει ισχυρή αρνητική συσχέτιση του mpg με το βάρος (-0.83), ο κυβισμός (-0.80), την ιπποδύναμη (-0.78) και τους κυλίνδρους (-0.78)


Το violin plot δείχνει ότι τα αυτοκίνητα με μικρότερο βάρος τείνουν να έχουν υψηλότερο MPG, αντίθετα, τα βαρύτερα αυτοκίνητα έχουν χαμηλότερο MPG. Η κατανομή στις μεσαίες κατηγορίες βάρους φαίνεται να είναι πιο συμμετρική.

Top 5 μάρκες με το υψηλότερο μέσο MPG:
brand
Nissan     31.32
Vw         31.84
Renault    32.88
Honda      33.76
Triumph    35.00


Bottom 5 μάρκες με το χαμηλότερο μέσο MPG:
brand
Hi           9.00
Chrysler    17.27
Amc         18.25
Mercury     19.12
Buick       19.18

Τα οχήματα που προέρχονται από την περιοχή 1(usa) έχουν πιο χαμηλό mpg
συγκριτικά με τις άλλες δύο κατηγορίες, με την τρίτη περιοχή(Asia) να έχει τα περισσότερα αυτοκίνητά της στην κατηγορία high mpg.

Η εξέλιξη της κατανάλωσης στον χρόνο: φαίνεται ότι όσο περνάνε τα χρόνια υπάρχει μια αυξητική τάση στην μέση τιμή του mpg, αυτό οφείλεται στις νέες τεχνολογίες και στην προσπάθεια για οικονομία καυσίμων λόγω περιβαλλοντικών και οικονομικών λόγων.
"""

