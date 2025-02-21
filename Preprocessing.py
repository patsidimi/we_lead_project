# -*- coding: utf-8 -*-

#εισάγω τις απαραίτητες βιβλιοθήκες
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings("ignore")

# Φόρτωση δεδομένων
df = pd.read_excel("/content/mpg.data.xlsx")

#drop unnamed empty columns
df=df.drop(columns=['Unnamed: 9'])
df=df.drop(columns=['Unnamed: 10'])
df=df.drop(columns=["Unnamed: 11"])
df=df.drop(columns=['Unnamed: 12'])

#rename displayments to displacemet
df=df.rename(columns={"displayments":"displacement"})

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

"""Από EDA είδαμε ότι υπάρχουν outliers στα χαρακτηριστικά:mpg, acceleration και horsepower"""

find_outliers(df, 'acceleration')

find_outliers(df, 'horsepower')

find_outliers(df, 'mpg')

def cap_outliers(df, column):
        """

    Περικοπή (capping) των ακραίων τιμών (outliers) μιας στήλης με βάση το IQR.

    Parameters
    ----------
    df : pandas.DataFrame  Το DataFrame που περιέχει τα δεδομένα
    column : str           Το όνομα της στήλης στην οποία θα εφαρμοστεί η περικοπή

    Returns
    -------
    pandas.DataFrame       Νέο DataFrame με τις ακραίες τιμές περικομμένες στα όρια του IQR

    Notes
    -----
    Η μέθοδος χρησιμοποιεί την τεχνική του Interquartile Range (IQR):
    - Υπολογίζει Q1 (25ο εκατοστημόριο) και Q3 (75ο εκατοστημόριο)
    - Ορίζει ως όρια: Q1 - 1.5*IQR και Q3 + 1.5*IQR
    - Περικόπτει τις τιμές που βρίσκονται εκτός αυτών των ορίων
        """
        df_copy = df.copy()
        Q1 = df[column].quantile(0.25)
        Q3 = df[column].quantile(0.75)
        IQR = Q3 - Q1

        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR

        df_copy[column] = df_copy[column].clip(lower=lower_bound, upper=upper_bound)
        return df_copy

#γίνεται περικοπή των outliers
df_capped = cap_outliers(df, 'acceleration')
df_capped = cap_outliers(df_capped, 'horsepower')
df_capped = cap_outliers(df_capped, 'mpg')
df=df_capped

#εμφανίζει όλες τις γραμμές στις οποίες το χαρακτηριστικό horsepower is null
df[df["horsepower"].isnull()]

#correletion between horsepower and other numerical

numerical_df = df.select_dtypes(include=['number'])
correlations = numerical_df.corr()['horsepower'].sort_values()
print("\nΣυσχέτιση με horsepower:")
print(correlations)

#replace horsepower NaN with the mean of horsepower from the cars with the same displacement

df["horsepower"]=df["horsepower"].fillna(round(df.groupby("displacement")["horsepower"].transform("mean")))

#replace horsepower NaN with the mean of horsepower from the cars with the same cylinders

df["horsepower"]=df["horsepower"].fillna(round(df.groupby("cylinders")["horsepower"].transform("mean")))

#διώχνει τις γραμμές που περιέχουν null στο mpg

df=df.dropna(subset=['mpg'])

# car name unique names, υπάρχουν πολλά διαφορετικά μοντέλο, δεν είναι χρήσιμο αυτό
print("Πλήθος διαφορετικών μοντέλων αυτοκινήτων:", df["car name"].nunique())

#col "brand" from car name col, δημιουργώ νέο χαρακτηριστικό "μάρκα", τραβάω την πρώτη λέξη από κάθε car name
df["brand"] = df["car name"].str.extract("(^.*?)\s")
#εμφανίζει τις μοναδικές τιμές που παίρνει το χαρακτηριστικό brand
print("Μοναδικές τιμές χαρακτηριστικού brand: \n",df["brand"].unique())

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
print("Εμφανίζει τα στιγμιότυπα που έχουν NaN στο χαρακτηριστικό brand: \n",df[df["brand"].isnull()])

# τα δύο που έχουν κενό την μάρκα είναι subaru, το προσθέτω
df["brand"]=df["brand"].fillna("Subaru")
#capitalize brand
df["brand"]=df["brand"].str.capitalize()
#drop car name col
df.drop("car name",axis=1,inplace=True)

# Κατηγοριοποίηση με ιστορικά όρια <15 low, >15 & <25 medium , >25 high που αφορούν την δεκαετία 70-80
df["mpg_category"] = pd.cut(df["mpg"],
                           bins=[0, 15, 25, float('inf')],
                           labels=['0', '1', '2'])
# Βασική ανάλυση
print("Κατανομή κατηγοριών:")
print(df["mpg_category"].value_counts())
print("\nΣτατιστικά ανά κατηγορία:")
print(df.groupby("mpg_category")["mpg"].describe())

#πλήθος διαφορετικών τιμών στο χαρακτηριστικό weight
print("Πλήθος διαφορετικών τιμών στο χαρακτηριστικό weight: \n",df["weight"].nunique())

#Υπάρχουν 351 διαφορετικά βάρη, τα ομαδοποιώ σε ισοπληθείς ομάδες
#5 weight categories 20%
df["weight_category"] = pd.qcut(df["weight"], q=5, labels=["1", "2", "3","4","5"])

print(df["weight_category"].value_counts())

#create a list with unique names of brand
unique_brands = df["brand"].unique()
#create a dictionary , keys will be the unique brands, values : index numbers
dict_brands={item: idx for idx, item in enumerate(unique_brands, start=1)}
#δημιουργία στο dataframe στήλης με τον αριθμό που αντιστοιχεί σε κάθε μάρκα
df['brand_num'] = df['brand'].map(dict_brands)

# Προετοιμασία δεδομένων για κατηγοριοποίηση, διώχνω τις στήλες 'mpg',"weight","brand","origin"
#Την πληροφορία που δίνει το origin τηγ παίρνω από την μάρκα
df = df.drop(['mpg',"weight","brand","origin"], axis=1)

df.head()

# Αποθήκευση
df.to_excel('processed_auto_mpg.xlsx', index=False)
