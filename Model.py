#εισάγω τις απαραίτητες βιβλιοθήκες
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib import colors
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
from collections import Counter
from sklearn.model_selection import train_test_split, StratifiedKFold,cross_val_score
from sklearn.metrics import (classification_report, confusion_matrix,
                           accuracy_score, precision_score, recall_score,
                           f1_score)
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.base import clone
from collections import Counter


import warnings
warnings.filterwarnings("ignore")

# Φόρτωση επεξεργασμένων δεδομένων
df = pd.read_excel("/content/processed_auto_mpg.xlsx")

df.head()

#Διαχωρισμός δεδομένων dummies
x = df.drop(["mpg_category"], axis = 1)
y = df["mpg_category"]

def compare_classifiers_with_smote(X, y, random_state=42):
    """
    Συγκρίνει διάφορους αλγόριθμους ταξινόμησης χρησιμοποιώντας SMOTE για εξισορρόπηση των κλάσεων.

    Parameters
    ----------
    X : pandas.DataFrame
        Τα χαρακτηριστικά (features) του συνόλου δεδομένων.
    y : pandas.Series
        Οι ετικέτες (labels) του συνόλου δεδομένων.
    random_state : int, optional (default=42)
        Σταθερά για αναπαραγωγή των αποτελεσμάτων.

    Returns
    -------
    dict
        Λεξικό με τα αποτελέσματα για κάθε ταξινομητή που περιέχει:
        - 'cv_results' : dict
            Αποτελέσματα cross-validation για κάθε μετρική.
        - 'classification_report' : str
            Αναλυτική αναφορά ταξινόμησης.
        - 'confusion_matrix' : array
            Πίνακας σύγχυσης.
        - 'model' : object
            Το εκπαιδευμένο μοντέλο.
        - 'f1_per_class' : array
            F1-scores για κάθε κλάση.

    Notes
    -----
    Η συνάρτηση εκτελεί τα εξής βήματα:
    1. Διαχωρίζει τα δεδομένα σε train και test sets (80-20)
    2. Εφαρμόζει StandardScaler για κανονικοποίηση
    3. Χρησιμοποιεί SMOTE για εξισορρόπηση των κλάσεων
    4. Εκτελεί 5-fold cross-validation
    5. Συγκρίνει τους ακόλουθους ταξινομητές:
       - Logistic Regression
       - Decision Tree
       - Random Forest
       - SVM
       - KNN
    6. Υπολογίζει τις ακόλουθες μετρικές:
       - Accuracy
       - Precision (macro)
       - Recall (macro)
       - F1-score (macro και weighted)
       - F1-score ανά κλάση
    """

    # Διαχωρισμός δεδομένων
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=random_state, stratify=y
    )
    print("Κατανομή κλάσεων πριν το SMOTE:")
    print(Counter(y_train))

    # Ορισμός scoring metrics για cross validation
    scoring = {
        'accuracy': 'accuracy',
        'precision_macro': 'precision_macro',
        'recall_macro': 'recall_macro',
        'f1_macro': 'f1_macro',
        'f1_weighted': 'f1_weighted'
    }

    # Αρχικοποίηση των classifiers
    classifiers = {
        'Logistic Regression': LogisticRegression(random_state=random_state, max_iter=1000),
        'Decision Tree': DecisionTreeClassifier(random_state=random_state),
        'Random Forest': RandomForestClassifier(random_state=random_state),
        'SVM': SVC(random_state=random_state),
        'KNN': KNeighborsClassifier()
    }

    results = {}

    for name, clf in classifiers.items():
        print(f"\nΕκπαίδευση {name}...")

        # Cross-validation με SMOTE
        cv_scores = []
        f1_scores_per_class = []

        # Δημιουργία του StratifiedKFold
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=random_state)

        for train_idx, val_idx in skf.split(X_train, y_train):
            # Split the data
            X_fold_train = X_train.iloc[train_idx]
            y_fold_train = y_train.iloc[train_idx]
            X_fold_val = X_train.iloc[val_idx]
            y_fold_val = y_train.iloc[val_idx]

            # Scale the features
            scaler = StandardScaler()
            X_fold_train_scaled = scaler.fit_transform(X_fold_train)
            X_fold_val_scaled = scaler.transform(X_fold_val)

            # Apply SMOTE
            smote = SMOTE(random_state=random_state)
            X_fold_train_balanced, y_fold_train_balanced = smote.fit_resample(
                X_fold_train_scaled, y_fold_train
            )

            # Train and predict
            clf_fold = clone(clf)
            clf_fold.fit(X_fold_train_balanced, y_fold_train_balanced)
            y_fold_pred = clf_fold.predict(X_fold_val_scaled)

            # Calculate scores
            cv_scores.append({
                'accuracy': accuracy_score(y_fold_val, y_fold_pred),
                'precision_macro': precision_score(y_fold_val, y_fold_pred, average='macro'),
                'recall_macro': recall_score(y_fold_val, y_fold_pred, average='macro'),
                'f1_macro': f1_score(y_fold_val, y_fold_pred, average='macro'),
                'f1_weighted': f1_score(y_fold_val, y_fold_pred, average='weighted')
            })

            # Calculate F1 score for each class
            f1_scores_per_class.append(
                f1_score(y_fold_val, y_fold_pred, average=None)
            )

        # Train final model on full training set
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        smote = SMOTE(random_state=random_state)
        X_train_balanced, y_train_balanced = smote.fit_resample(X_train_scaled, y_train)

        clf.fit(X_train_balanced, y_train_balanced)
        y_pred = clf.predict(X_test_scaled)

        # Calculate average scores across folds
        cv_results = {}
        for metric in scoring.keys():
            scores = [fold[metric] for fold in cv_scores]
            cv_results[f'test_{metric}'] = np.array(scores)

        # Calculate average F1 scores per class
        f1_per_class = np.mean(f1_scores_per_class, axis=0)

        # Store results
        results[name] = {
            'cv_results': cv_results,
            'classification_report': classification_report(y_test, y_pred),
            'confusion_matrix': confusion_matrix(y_test, y_pred),
            'model': clf,
            'f1_per_class': f1_per_class
        }

        # Print detailed results for this classifier
        print(f"\nΑποτελέσματα για {name}:")
        print("\nCross-validation scores:")
        for metric, scores in cv_results.items():
            print(f"{metric}: {scores.mean():.4f} (+/- {scores.std() * 2:.4f})")

        print("\nF1 scores per class:")
        for i, score in enumerate(f1_per_class):
            print(f"Class {i}: {score:.4f}")

        print("\nClassification Report:")
        print(results[name]['classification_report'])

    return results

def create_metrics_summary(results):
    """
    Δημιουργεί συγκεντρωτικό πίνακα με όλες τις μετρικές για όλα τα μοντέλα
    και παράγει οπτικοποιήσεις των αποτελεσμάτων.

    Parameters
    ----------
    results : dict
        Λεξικό με τα αποτελέσματα των μοντέλων, όπου κάθε στοιχείο περιέχει:
        - 'cv_results' : dict
            Αποτελέσματα cross-validation για κάθε μετρική
        - 'f1_per_class' : array
            F1-scores για κάθε κλάση
        - 'model' : object
            Το εκπαιδευμένο μοντέλο

    Returns
    -------
    pandas.DataFrame
        DataFrame με τις συγκεντρωτικές μετρικές για κάθε μοντέλο, που περιλαμβάνει:
        - Βασικές μετρικές (accuracy, precision, recall, f1)
        - Τυπικές αποκλίσεις των μετρικών
        - F1-scores ανά κλάση

    Notes
    -----
    Η συνάρτηση δημιουργεί τις εξής οπτικοποιήσεις:
    1. Heatmap
        - Συγκριτική απεικόνιση όλων των μετρικών για όλα τα μοντέλα
        - Χρωματική κλίμακα για εύκολη σύγκριση

    2. Bar plots
        - Ξεχωριστό γράφημα για κάθε μετρική
        - Περιλαμβάνει error bars για την τυπική απόκλιση
        - Εμφανίζει τις ακριβείς τιμές πάνω από κάθε ράβδο

    3. Radar plot
        - Συγκριτική απεικόνιση των βασικών μετρικών
        - Επιτρέπει την οπτική σύγκριση των μοντέλων σε όλες τις μετρικές ταυτόχρονα

    Examples
    --------
    >>> results = compare_classifiers_with_smote(X, y)
    >>> metrics_df = create_metrics_summary(results)

    """
    # Δημιουργία λίστας για το DataFrame
    data = []

    for name, result in results.items():
        model_metrics = {'Model': name}

        # Προσθήκη βασικών μετρικών από cross-validation
        if 'cv_results' in result:
            for metric_key in result['cv_results'].keys():
                metric_name = metric_key.replace('test_', '')
                scores = result['cv_results'][metric_key]
                model_metrics[metric_name] = scores.mean()
                model_metrics[f'{metric_name}_std'] = scores.std()

        # Προσθήκη F1 scores ανά κλάση
        if 'f1_per_class' in result:
            for i, score in enumerate(result['f1_per_class']):
                model_metrics[f'f1_class_{i}'] = score

        data.append(model_metrics)

    # Δημιουργία DataFrame
    df_metrics = pd.DataFrame(data)

    # Εύρεση των διαθέσιμων μετρικών (εξαιρώντας τις στήλες _std)
    metrics = [col for col in df_metrics.columns
              if col != 'Model' and not col.endswith('_std')]

    # Στρογγυλοποίηση των αριθμών
    numeric_columns = df_metrics.select_dtypes(include=[np.number]).columns
    df_metrics[numeric_columns] = df_metrics[numeric_columns].round(4)

    # Εμφάνιση του πίνακα
    print("\n=== Συγκεντρωτικός Πίνακας Μετρικών ===")
    print(df_metrics.to_string(index=False))

    # 1. Heatmap των βασικών μετρικών
    plt.figure(figsize=(12, 6))
    metrics_for_heatmap = df_metrics[['Model'] + metrics].set_index('Model')
    sns.heatmap(metrics_for_heatmap,
                annot=True,
                fmt='.4f',
                cmap='Blues',
                cbar_kws={'label': 'Score'})
    plt.title('Σύγκριση Μοντέλων ανά Μετρική')
    plt.tight_layout()
    plt.show()

    # 2. Bar plots για κάθε μετρική
    n_metrics = len(metrics)
    n_cols = 2
    n_rows = (n_metrics + 1) // 2

    plt.figure(figsize=(15, 5*n_rows))

    for i, metric in enumerate(metrics, 1):
        plt.subplot(n_rows, n_cols, i)

        # Δημιουργία του bar plot
        bars = plt.bar(df_metrics['Model'], df_metrics[metric])

        # Προσθήκη error bars αν υπάρχουν
        if f'{metric}_std' in df_metrics.columns:
            plt.errorbar(df_metrics['Model'],
                        df_metrics[metric],
                        yerr=df_metrics[f'{metric}_std'],
                        fmt='none',
                        color='black',
                        capsize=5)

        # Προσθήκη τιμών πάνω από τις μπάρες
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.4f}',
                    ha='center', va='bottom')

        plt.title(f'{metric} ανά Μοντέλο')
        plt.xticks(rotation=45)
        plt.ylabel('Score')

    plt.tight_layout()
    plt.show()

    # 3. Radar plot
    # Επιλογή μόνο των βασικών μετρικών για το radar plot
    basic_metrics = ['accuracy', 'precision_macro', 'recall_macro', 'f1_macro', 'f1_weighted']
    radar_metrics = [m for m in basic_metrics if m in metrics]

    plt.figure(figsize=(10, 10))

    # Προετοιμασία των δεδομένων για το radar plot
    models = df_metrics['Model'].tolist()
    metrics_values = df_metrics[radar_metrics].values

    # Δημιουργία των γωνιών για το plot
    angles = np.linspace(0, 2*np.pi, len(radar_metrics), endpoint=False)
    angles = np.concatenate((angles, [angles[0]]))

    # Plot για κάθε μοντέλο
    ax = plt.subplot(111, projection='polar')
    for i, model in enumerate(models):
        values = metrics_values[i]
        values = np.concatenate((values, [values[0]]))
        ax.plot(angles, values, 'o-', linewidth=2, label=model)
        ax.fill(angles, values, alpha=0.25)

    # Προσθήκη των ετικετών
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(radar_metrics)
    ax.set_title('Σύγκριση Μοντέλων - Radar Plot')
    plt.legend(loc='upper right', bbox_to_anchor=(0.1, 0.1))

    plt.show()

    # Εύρεση του καλύτερου μοντέλου για κάθε μετρική
    print("\n=== Καλύτερο Μοντέλο ανά Μετρική ===")
    for metric in metrics:
        best_model = df_metrics.loc[df_metrics[metric].idxmax()]
        print(f"{metric:15} : {best_model['Model']:20} (Score: {best_model[metric]:.4f})")

    return df_metrics

results_with_smote = compare_classifiers_with_smote(x, y)
metrics_summary_with_smote = create_metrics_summary(results_with_smote)

def train_random_forest_with_smote(X, y, random_state=42):
    """
    Εκπαιδεύει ένα Random Forest μοντέλο χρησιμοποιώντας SMOTE και StandardScaler.

    Parameters
    ----------
    X : pandas.DataFrame
        Τα χαρακτηριστικά του συνόλου δεδομένων
    y : pandas.Series
        Οι ετικέτες του συνόλου δεδομένων
    random_state : int, optional (default=42)
        Σταθερά για αναπαραγωγή των αποτελεσμάτων

    Returns
    -------
    dict
        Dictionary με το μοντέλο και τα αποτελέσματα
    """
    # Διαχωρισμός δεδομένων
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=random_state, stratify=y
    )

    # Εμφάνιση αρχικής κατανομής
    print("Κατανομή κλάσεων πριν το SMOTE:")
    print(pd.Series(y_train).value_counts())

    # Κανονικοποίηση
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Εφαρμογή SMOTE
    smote = SMOTE(random_state=random_state)
    X_train_balanced, y_train_balanced = smote.fit_resample(X_train_scaled, y_train)

    print("\nΚατανομή κλάσεων μετά το SMOTE:")
    print(pd.Series(y_train_balanced).value_counts())

    # Δημιουργία και εκπαίδευση του μοντέλου
    rf = RandomForestClassifier(
        n_estimators=100,
        max_depth=None,
        min_samples_split=2,
        min_samples_leaf=1,
        random_state=random_state
    )

    # Cross-validation
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=random_state)
    cv_scores = cross_val_score(rf, X_train_balanced, y_train_balanced, cv=cv)

    print("\nCross-validation scores:", cv_scores)
    print(f"Mean CV score: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")

    # Εκπαίδευση τελικού μοντέλου
    rf.fit(X_train_balanced, y_train_balanced)

    # Προβλέψεις
    y_pred = rf.predict(X_test_scaled)

    # Αποτελέσματα
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))

    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.show()

    # Feature Importance
    feature_importance = pd.DataFrame({
        'feature': X.columns,
        'importance': rf.feature_importances_
    }).sort_values('importance', ascending=False)

    plt.figure(figsize=(12, 6))
    sns.barplot(x='importance', y='feature', data=feature_importance)
    plt.title('Feature Importance')
    plt.show()

    # Αποθήκευση των αποτελεσμάτων
    results = {
        'model': rf,
        'scaler': scaler,
        'cv_scores': cv_scores,
        'classification_report': classification_report(y_test, y_pred),
        'confusion_matrix': cm,
        'feature_importance': feature_importance,
        'predictions': y_pred
    }

    return results


def save_rf_model(results, filename='rf_model.joblib'):
    """
    Αποθηκεύει ένα μοντέλο Random Forest και τον αντίστοιχο scaler σε ένα αρχείο.

    Παράμετροι
    ----------
    results : tuple ή μοντέλο
        Το μοντέλο Random Forest ή ένα tuple που περιέχει το μοντέλο και τον scaler
        που θα αποθηκευτούν.

    filename : str, προαιρετικό (default='rf_model.joblib')
        Το όνομα του αρχείου στο οποίο θα αποθηκευτεί το μοντέλο.
        Πρέπει να έχει κατάληξη .joblib

    Returns
    -------
    None
        Η συνάρτηση εκτυπώνει μήνυμα επιβεβαίωσης μετά την αποθήκευση.
    """
    import joblib
    joblib.dump(results, filename)
    print(f"Το μοντέλο αποθηκεύτηκε ως {filename}")


def load_rf_model(filename='rf_model.joblib'):
    """
    Φορτώνει ένα αποθηκευμένο μοντέλο Random Forest και τον αντίστοιχο scaler από ένα αρχείο.

    Παράμετροι
    ----------
    filename : str, προαιρετικό (default='rf_model.joblib')
        Το όνομα του αρχείου από το οποίο θα φορτωθεί το μοντέλο.
        Πρέπει να έχει κατάληξη .joblib

    Returns
    -------
    model : tuple ή μοντέλο
        Επιστρέφει το μοντέλο Random Forest ή ένα tuple που περιέχει
        το μοντέλο και τον scaler που είχαν αποθηκευτεί.

    Raises
    ------
    FileNotFoundError
        Αν το αρχείο που καθορίζεται από το filename δεν υπάρχει.
    """
    import joblib
    return joblib.load(filename)


def predict_rf(results, X_new):
    """
    Κάνει προβλέψεις για νέα δεδομένα.

    Parameters
    ----------
    results : dict
        Το dictionary με το μοντέλο και τον scaler
    X_new : array-like
        Τα νέα δεδομένα για πρόβλεψη

    Returns
    -------
    dict
        Dictionary με προβλέψεις και πιθανότητες
    """
    # Κανονικοποίηση των νέων δεδομένων
    X_new_scaled = results['scaler'].transform(X_new)

    # Προβλέψεις
    predictions = results['model'].predict(X_new_scaled)
    probabilities = results['model'].predict_proba(X_new_scaled)

    return {
        'predictions': predictions,
        'probabilities': probabilities
    }

# Εκπαίδευση του μοντέλου
results = train_random_forest_with_smote(x, y)

# Αποθήκευση του μοντέλου
save_rf_model(results, 'rf_model.joblib')
