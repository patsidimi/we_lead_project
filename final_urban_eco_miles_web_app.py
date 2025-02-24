# -*- coding: utf-8 -*-


pip install gradio

import gradio as gr
import numpy as np
import pandas as pd
from datetime import datetime
import joblib
import os

def load_rf_model(filename='rf_model.joblib'):
    """
    Φορτώνει ένα αποθηκευμένο μοντέλο Random Forest από αρχείο.

    Αυτή η συνάρτηση χρησιμοποιεί τη βιβλιοθήκη `joblib` για να φορτώσει
    ένα προεκπαιδευμένο μοντέλο Random Forest από το καθορισμένο αρχείο.

    Args:
        filename (str): Το όνομα του αρχείου που περιέχει το αποθηκευμένο μοντέλο.
                        Προεπιλεγμένη τιμή είναι το 'rf_model.joblib'.

    Returns:
        object: Το φορτωμένο μοντέλο Random Forest.

    Raises:
        FileNotFoundError: Αν το αρχείο δεν υπάρχει στη καθορισμένη τοποθεσία.
        Exception: Για άλλα σφάλματα που μπορεί να προκύψουν κατά τη διαδικασία φόρτωσης.
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

# Φόρτωση του μοντέλου και του scaler
loaded_results = load_rf_model('/content/rf_model.joblib')
model = loaded_results['model']
scaler = loaded_results['scaler']

# Mappings
brand_mapping = {
    'Chevrolet': 1, 'Buick': 2, 'Plymouth': 3, 'Amc': 4, 'Ford': 5,
    'Pontiac': 6, 'Dodge': 7, 'Toyota': 8, 'Nissan': 9, 'Vw': 10,
    'Peugeot': 11, 'Audi': 12, 'Saab': 13, 'Bmw': 14, 'Hi': 15,
    'Mercury': 16, 'Opel': 17, 'Fiat': 18, 'Oldsmobile': 19,
    'Chrysler': 20, 'Mazda': 21, 'Volvo': 22, 'Renault': 23,
    'Honda': 24, 'Subaru': 25, 'Mercedes': 26, 'Cadillac': 27,
    'Triumph': 28
}

prediction_mapping = {
    0: "low mpg",
    1: "medium mpg",
    2: "high mpg"
}
brand_logos = {
    'chevrolet': "https://di-uploads-pod2.dealerinspire.com/biggerschevy/uploads/2018/02/2004-Chevrolet-Bowtie.jpg",
    'buick': "https://www.buick.com/content/dam/buick/na/us/en/portable-nav/buick-logo-310x89.png",
    'plymouth': "https://1000logos.net/wp-content/uploads/2020/04/Plymouth-Logo-1969-500x352.jpg",
    'amc':"https://1000logos.net/wp-content/uploads/2020/12/AMC-Logo-500x315.png",
    'ford': "https://1000logos.net/wp-content/uploads/2018/02/Ford-logo-500x281.jpg",
    'pontiac': "https://1000logos.net/wp-content/uploads/2020/04/Pontiac-Logo-1959-500x281.png",
    'dodge': "https://1000logos.net/wp-content/uploads/2018/04/Dodge-Logo-1964-500x300.png",
    'toyota': 'https://1000logos.net/wp-content/uploads/2018/02/Toyota-Logo-1978-500x281.png',
    'nissan': "https://1000logos.net/wp-content/uploads/2020/03/Nissan-Logo-1970-500x281.jpg",
    'vw': "https://1000logos.net/wp-content/uploads/2019/12/Volkswagen-Logo-1960.jpg",
    'peugeot': "https://1000logos.net/wp-content/uploads/2019/12/Peugeot-Logo-1975-500x333.jpg",
    'audi': "https://1000logos.net/wp-content/uploads/2018/05/Audi-Logo-1969-500x305.jpg",
    'saab': "https://1000logos.net/wp-content/uploads/2020/04/Saab-Logo-1987-500x333.jpg",
    'bmw': "https://1000logos.net/wp-content/uploads/2018/02/BMW-Logo-1963-500x281.png",
    'hi': "http://www.hansenwebdesign.com/truck/graphics/logos/man_on_tractor_logo_1.png",
    'mercury': "https://1000logos.net/wp-content/uploads/2020/04/Mercury-Logo-1984-500x282.png",
    'opel': "https://1000logos.net/wp-content/uploads/2020/04/Opel-Logo-1970-500x341.jpg",
    'fiat': "https://1000logos.net/wp-content/uploads/2020/02/Fiat-Logo-1972-500x300.jpg" ,
    'oldsmobile': "https://1000logos.net/wp-content/uploads/2020/04/Oldsmobile-Logo-1960-500x361.jpg",
    'chrysler': "https://1000logos.net/wp-content/uploads/2020/04/Chrysler-Logo-1962-500x281.png",
    'mazda': "https://1000logos.net/wp-content/uploads/2019/12/Mazda-Logo-1954-500x315.png",
    'volvo': "https://1000logos.net/wp-content/uploads/2020/03/Volvo-Logo-1959-99-500x315.png",
    'renault': "https://1000logos.net/wp-content/uploads/2019/12/Renault-Logo-1973-500x333.png",
    'honda':"https://1000logos.net/wp-content/uploads/2018/03/Honda-Logo-1969-500x281.png",
    'subaru': "https://1000logos.net/wp-content/uploads/2018/03/Subaru-Logo-1970-500x281.jpg",
    'mercedes': "https://1000logos.net/wp-content/uploads/2018/04/Mercedes-Logo-1933-500x304.jpg",
    'cadillac': "https://1000logos.net/wp-content/uploads/2020/04/Cadillac-Logo-1980-500x282.png",
    'triumph': "https://1000logos.net/wp-content/uploads/2020/04/Triumph-Logo-1936-500x333.jpg"

}

def save_to_csv(cylinders, displacement, horsepower, acceleration,
                model_year, weight_cat, brand, prediction):
    """
    Αποθηκεύει τα δεδομένα εισόδου και τις προβλέψεις σε αρχείο CSV.

    Αυτή η συνάρτηση δημιουργεί ένα νέο αρχείο CSV ή προσθέτει μια νέα εγγραφή
    σε υπάρχον αρχείο (append mode), αποθηκεύοντας τα δεδομένα εισόδου μαζί με
    την αντίστοιχη πρόβλεψη και την τρέχουσα χρονική σήμανση (timestamp).

    Args:
        cylinders (int): Αριθμός κυλίνδρων του οχήματος.
        displacement (float): Κυβισμός (σε κυβικά εκατοστά) του κινητήρα.
        horsepower (float): Ιπποδύναμη του οχήματος.
        acceleration (float): Επίδοση επιτάχυνσης (π.χ. χρόνος 0-100 km/h).
        model_year (int): Έτος κατασκευής του οχήματος.
        weight_cat (str): Κατηγορία βάρους του οχήματος.
        brand (str): Επωνυμία ή μάρκα του οχήματος.
        prediction (float): Αποτέλεσμα της πρόβλεψης του μοντέλου.

    Returns:
        None: Η συνάρτηση δεν επιστρέφει τιμή. Αποθηκεύει τα δεδομένα στο αρχείο 'predictions_log.csv'.

    Notes:
        - Τα δεδομένα αποθηκεύονται με μορφή χωρίς κεφαλίδες (header=False).
        - Εάν το αρχείο δεν υπάρχει, δημιουργείται αυτόματα.
        - Προστίθεται timestamp για κάθε νέα εγγραφή.

    Raises:
        Exception: Για σφάλματα που σχετίζονται με την εγγραφή στο αρχείο CSV
                   (π.χ. προβλήματα αδειών εγγραφής ή σφάλματα αρχείου).
    """
    data = {
        'timestamp': datetime.now(),
        'cylinders': cylinders,
        'displacement':  displacement,
        'horsepower': horsepower,
        'acceleration': acceleration,
        'model_year': model_year,
        'weight_cat': weight_cat,
        'brand': brand,
        'prediction': prediction
    }
    file_path = 'predictions_log.csv'
    file_exists = os.path.isfile(file_path)

    CSV_HEADERS = ["timestamp", "cylinders", "displacement", "horsepower", "acceleration",
               "model_year", "weight_cat", "brand", "prediction"]
    df = pd.DataFrame([data])
    df.to_csv('predictions_log.csv', mode='a', header=not file_exists , index=False)



def brand_to_number(brand):

    """Μετατρέπει το όνομα της μάρκας σε έναν αντίστοιχο αριθμητικό κωδικό.

    Αυτή η συνάρτηση χρησιμοποιεί έναν προκαθορισμένο χάρτη (dictionary) `brand_mapping`
    για να αντιστοιχίσει το όνομα μιας μάρκας (brand) σε έναν μοναδικό αριθμό.

    Args:
        brand (str): Το όνομα της μάρκας που θα μετατραπεί σε αριθμό.

    Returns:
        int: Ο αριθμητικός κωδικός που αντιστοιχεί στη μάρκα.

    """
    return brand_mapping[brand]

#read xlsx
df=pd.read_excel("/content/mpg.data.xlsx")

#rename displayments
df=df.rename(columns={"displayments":"displacement"})

def make_prediction(cylinders,  displacement, horsepower, acceleration,
                   model_year, weight_cat, brand):
    """
    Κάνει πρόβλεψη της κατανάλωσης καυσίμου (mpg) χρησιμοποιώντας το μοντέλο Random Forest.

    Η συνάρτηση δέχεται χαρακτηριστικά οχήματος, επεξεργάζεται τα δεδομένα,
    και χρησιμοποιεί ένα εκπαιδευμένο μοντέλο Random Forest για να κάνει πρόβλεψη.
    Παράγει επίσης ένα HTML output με την πρόβλεψη και τις αντίστοιχες πιθανότητες.

    Βήματα λειτουργίας:
    1. Μετατροπή της μάρκας του οχήματος σε αριθμητικό κωδικό.
    2. Δημιουργία πίνακα χαρακτηριστικών και κανονικοποίηση δεδομένων.
    3. Πρόβλεψη κατηγορίας κατανάλωσης καυσίμου και υπολογισμός πιθανοτήτων.
    4. Αποθήκευση δεδομένων και αποτελεσμάτων σε αρχείο CSV.
    5. Επιστροφή HTML μορφοποιημένου αποτελέσματος με χρωματική ένδειξη.

    Args:
        cylinders (int): Αριθμός κυλίνδρων του οχήματος.
        displacement (float): Κυβισμός (σε κυβικά εκατοστά) του κινητήρα.
        horsepower (float): Ιπποδύναμη του οχήματος.
        acceleration (float): Επίδοση επιτάχυνσης
        year (int): Έτος κατασκευής του οχήματος.
        weight_cat (str): Κατηγορία βάρους του οχήματος.
        brand (str): Μάρκα του οχήματος.

    Returns:
        tuple:
            - output_html (str): HTML κώδικας με το αποτέλεσμα της πρόβλεψης και τις πιθανότητες.


    Raises:
        Exception: Για οποιοδήποτε σφάλμα κατά την επεξεργασία των δεδομένων
                   ή την εκτέλεση του μοντέλου, επιστρέφεται μήνυμα σφάλματος.
    """
    try:

        mean_values = {
            'cylinders': df['cylinders'].mean(),
            'displacement': df['displacement'].mean(),
            'horsepower': df['horsepower'].mean(),
            'acceleration': df['acceleration'].mean(),
            'model_year': df['model year'].mean()
        }

        # Έλεγχος για κενά πεδία και αντικατάσταση με μέσες τιμές
        # Επίσης μετατροπή αρνητικών τιμών σε απόλυτες
        cylinders = abs(float(cylinders)) if cylinders not in [None, ''] else mean_values['cylinders']
        displacement = abs(float(displacement)) if displacement not in [None, ''] else mean_values['displacement']
        horsepower = abs(float(horsepower)) if horsepower not in [None, ''] else mean_values['horsepower']
        acceleration = abs(float(acceleration)) if acceleration not in [None, ''] else mean_values['acceleration']
        model_year = abs(float(model_year)) if model_year not in [None, ''] else mean_values['model_year']



        # Μετατροπή της μάρκας σε αριθμό
        brand_num = brand_to_number(brand)

        # Δημιουργία του input array
        features = [cylinders, displacement, horsepower, acceleration,
                   model_year, weight_cat, brand_num]
        input_data = np.array([features])

        # Κανονικοποίηση των δεδομένων
        input_scaled = scaler.transform(input_data)

        # Πρόβλεψη
        raw_prediction = model.predict(input_scaled)[0]
        probabilities = model.predict_proba(input_scaled)[0]

        # Χρωματικός κώδικας για κάθε κατηγορία
        color_mapping = {
            "low mpg": "#E74C3C",    # κόκκινο
            "medium mpg": "#B8860B",  # κίτρινο
            "high mpg": "#228B22"     # πράσινο
        }

        prediction = prediction_mapping.get(int(raw_prediction), 'unknown')

        # Προσθήκη ένδειξης για τα πεδία που χρησιμοποιήθηκαν μέσες τιμές
        missing_fields = []
        if cylinders is None: missing_fields.append("cylinders")
        if displacement is None: missing_fields.append("displacement")
        if horsepower is None: missing_fields.append("horsepower")
        if acceleration is None: missing_fields.append("acceleration")
        if model_year is None: missing_fields.append("model year")

        # Δημιουργία HTML output με επιπλέον πληροφορίες για τις μέσες τιμές
        bg_color = color_mapping.get(prediction, "#ffffff")
        output_html = f"""
        <div style="background-color: {bg_color}; padding: 10px; border-radius: 5px;">
            <p><strong>Πρόβλεψη:</strong> {prediction}</p>
            <p><strong>Πιθανότητες:</strong></p>
        """

        for i, prob in enumerate(probabilities):
            mpg_category = prediction_mapping.get(i, f"Κατηγορία {i}")
            output_html += f"<p>{mpg_category}: {prob:.3f}</p>"

        # Προσθήκη πληροφοριών για τις μέσες τιμές που χρησιμοποιήθηκαν
        if missing_fields:
            output_html += f"""
            <p style="color: #666; font-style: italic;">
                Σημείωση: Χρησιμοποιήθηκαν μέσες τιμές για τα πεδία: {', '.join(missing_fields)}
            </p>
            """

        output_html += "</div>"
        # Παίρνουμε το λογότυπο της μάρκας
        brand_logo = brand_logos.get(brand.lower(), None)

        save_to_csv(cylinders, displacement, horsepower, acceleration,
                model_year, weight_cat, brand, prediction)

        return output_html, brand_logo

    except Exception as e:
        return f"Σφάλμα: {str(e)}", None

def clear_inputs():

    """
    Επιστρέφει κενές τιμές για όλα τα πεδία
    """

    return '', '', '', '', '', '', ''

with gr.Blocks() as demo:
    gr.Image("/content/logo.jpeg", show_label=True, height=200, width=1000)
    gr.Markdown("# 🌿 Urban eco miles")

    with gr.Row():
        # Πεδίο εισαγωγής δεδομένων
        with gr.Column(scale=1):
            cylinders = gr.Number(label="cylinders", value="")
            displacement = gr.Number(label="displacement", value="")
            horsepower = gr.Number(label="horsepower", value="")
            acceleration = gr.Number(label="acceleration", value="")
            model_year =  gr.Dropdown(
                choices=["68", "69", "70", "71", "72","73", "74", "75","76", "77", "78","79", "80", "81","82","83","84"],
                label="model_year",
                info="Επιλέξτε έτος έκδοσης μοντέλου"
            )
            weight_cat = gr.Dropdown(
                choices=["1", "2", "3", "4", "5"],
                label="weight_cat",
                info="Επιλέξτε κατηγορία βάρους"
            )
            brand = gr.Dropdown(
                choices=sorted(brand_mapping.keys()),
                label="brand",
                info="Επιλέξτε μάρκα αυτοκινήτου"
            )

        # Αποτελέσματα Πρόβλεψης (Τοποθετημένα στο ίδιο ύψος με τα πεδία)
        with gr.Column(scale=1):
            output = gr.HTML(label="Αποτέλεσμα")
            brand_logo_output = gr.Image(height=200, width=300)

    # Κουμπιά υποβολής και εκκαθάρισης
    with gr.Row():
        with gr.Column(scale=0.5):
            submit_btn = gr.Button("🚀 Κάνε Πρόβλεψη", scale=0.5)
        with gr.Column(scale=0.5, min_width=0):
            clear_btn = gr.Button("🗑️ Καθαρισμός Πεδίων", variant="secondary", scale=0.5)

    # Συνδέσεις κουμπιών με συναρτήσεις
    submit_btn.click(
        fn=make_prediction,
        inputs=[cylinders, displacement, horsepower, acceleration, model_year, weight_cat, brand],
        outputs=[output, brand_logo_output]
    )

    clear_btn.click(
        clear_inputs,
        inputs=[],
        outputs=[cylinders, displacement, horsepower, acceleration, model_year, weight_cat, brand]
    )

demo.launch()
