# Import des outils de visualisation
import streamlit as st
import hydralit_components as hc
import matplotlib.pyplot as plt

import pandas as pd

# Import des fonctions pour la régression logistique
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

# Récupération des données
import seaborn as sns

# Import files
import constant
import text
import code_text


def main() -> None:
    """
        Affiche le menu supérieur horizontal et gère l'affichage des pages.
    """
    # Configuration du menu
    st.set_page_config(layout='wide', initial_sidebar_state='collapsed',)
    menu = hc.nav_bar(
        menu_definition=constant.menu,
        hide_streamlit_markers=True
    )

    # Affichage de la page d'accueil
    if menu == "home":
        "# 🏠 Accueil"
        st.write(text.home)

    # Affichage des données
    if menu == "data":
        "# 📖 Lecture des données"
        st.session_state.data = load_data()
        print_data(st.session_state.data)
        st.write(text.presentation_data)
        print_code(code_text.load_data, "load_data", True)

    # TODO
    # Affichage de la régression logistique
    if menu == "régression":
        "# 📉 Régression logistique"
        if "data" not in st.session_state:
            st.write("# Veuillez charger les données dans l'onglet Lecture des données.")
        else:
            st.write(st.session_state.data.head())
            regression_logistique()

    # TODO ajouter individu via un appel à une fonction :)
    # Affichage de la possibilité d'ajouter un individu
    if menu == "add":
        "# ➕ Tester un nouvel individu"
        if "lr_model" not in st.session_state:
            st.write("# Veuillez entraîner le modèle dans l'onglet Régression logistique.")
        else:
            add_individual(st.session_state.lr_model)


def print_code(text: str, key: str, separator: bool = False, show_code_by_default: bool = True) -> None:
    """Affiche le texte sous forme de code.\\
    - Le paramètre key permet de donner un identifiant à la checkbox pour éviter une erreur qui apparaît lorsque
    plusieurs checkbox n'ont pas de clés et ont la même structure.\\
    - Le paramètre separator permet d'afficher une ligne horizontale avant la checkbox pour séparer le code de
    la partie précédente de la page. Par défaut, c'est à False.
    - Le paramètre show_code_by_default permet de choisir si le code est montré par défaut au non.
    Par défaut c'est True, donc le code est affiché.
    """
    if separator:
        st.write("---")
    show_code = st.checkbox(
        label="Montrer le code",
        value=show_code_by_default,
        key=key
    )

    if show_code:
        st.code(text, 'python')


def load_data() -> pd.DataFrame:
    """
        Retourne les données au format DataFrame du titanic.
        L'utilisation de cache=False permet de récupérer les données quoi qu'il arrive,
        sinon, on risque d'avoir une erreur du type "this dataset doesn't exists".
    """
    return sns.load_dataset("titanic", cache=False)


def print_data(data: pd.DataFrame) -> None:
    """
        Créé une checkbox permettant d'afficher ou non les données.
    """
    show_data = st.checkbox(
        label="Montrer les données",
        value=False
    )
    if show_data:
        st.write(data)


def regression_logistique() -> None:
    """
        Permet de choisir la variable cible et de faire la régression logistique.
        D'afficher les résultats textuellement et graphiquement.
    """
    # Récupération de la variable cible et séparation
    # Enlever les valeurs NaN
    data = st.session_state.data.dropna()
    target_variable = st.selectbox(
        label="Choisissez la variable cible :",
        options=[col for col in data.columns]
    )
    X = data.drop(columns=[target_variable])
    y = data[target_variable]

    # Les colonnes de type object et category sont déjà traités, donc il suffit de les supprimer.
    # Sinon, il aurait fallu encoder les colonnes en remplacement les valeurs par 0 et 1 par exemple ou en
    # changeant la colonne (exemple la colonne genre devient 'male' et affiche 1 si c'est un homme,
    # 0 si c'est une femme)
    for col in X.select_dtypes(include=['object', 'category']).columns:
        X = X.drop(columns=[col])

    # Division des données en ensembles d'entraînement et de test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Création et ajustement du modèle de régression logistique
    lr = LogisticRegression()
    lr.fit(X_train, y_train)

    # Prédiction et évaluation du modèle
    y_pred = lr.predict(X_test)
    st.subheader("Évaluation du modèle de régression logistique")
    st.write(classification_report(y_test, y_pred))
    st.write("Matrice de confusion :")
    st.write(confusion_matrix(y_test, y_pred))

    # Visualisation
    # Diagramme de dispersion
    st.subheader("Diagramme de dispersion")
    st.write("*Le chargement peut prendre du temps*")

    # Sélection des colonnes numériques à inclure dans le pairplot
    numerical_columns = X.select_dtypes(include=['int64', 'float64']).columns
    pairplot_data = pd.concat([X[numerical_columns], y], axis=1)

    # Pairplot
    fig = sns.pairplot(pairplot_data, hue=target_variable)

    # Sauvegarder le modèle dans st.session_state
    st.session_state.lr_model = lr

    # Affichage des caractéristiques et de leurs coefficients
    # st.subheader("Caractéristiques et coefficients du modèle de régression logistique")
    # feature_names = X_train.columns
    # coefficients = lr.coef_[0]

    # for feature, coef in zip(feature_names, coefficients):
    #     st.write(f"{feature}: {coef}")
    
    st.pyplot(fig)

    # Graphique de répartition de la variable cible
    st.subheader("Graphique de répartition de la variable cible")
    fig, ax = plt.subplots()
    sns.countplot(x=target_variable, data=st.session_state.data, ax=ax)
    st.pyplot(fig)

def add_individual(lr_model):
    "# ➕ Ajouter un individu"

    # Collecte des données de l'individu
    age = st.slider("Age de l'individu", min_value=1, max_value=100, value=30)
    adult_male = st.checkbox("Homme", value=True)
    pclass = st.selectbox("Classe du ticket", options=[1, 2, 3], index=0)
    sibsp = st.slider("Nombre de frères/soeurs ou conjoints à bord", min_value=0, max_value=10, value=0)
    parch = st.slider("Nombre de parents/enfants à bord", min_value=0, max_value=10, value=0)
    fare = st.number_input("Prix payé", min_value=0.0, max_value=500.0, value=30.0, step=10.0)
    alone = st.checkbox("Voyage en solo", value=True)  # Vous devez ajuster cela en fonction de votre logique

    # Préparation des données de l'individu
    individual_data = pd.DataFrame({
        'pclass': [pclass],
        'age': [age],
        'sibsp': [sibsp],
        'parch': [parch],
        'fare': [fare],
        'adult_male': [adult_male],
        'alone': [alone]
    })

    # Prédiction de survie avec le modèle sauvegardé
    prediction = lr_model.predict(individual_data)

    # Affichage de la prédiction
    st.subheader("Prédiction de l'appartenance de l'individu")
    # Modification du texte en fonction de la prédiction
    if prediction[0] == 1:
        st.write(f"L'individu est prédit comme étant : vivant(e)")
    else:
        st.write(f"L'individu est prédit comme étant : mort(e)")

if __name__ == "__main__":
    main()
