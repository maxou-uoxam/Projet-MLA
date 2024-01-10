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

    # TODO
    # Affichage de la possibilité d'ajouter un individu
    if menu == "add":
        "# ➕ Ajouter un individu"


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
    fig = sns.pairplot(data, hue=target_variable)
    st.pyplot(fig)

    # Graphique de répartition de la variable cible
    st.subheader("Graphique de répartition de la variable cible")
    fig, ax = plt.subplots()
    sns.countplot(x=target_variable, data=st.session_state.data, ax=ax)
    st.pyplot(fig)


if __name__ == "__main__":
    main()
