load_data = """
# Retourne les données au format DataFrame du titanic.
# L'utilisation de cache=False permet de récupérer les données quoi qu'il arrive,
# sinon, on risque d'avoir une erreur du type "this dataset doesn't exists".
return sns.load_dataset("titanic", cache=False)
"""

regression_logistique = """
# Permet de choisir la variable cible et de faire la régression logistique.
# D'afficher les résultats textuellement et graphiquement.

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
# Sauvegarder le modèle dans st.session_state
st.session_state.lr_model = lr

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
st.pyplot(fig)

# Graphique de répartition de la variable cible
st.subheader("Graphique de répartition de la variable cible")
fig, ax = plt.subplots()
sns.countplot(x=target_variable, data=st.session_state.data, ax=ax)
st.pyplot(fig)
"""

add_individual = """
"# Ajouter un individu"

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
    st.write("L'individu est prédit comme étant : vivant(e)")
else:
    st.write("L'individu est prédit comme étant : mort(e)")

"""
