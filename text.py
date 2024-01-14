home = """
# Projet de MLA (Machine Learning et Applications)

Ce document présente le travail effectué par Maxime Dangelser et Nicolas Vanhoucke dans le cadre du cours de
Machine Learning et Applications des données du Master 2 MIAGE parcours Business Intelligence de Polytech Lyon.\\
Le sujet porte sur le développement d'un programme de régression en python.

## Présentation du langage utilisé :

Lors de ce projet nous avons utilisé le langage de programmation Python.

Python est un langage de programmation interprété, à syntaxe claire et simple, et qui met l'accent sur la
lisibilité du code. Il est facile à apprendre et à utiliser pour les débutants en programmation,
mais aussi assez puissant pour être utilisé dans des projets complexes.
Python est un langage polyvalent, utilisé dans une grande variété d'applications, notamment pour la science des données,
l'apprentissage automatique, la création d'applications web, l'automatisation de tâches, les scripts système,
les jeux et bien plus encore.
Il a une grande communauté de développeurs qui ont créé une multitude de bibliothèques et de frameworks,
tels que Pandas, Numpy, Matplotlib, Django, Flask, etc., qui facilitent le développement de projets de différentes
tailles et complexités.\\
En outre, Python est un langage open-source, ce qui signifie que tout le monde peut l'utiliser, le modifier et le
distribuer librement. C'est un langage très populaire et largement utilisé dans l'industrie,
l'éducation et la recherche.

## Présentation des bibliothèques utilisées :

Pour effectuer ce travail nous avons utilisé différentes bibliothèques présentées ci-dessous.

### Pandas :

Pandas est une bibliothèque open-source pour Python qui permet de manipuler et d'analyser des données tabulaires.
Pandas fournit des fonctionnalités pour lire et écrire des données dans différents formats de fichiers, tels que CSV,
Excel, JSON, SQL et bien plus encore. Elle permet également de nettoyer et de préparer les données pour l'analyse,
notamment en gérant les valeurs manquantes et en effectuant des opérations de fusion, de regroupement et de filtrage.

### Streamlit :

Streamlit est une bibliothèque open-source de Python qui permet de créer facilement des applications web interactives
à partir de scripts Python. Elle fournit une interface simple et intuitive pour créer des applications web en utilisant
des commandes simples telles que "st.title", "st.write", "st.plot" et bien d'autres encore.
Streamlit permet aux utilisateurs de créer des applications web interactives à partir de scripts Python en quelques
minutes, sans avoir à apprendre des langages de programmation web tels que HTML, CSS ou JavaScript.
Elle prend également en charge les graphiques interactifs, les cartes, les animations, les widgets, les formulaires et
bien d'autres encore.
Streamlit est très populaire dans la communauté des scientifiques des données, car elle permet de créer des tableaux de
bord interactifs pour visualiser et explorer des données rapidement et facilement. Elle est également très utile pour
la création de prototypes, la démonstration de concepts, la formation et l'enseignement.

### Matplotlib :

Matplotlib est une bibliothèque de visualisation de données en Python qui permet de créer des graphiques statiques.\\
Matplotlib.pyplot, souvent abrégé en plt, est un module de Matplotlib qui fournit une interface de programmation
similaire à celle de MATLAB pour créer des graphiques en utilisant des commandes simples.

### Scikit-learn (sklearn) :

Scikit-learn est une bibliothèque Python largement utilisée pour l'apprentissage automatique et l'analyse de données.\\
Elle offre une large gamme d'algorithmes d'apprentissage automatique, de prétraitement des données, d'évaluation de
modèles, et des outils pour la sélection de modèles.\\
Scikit-learn facilite la création de modèles de régression, de classification, de clustering, et d'autres tâches
courantes en apprentissage automatique grâce à une interface conviviale et bien documentée.\\
Elle est souvent utilisée pour l'entraînement de modèles de machine learning, notamment la régression logistique,
les forêts aléatoires, les machines à vecteurs de support, etc.

### Seaborn :
Seaborn, d'autre part, est une bibliothèque Python de visualisation de données basée sur Matplotlib.\\
Elle simplifie la création de graphiques informatifs et esthétiquement plaisants en fournissant une interface de
haut niveau pour la création de graphiques statistiques.
\\Seaborn offre un ensemble de styles par défaut et de palettes de couleurs qui améliorent la lisibilité des graphiques,
ce qui en fait un choix populaire pour explorer et présenter des données.\\
Elle prend en charge divers types de graphiques, notamment les diagrammes en barres, les diagrammes à dispersion,
les diagrammes en boîte, les diagrammes de densité, et d'autres, ce qui en fait une bibliothèque précieuse pour
l'analyse exploratoire des données.\\
Ici, elle sera également utiliser pour récupérer les données (titanic, etc.)

## Présentation du code :

Le code est trouvable dans les fichiers main.py, constant.py et est commenté.\\
Il est également trouvable tout au long de ce site via l'option "Montrer le code".\\
Des différences peuvent exister entre les 2 versions soit à cause d'un oubli de mise à jour après modification,
soit pour des soucis de lisibilité et de compréhension pour la version en ligne. En règle générale, le code affichée
sur le site est écrit dans une fonction dans le fichier main.py.\\
**Vous pouvez trouver plus d'informations sur la structure du projet dans le fichier \"Rapport.docx\"**

Le fichier text.py permet d'afficher le texte dans les pages en allégeant le fichier main.py.\\
Le fichier code_text.py permet d'afficher le code dans les pages en allégeant le fichier main.py.
"""

presentation_data = """
## Description des données :

Nous utilisons le dataset "titanic" de la bibliothèque seaborn.\\
Cette ensemble est souvent utilisé pour l'apprentissage automatique ce qui nous as rassuré sur le choix de ce dataset.\\
De plus, il offre un aperçu fascinant des sur les passagers du RMS Titanic lors de son fatal voyage de 1912.\\
Maxime ayant visité l'exposition sur le titanic à Paris, c'est assez sympa de travailler sur ce sujet.

### Origine historique :
Les données proviennent du RMS Titanic qui est un paquebot transatlantique britannique qui a sombré
après avoir heurté un iceberg entraînant la mort de plus de 1 500 personnes sur 2 224.\\
Nous pensons généralement que les personnes de première classe ont pu survivre avec un accès prioritaire aux
bateaux, également, nous pensons que les enfants et les femmes ont eu la piorité également.\\
**La richesse donne-t-elle un droit de survie ou non ?**\\
**Est-ce que face à la détresse, la règle des enfants et les femmes d'abord a été appliquée ou non ?**

### Explications colonnes :
Il y a 3 variables catégorielles:
- **sex**.
- **class** qui est la classe.
- **embarked_town** qui est le port d'embarquement.
- **who**.
On peut comprendre **who** comme suit :
- **child** si l'individu était enfant (<18).
- **man** si l'individu était homme.
- **woman** si il/elle était une femme.

Et 5 variables quantitatives :
- **age**.
- **sibsp** : nombre de frères et sœurs vivants.
- **parch** : nombre de parents vivants.
- **fare** prix payé.
- **pclass**
- **alone**

Normalement, nous aurions dû encoder les variables catégorielles, mais le fichier de seaborn est déjà traité.\\
**pclass** correspond à la class :
- 1 -> *First*
- 2 -> *Second*
- 3 -> *Third*

**adult_male** :
- True -> Si c'est un adulte et un homme
- False -> Si c'est une femme ou un enfant.

**embarked** : Diminutif du port d'embarquement.
"""
