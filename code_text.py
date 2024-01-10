load_data = """
# Retourne les données au format DataFrame du titanic.
# L'utilisation de cache=False permet de récupérer les données quoi qu'il arrive,
# sinon, on risque d'avoir une erreur du type "this dataset doesn't exists".
return sns.load_dataset("titanic", cache=False)
"""
