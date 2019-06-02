# Recommendeur de hashtag pour image

Permet de creer avec Tensorflow et Keras un recommandeur de hastag pour image, avec des algorithmes de deep learning.

## Installer les pré-requis
Je vous laisse installer vous même les libraires nécessaires, par conda ou pip selon vos préferences et environnements
- Tensorflow
- Keras
- Pandas
- TQDM
- PIL

Téléchargez le dataset HARRISON, le dézipper dans ./HARRISSON/.


## Customization des hyper paramètres
Disponible dans le fichier trainModel.py :

| Attribut | Description                    |
| ------------- | ------------------------------ |
| `CLASSE`      |  Lien vers le fichier contenant la liste totales des classes avec leurs indices respectifs  |
| `NB_CLASSES`   |  Nombre de classe ; définie le nombre de neuronnes sur le dernier layer     |
| `NB_EPOCH`   | Nombre d'iteration pour l'entrainement    |
| `BATCH_SIZE`   | Nombre d'image par lot    |
| `SHUFFLE`   | Permet de melanger notre dataset    |
| `IMG_SIZE`   | Taille des images à resize    |
| `TRAINSIZE_RATIO`   | Ratio pour gerer la taille des jeux d'entrainement et de validation    |
| `DIRECTORY_DATA`   | Chemin parent vers les images    |
| `DIRECTORY_TRAINED_MODEL`   | Chemin ou save le modele    |
| `COLOR_MODE`   | Choix des canaux des images : couleur ou noir/blanc    |


## Générer et entrainer un model
`$ python trainModel.py`

## Prédiction de nouvelles images
Donnez les chemins d'accès à vos photos tests dans le fichier ./dataTest/imgTest.txt, puis lancez :
`$ python autoPredic.py`
