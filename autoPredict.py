# IMPORT
from keras_preprocessing.image import ImageDataGenerator
import pandas as pd
from keras.models import load_model
import json




# Chargement du model
model = load_model('./trainedModel/model.hdf5')

df = pd.read_csv("./dataTest/imgTest.txt")

# Permet de normaliser nos imades d'entrées
test_datagen=ImageDataGenerator(rescale=1./255.)

# Chargement des images de test
test_generator=test_datagen.flow_from_dataframe(
                                    dataframe=df,
                                    directory="./dataTest/",
                                    x_col="path",
                                    batch_size=1,
                                    seed=42,
                                    shuffle=False,
                                    class_mode=None,
                                    target_size=(96,96))


STEP_SIZE_TEST=test_generator.n//test_generator.batch_size

# Reset obligatoire pour avoir le bon ordre des outputs
test_generator.reset()


pred=model.predict_generator(   test_generator,
                                steps=STEP_SIZE_TEST,
                                verbose=1)

# Interupteur pour lever la prediction si on a au moins 10% de confidence
booleanPrediction = (pred > 0.1)


listPrediction=[]
#On récuperer le dictionnaire JSON des labels avec leur indice respectif
labelsList = json.load(open("./classIndice.txt"))
labelsList = dict((v,k) for k,v in labelsList.items())


for img in booleanPrediction:
    correctPredictList=[]
    for index,cls in enumerate(img):
        if cls:
            correctPredictList.append(labelsList[index])
            print(cls)

    listPrediction.append(",".join(correctPredictList))

# Tableau contenant l'ensemble des chemin des images
pathImg=test_generator.filenames

# On creer un dataframe pour concat les images avec leurs predictions correctes
results=pd.DataFrame({"Chemin img":pathImg,
                      "Predictions":listPrediction})
# On affiche les résultats
print(results)
# On Save les resultats dans un fichier CSV
results.to_csv("results.csv",index=False)