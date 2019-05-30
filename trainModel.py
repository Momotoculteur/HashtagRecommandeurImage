from keras import Sequential
from keras.applications import VGG16, MobileNetV2
from keras.applications.inception_resnet_v2 import InceptionResNetV2
from keras.layers.core import Dense, Flatten, Dropout
from keras.models import Model
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ModelCheckpoint, EarlyStopping
import pandas as pd
from tqdm import tqdm
from keras.optimizers import *
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
import json

# Lecture du fichier des classes
CLASSE = pd.read_csv("./HARRISON/listClass.txt",sep=',', names=["classe", "index"])

# Lecture du DF des datas
df = pd.read_csv("./HARRISON/data.txt")
# Si nbLabels >1, conversion en liste
# Permet au réseau de comprendre qu'il y a plusieurs entrée concernant une même image
df["labels"]=df["labels"].apply(lambda x:x.split(","))

# GLOBAL VAR

#NB_CLASSES = CLASSE.shape[0] 998 classe original
NB_CLASSES = 994 # Permet de fix le nombre de classe manquante du dataset
NB_EPOCH = 1
BATCH_SIZE = 32
SHUFFLE = True
IMG_SIZE = (96,96)
TRAINSIZE_RATIO = 0.8
TRAINSIZE = int(df.shape[0] * TRAINSIZE_RATIO)
LIST_CLASS = []
DIRECTORY_DATA = "./HARRISON/"
DIRECTORY_TRAINED_MODEL = './trainedModel/model.hdf5'
COLOR_MODE ="rgb"

# Chargement des noms classes
for index, row in tqdm(CLASSE.iterrows(), total=CLASSE.shape[0]):
    LIST_CLASS.append(row['classe'])


# DEFINITION DES CALLBACKS
save_model_callback = ModelCheckpoint(DIRECTORY_TRAINED_MODEL,
                                      verbose=1,
                                      save_best_only=True,
                                      save_weights_only=False,
                                      mode='auto',
                                      period=1,
                                      monitor='val_acc')

early_stopping = EarlyStopping(verbose=1,monitor='val_acc', min_delta=0, patience=3, mode='auto')




# Normalisation des images
datagen=ImageDataGenerator(rescale=1./255.)
test_datagen=ImageDataGenerator(rescale=1./255.)


# Contient les images pour le jeu d'entrainement
train_generator=datagen.flow_from_dataframe(dataframe=df[:TRAINSIZE],
                                            directory=DIRECTORY_DATA,
                                            x_col="path",
                                            y_col="labels",
                                            batch_size=BATCH_SIZE,
                                            seed=42,
                                            shuffle=SHUFFLE,
                                            class_mode="categorical",
                                            classes=LIST_CLASS,
                                            target_size=IMG_SIZE,
                                            color_mode=COLOR_MODE)

# Contient les images pour le jeu de validation
valid_generator=test_datagen.flow_from_dataframe(dataframe=df[TRAINSIZE:],
                                                 directory=DIRECTORY_DATA,
                                                 x_col="path",
                                                 y_col="labels",
                                                 batch_size=BATCH_SIZE,
                                                 seed=42,
                                                 shuffle=SHUFFLE,
                                                 class_mode="categorical",
                                                 classes=LIST_CLASS,
                                                 target_size=IMG_SIZE,
                                                 color_mode=COLOR_MODE)




# Permet de sauvegarder les indices de labels de classe
# Utilisé lors de prediction de nouveaux fichiers
labels = train_generator.class_indices
with open('./classIndice.txt', 'w') as file:
    file.write(json.dumps(labels))





# On récup le model InceptionResNetV2 pré train sur imagenet
#baseModel = InceptionResNetV2(include_top=False, weights='imagenet', input_tensor=None, input_shape=(150,150,3), pooling='max')
baseModel = MobileNetV2(input_shape=(96,96,3), alpha=1.0, include_top=False, weights='imagenet', input_tensor=None, pooling='max')

# Permet de geler les couches basses pour transfer learning
for layer in baseModel.layers:
    layer.trainable=False


# On ajoute notre layer de classification
topModel = Dense(4096,activation='relu', trainable=True)(baseModel.output)
topModel = Dense(NB_CLASSES,activation='sigmoid', trainable=True)(topModel)
# On joint nos deux parties pour former un unique model
model = Model(inputs=baseModel.input, outputs=topModel)


# Compilation du model
model.compile(Adamax(lr=0.001, beta_1=0.9, beta_2=0.999, decay=0.0),loss='binary_crossentropy',metrics=['accuracy'])

# COMPOSITION du modele
model.summary()


STEP_SIZE_TRAIN=train_generator.n//train_generator.batch_size
STEP_SIZE_VALID=valid_generator.n//valid_generator.batch_size

# Entrainement du modele
model.fit_generator(generator=train_generator,
                    steps_per_epoch=STEP_SIZE_TRAIN,
                    validation_data=valid_generator,
                    validation_steps=STEP_SIZE_VALID,
                    epochs=NB_EPOCH,
                    callbacks=[early_stopping,save_model_callback],
                    verbose=1
)

# Evaluation du modele
model.evaluate_generator(generator=valid_generator,
steps=STEP_SIZE_VALID)





