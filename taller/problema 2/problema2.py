### El clasificador de KNN de scikit learn, utiliza por defecto para comparar la 
### distancia entre dos imágenes, la distancia euclidiana. Esta distancia es adecuada
### para calcular la distancia entre imágnenea, ya que al representarse como vectores,
### esta mide la distancia entre los valores de los pixeles de las imágenes y esto
### es util para identificar imágenes que se asemejen 

### PARA EJECUTAR DEBE SER MEDIANTE LÍNEA DE COMANDOS Y DEBE SER DE LA SIGUIENTE MANERA
### python problema2.py i Fotos
### SE DEBE SUMINISTRAR LA RUTA DEL DATASET DE FOTOS


import argparse
import os

import cv2
import numpy as np
from imutils import paths
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder
from skimage import io, color, measure

# Definimos los parámetros de entrada:
argument_parser = argparse.ArgumentParser()
argument_parser.add_argument('-i', '--input', required=True,
                             help='Ruta al directorio donde se hallan las imágenes del conjunto de datos.')
argument_parser.add_argument('-k', '--neighbors', type=int, default=5,
                             help='Número de vecinos a tomar en cuenta por el algoritmo.')
argument_parser.add_argument('-j', '--jobs', type=int, default=1,
                             help='Número de hilos a usar por k-NN (-1 usa todo los cores disponibles).')
arguments = vars(argument_parser.parse_args())

# Asumimos que las imágenes están contenidas en subdirectorios correspondientes a su categoría.
# Esta línea nos permite cargar las rutas a todas estas imágenes en memoria.
image_paths = list(paths.list_images(arguments['input']))

# Cargamos las imágenes y etiquetas, iterando por cada ruta en image_paths.
data = []
labels = []
for index, image_path in enumerate(image_paths):
    # Cargamos la imagen.
    image = cv2.imread(image_path)
    # Extraemos la etiqueta del nombre del subdirectorio (male o female).
    label = image_path.split(os.path.sep)[-2]

    # Redimensiona la imagen a 32x32x3
    image = cv2.resize(image, (32, 32), interpolation=cv2.INTER_AREA)

    # Añade la imagen y su etiqueta a las correspondientes listas.
    data.append(image)
    labels.append(label)

    # Reportamos el progreso cada 100 imágenes.
    if (index + 1) % 100 == 0:
        print(f'Procesadas {index + 1}/{len(image_paths)} imágenes.')

# Convertimos las listas en arreglos de Numpy.
data = np.array(data)
labels = np.array(labels)

# Vectorizamos
data = data.reshape((data.shape[0], np.prod(data.shape[1:])))

print(f'La matriz de datos pesa: {data.nbytes / (1024 * 1024.0)}MB')

# A partir de aquí entrenaremos el algoritmo K-Nearest Neighbors. Empezaremos convirtiendo las etiquetas en una representación
# numérica utilizable por el algoritmo, gracias a LabelEncoder.
label_encoder = LabelEncoder()
labels = label_encoder.fit_transform(labels)

# Usaremos 20% de los datos para prueba, y 80% para entrenar.
X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, random_state=42)

# Entrenamos el modelo.
model = KNeighborsClassifier(n_neighbors=arguments['neighbors'], n_jobs=arguments['jobs'])
model.fit(X_train, y_train)

# Imprimimos el reporte de clasificación.
print(classification_report(y_test, model.predict(X_test), target_names=label_encoder.classes_))




while True:
    image_path = input("Digita ruta de imagen de a clasificar> ")
    if (image_path == ''):
        break
    image = cv2.imread(image_path)

    # Asegúrate de que la imagen tenga las mismas dimensiones que las imágenes de entrenamiento (32x32 píxeles)
    image = cv2.resize(image, (32, 32), interpolation=cv2.INTER_AREA)

    # Vectoriza la imagen
    image_vector = image.reshape(1, -1)

    # Realiza una predicción utilizando el modelo entrenado
    predicted_label = model.predict(image_vector)

    # Convierte la etiqueta predicha de nuevo a su representación original si es necesario
    predicted_class = label_encoder.inverse_transform(predicted_label)

    print(f"La imagen se clasifica como: {predicted_class[0]}")
