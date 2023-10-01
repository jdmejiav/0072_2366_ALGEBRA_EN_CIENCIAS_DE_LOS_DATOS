

from PIL import Image, ImageDraw
import random
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.vgg16 import preprocess_input
from tensorflow.keras.applications.vgg16 import VGG16
import numpy as np
from tensorflow.keras.preprocessing.image import array_to_img
import matplotlib.pyplot as plt


# Tamaño de la imagen
width = 400
height = 600

# Crea una nueva imagen RGB
imagen = Image.new("RGB", (width, height))
dibujo = ImageDraw.Draw(imagen)

# Rellena la imagen con píxeles de color aleatorio
for x in range(width):
    for y in range(height):
        # Genera un color aleatorio en formato RGB (0-255 para cada canal)
        color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
        dibujo.point((x, y), fill=color)

# Guarda la imagen generada
imagen.save("imagen_aleatoria.png")


# Carga una imagen
imagen_aleatoria = 'imagen_aleatoria.png'  # Reemplaza 'imagen.jpg' con la ruta de tu imagen
img = image.load_img(imagen_aleatoria, target_size=(600, 400))

# Convierte la imagen en un arreglo numpy
img_aleatoria_array = image.img_to_array(img)


# Carga una imagen
imagen_objetivo = 'image.jpeg'  # Reemplaza 'imagen.jpg' con la ruta de tu imagen
img = image.load_img(imagen_objetivo, target_size=(600, 400))

# Convierte la imagen en un arreglo numpy
img_objetivo_array = image.img_to_array(img)

n = int(input("Ingresa n deseado> "))


sucesion = [img_aleatoria_array]
for i in range(n):
    termino_actual = img_objetivo_array - (img_objetivo_array - sucesion[-1]) / 1.1 if sucesion else np.zeros_like(img_objetivo_array)
    sucesion.append(termino_actual)

imagen_reconstruida = array_to_img(sucesion[-1])
plt.imshow(imagen_reconstruida)
plt.show()

distancia_minima = 0.5

# Muestra los primeros 10 términos de la sucesión
for i in range(10):
    imagen_reconstruida = Image.fromarray(np.uint8(sucesion[i]))
    plt.imshow(imagen_reconstruida, cmap='gray')
    plt.title(f'Término {i + 1}')
    plt.show()

# Encuentra el valor de n para que la distancia sea menor a 0.5
for i, imagen in enumerate(sucesion):
    distancia = np.linalg.norm(img_objetivo_array - imagen)
    if distancia < distancia_minima:
        print("Valor para que n sea menor que 0.5 ", i)
        break


while True:
    iteracion = int(input("Paso específico> "))
    imagen_reconstruida = array_to_img(sucesion[iteracion-1])

    distancia = np.sum((img_objetivo_array - sucesion[iteracion - 1]) ** 2 ) ** (1/2)

    print(distancia)
    plt.imshow(imagen_reconstruida)
    plt.show()




