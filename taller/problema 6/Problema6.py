import numpy as np
'''
h2 = [[1/2,1/3,1/4],
      [1/3,1/4,1/5],
      [1/4,1/5,1/6]]
'''

h2 = [[7,23],[4,89]]

test_cases = [[1,0],  [0,1]]


p = 1
dimension = len(h2)

# Generar un vector aleatorio con valores en una distribución normal estándar

for i in range(5):
    vector_aleatorio = np.random.randn(dimension)

    # Calcular la norma euclidiana del vector
    norma = np.linalg.norm(vector_aleatorio, ord = p)

    # Normalizar el vector dividiendo por su norma
    vector_normalizado = vector_aleatorio / norma
    test_cases.append(vector_normalizado)

max = 0

for vec in test_cases:
    producto = np.dot(h2,vec)
    resultado = np.linalg.norm(producto, ord = p)
    print(resultado)

    if (max < resultado):
        max = resultado

print(max)
