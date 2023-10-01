import numpy as np
def generar_vector_minkowski(n, p):
    # Paso 1: Genera un vector aleatorio en [-0.1, 0.1] en cada componente.
    vector = np.random.uniform(-0.1, 0.1, n)
    
    # Paso 2: Normaliza el vector para que su norma p sea 0.1.
    norma = np.linalg.norm(vector, ord=p)
    vector_normalizado = vector / norma * 0.1
    
    return vector_normalizado

# Ejemplo de uso:
n = int(input("Ingrese n: "))
p = int(input("Ingrese p: "))

vector_generado = generar_vector_minkowski(n, p)
print("Vector generado:", vector_generado)
print("Norma p del vector generado:", np.linalg.norm(vector_generado, ord=p))