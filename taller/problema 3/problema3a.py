import numpy as np
import matplotlib.pyplot as plt



# Definir una cuadrícula de puntos en el plano
x = np.linspace(-2.5, 2.5, 500)
y = np.linspace(-2.5, 2.5, 500)
X, Y = np.meshgrid(x, y)


while True:
    r = float(input("Valor de r: "))
    p = float(input("Valor de p: "))
    if (r > 2 or r < 0.1):
        print("Valor de r inválido ingresa un valor entre 0.1 y 2")
    if (p<1):
        print("Valor de p inválido ingresa un valor desde 1 a ∞")

    D = (np.abs(X)**p + np.abs(Y)**p)**(1/p)

    # Filtar puntos dentro de la bola (distancia <= r)
    inside_ball = D <= r

    plt.figure(figsize=(6, 6))
    plt.title(f'Bola de radio {r} (Minkowski p={p})')

    # Dibujar la bola
    plt.imshow(inside_ball, extent=(-2.5, 2.5, -2.5, 2.5), origin='lower', cmap='binary')
    plt.colorbar()
    plt.gca().set_aspect('equal', adjustable='box')
    plt.xlabel('Eje X')
    plt.ylabel('Eje Y')

    # Guardar la figura en un archivo (opcional)
    plt.savefig(f'bola_{r}_p_{p}.png')

    # Mostrar las figuras
    plt.show()
