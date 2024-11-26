import numpy as np
import matplotlib.pyplot as plt

# Definir el espacio de ángulos
angle_space = np.linspace(-np.pi, np.pi, 20)

# Crear una figura y un eje
fig, ax = plt.subplots(figsize=(6,6))
ax.set_aspect('equal')

# Graficar el círculo
circle = plt.Circle((0, 0), 1, color='lightblue', fill=True, alpha=0.5)
ax.add_artist(circle)

# Graficar los ángulos en el círculo
for angle in angle_space:
    x = np.cos(angle)
    y = np.sin(angle)
    ax.plot([0, x], [0, y], color='red')  # Línea desde el centro hasta el punto
    ax.scatter(x, y, color='red')  # Puntos donde caen los ángulos

# Establecer límites y etiquetas
ax.set_xlim(-1.5, 1.5)
ax.set_ylim(-1.5, 1.5)
ax.set_title('Ángulos generados por np.linspace(-π, π, 20)')
ax.grid(True)

# Mostrar la gráfica
plt.show()
