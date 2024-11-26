import pickle
import numpy as np
import matplotlib.pyplot as plt

# Función para cargar la tabla Q desde el archivo pickle
def cargar_tabla_q(archivo):
    with open(archivo, 'rb') as f:
        q = pickle.load(f)
    return q

# Función para graficar la tabla Q como bloques
def graficar_tabla_q_en_bloques(q, angle_space, ang_vel_space):
    # Crear figura para el gráfico
    fig, ax = plt.subplots(figsize=(14, 10))

    # Reshape de la tabla Q para que cada fila/columna sea un bloque para una acción
    # q tiene dimensiones (ángulo, velocidad angular, acciones), así que necesitamos graficar
    # cada acción por separado

    n_actions = q.shape[2]

    # Definir los colores y mapa para el heatmap
    cmap = 'plasma'  # Mejor contraste visual con una gama de colores más rica

    # Para cada acción, graficamos una "subtabla" de Q
    for action_idx in range(n_actions):
        # Graficar los valores de Q para esta acción específica
        q_action = q[:, :, action_idx]

        # Crear un mapa de calor para cada acción
        cax = ax.imshow(q_action, cmap=cmap, interpolation='nearest', alpha=0.8, origin='lower', aspect='auto')

        # Añadir líneas divisorias entre los bloques para mejor visualización
        ax.grid(color='white', linestyle='-', linewidth=1)

        # Añadir una leyenda con el torque asociado a esta acción
        torque = np.linspace(-2.0, 2.0, n_actions)[action_idx]
        ax.text(0.5, 1.05, f'Acción: {torque:.2f} Torque', ha='center', va='bottom', fontsize=14, color='black')

    # Añadir título y etiquetas
    ax.set_title('Tabla Q en Bloques: Decisiones del Agente', fontsize=18, fontweight='bold', color='darkblue')
    ax.set_xlabel('Velocidad Angular (rad/s)', fontsize=14, fontweight='bold')
    ax.set_ylabel('Ángulo (radianes)', fontsize=14, fontweight='bold')

    # Establecer los ticks de los ejes
    ax.set_xticks(np.arange(0, q.shape[1], step=3))
    ax.set_yticks(np.arange(0, q.shape[0], step=3))
    ax.set_xticklabels(np.round(ang_vel_space[::3], 2), fontsize=12)
    ax.set_yticklabels(np.round(angle_space[::3], 2), fontsize=12)

    # Añadir la barra de color para los valores de Q
    cbar = fig.colorbar(cax, ax=ax, label='Valor de Q', shrink=0.8)
    cbar.ax.tick_params(labelsize=12)

    # Mejorar el layout y hacer que se vea bien en pantalla
    plt.tight_layout()

    # Mostrar el gráfico
    plt.show()

# Función principal
def main():
    # Cargar la tabla Q desde el archivo pickle
    q = cargar_tabla_q('pendulum.pkl')

    # Espacios de ángulo y velocidad angular
    angle_space = np.linspace(-np.pi, np.pi, q.shape[0])
    ang_vel_space = np.linspace(-4, 4, q.shape[1])

    # Graficar la tabla Q en bloques
    graficar_tabla_q_en_bloques(q, angle_space, ang_vel_space)

# Ejecutar el código
if __name__ == '__main__':
    main()
