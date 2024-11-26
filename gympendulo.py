# Examen Final
#Universitario Chura Padilla Walberth Jesus
#CU. 104-468
import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
import pickle

# Función principal de entrenamiento y simulación
def run(is_training=True, render=False, max_episodes=100000, max_steps_per_episode=200):
    # Crea el entorno Pendulum, habilita la visualización si render es True
    env = gym.make('Pendulum-v1', render_mode='human' if render else None)

    accionTomada = ""
    # Define los límites de espacio para el ángulo y la velocidad angular
    angle_space = np.linspace(-np.pi, np.pi, 20)  # Espacio para el ángulo
    ang_vel_space = np.linspace(-4, 4, 20)        # Espacio para la velocidad angular

    # Inicializa la tabla Q con ceros, con dimensiones según el espacio de discretización y las acciones posibles
    n_actions = 20  # Número de acciones discretas (torques)
    q = np.zeros((len(angle_space) + 1, len(ang_vel_space) + 1, n_actions))

    # Define los parámetros de entrenamiento
    learning_rate_a = 0.20  # Tasa de aprendizaje alfa
    discount_factor_g = 0.99  # Factor de descuento para el futuro valor de beta
    epsilon = 1  # Valor inicial de epsilon (para exploración aleatoria)
    epsilon_decay_rate = 0.00001  # Tasa de decaimiento de epsilon
    rng = np.random.default_rng()  # Generador de números aleatorios para exploración

    rewards_per_episode = []  # Lista para almacenar recompensas de cada episodio

    # Bucle principal para cada episodio
    for i in range(max_episodes):
        
        # Inicializa el estado y lo discretiza
        state, _ = env.reset()
        state_a = np.digitize(state[0], angle_space)  # Índice del ángulo
        state_av = np.digitize(state[1], ang_vel_space)  # Índice de la velocidad angular

        terminated = False  # Marca de finalización del episodio
        rewards = 0  # Acumulador de recompensa en el episodio
        steps = 0  # Contador de pasos por episodio
        vecesArriba = 0
        vecesAbajo = 0

        # Bucle de interacción con el entorno hasta que termine el episodio o se alcance el límite de pasos
        while not terminated and steps < max_steps_per_episode:
            reward = 0
            # Selección de acción: exploración o explotación
            if is_training and rng.random() < epsilon:
                accionTomada = "Exploracion"
                action = np.random.choice(n_actions)  # Exploración: elige acción aleatoria
            else:
                accionTomada = "Explotacion"
                action = np.argmax(q[state_a, state_av, :])  # Explotación: elige la mejor acción según Q

            # Mapea la acción discreta a un torque continuo
            torque = np.linspace(-2.0, 2.0, n_actions)[action]

            # Ejecuta la acción en el entorno y observa el nuevo estado y recompensa
            new_state, _, terminated, truncated, _ = env.step([torque])

            # Calcula la nueva recompensa personalizada
            angle = new_state[0]  # Ángulo
            ang_vel = new_state[1]  # Velocidad angular

            # Lógica de recompensa según la ubicación del péndulo en el círculo
            sin_angle = np.sin(angle)  # Usamos el seno del ángulo para determinar la posición en el círculo

            if sin_angle > 0.8:
                steps -= 1
                vecesArriba += 1
                reward += 10.0
            elif sin_angle > 0.6:  # Si el seno está cercano a 1 (parte superior)
                reward += 1.0  # Recompensa alta cuando el péndulo está arriba
            elif sin_angle < -0.8:  # Si el seno está cercano a -1 (parte inferior)
                steps += 1
                reward -= 5.0
                vecesAbajo += 1
            elif sin_angle < -0.6:  # Si el seno está cercano a -1 (parte inferior)
                reward -= 2  # Penalización fuerte cuando el péndulo está en la parte inferior
            else:  # Si está cerca de los lados
                reward -= 0.5  # Penalización moderada en los lados

            # Penalización por velocidad angular
            reward -= 0.1 * np.abs(ang_vel)

            # Discretiza el nuevo estado
            new_state_a = np.digitize(new_state[0], angle_space)
            new_state_av = np.digitize(new_state[1], ang_vel_space)

            # Actualización de la tabla Q si está en modo entrenamiento
            if is_training:
                q[state_a, state_av, action] += learning_rate_a * (
                    reward + discount_factor_g * np.max(q[new_state_a, new_state_av, :])
                    - q[state_a, state_av, action]
                )
                

            # Actualiza el estado actual y acumula la recompensa
            state = new_state
            state_a = new_state_a
            state_av = new_state_av
            rewards += reward
            steps += 1  # Incrementa el contador de pasos

        # Guarda la recompensa total del episodio
        rewards_per_episode.append(rewards)
        # Calcula la media de recompensas de los últimos 100 episodios
        mean_rewards = np.mean(rewards_per_episode[max(0, len(rewards_per_episode)-100):])

        # Imprime información de progreso cada 100 episodios durante el entrenamiento
        if is_training and i % 100 == 0:
            print(f'episodio: {i}  recompensa: {rewards:0.2f}  Epsilon: {epsilon:0.2f}  recompensa media: {mean_rewards:0.2f} accionTomada:{accionTomada}')

        # Decaimiento de epsilon para disminuir exploración con el tiempo
        epsilon = max(epsilon - epsilon_decay_rate, 0.1)
        env.close()  # Cierra el entorno para liberar recursos

        # Determina si el episodio debe renderizarse
        current_render = render or (is_training and i % 500 == 0)
        if current_render:
            env = gym.make('Pendulum-v1', render_mode='human')
            print(f'recompensa:{rewards:0.2f} episodio: {i} Epsilon: {epsilon:0.2f} accion:{accionTomada} vecesArriba:{vecesArriba} vecesAbajo:{vecesAbajo}')
        else:
            env = gym.make('Pendulum-v1')

    # Guarda la tabla Q en un archivo si está en modo entrenamiento
    if is_training:
        with open('pendulum.pkl', 'wb') as f:
            pickle.dump(q, f)

    # Calcula y grafica la media de recompensas para observar la evolución del entrenamiento
    mean_rewards = [np.mean(rewards_per_episode[max(0, t-100):(t+1)]) for t in range(len(rewards_per_episode))]
    plt.plot(mean_rewards)
    plt.xlabel('Episode')
    plt.ylabel('Mean Reward (last 100 episodes)')
    plt.title('Training Progress')
    plt.savefig('pendulum.png')  # Guarda la gráfica como imagen
    plt.show()  # Muestra la gráfica después de guardarla

# Ejecuta la función principal de entrenamiento y simulación
if __name__ == '__main__':
    run(is_training=True, render=False, max_episodes=100000, max_steps_per_episode=200)  # Entrenamiento