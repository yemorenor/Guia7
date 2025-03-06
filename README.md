import numpy as np

# Parámetros del problema
capacidad_plantas = [3, 6, 5, 4]
demanda_ciudades = [4, 3, 5, 3]
costos_transporte = np.array([
    [1, 4, 3, 6],
    [4, 1, 4, 5],
    [3, 4, 1, 4],
    [6, 5, 4, 1]
])
costos_generacion = [680, 720, 660, 750]  # $/GW

# Configuración AG
POBLACION = 100
GENERACIONES = 200
PROB_CRUCE = 0.8
PROB_MUTACION = 0.05

def crear_individuo():
    # Generar matriz 4x4 que cumpla restricciones de capacidad
    individuo = np.zeros((4,4))
    for i in range(4):
        total = 0
        for j in range(4):
            max_posible = min(capacidad_plantas[i] - total, demanda_ciudades[j])
            if max_posible <= 0:
                individuo[i,j] = 0
                continue
            individuo[i,j] = np.random.uniform(0, max_posible)
            total += individuo[i,j]
    return individuo

def calcular_aptitud(individuo):
    # Verificar restricciones
    penalizacion = 0
    # Verificar capacidad plantas
    for i in range(4):
        suma = individuo[i,:].sum()
        if suma > capacidad_plantas[i]:
            penalizacion += 1e6 * (suma - capacidad_plantas[i])

    # Verificar demanda ciudades
    for j in range(4):
        suma = individuo[:,j].sum()
        if suma < demanda_ciudades[j]:
            penalizacion += 1e6 * (demanda_ciudades[j] - suma)

    # Calcular costos
    costo_transporte = (individuo * costos_transporte).sum()
    costo_generacion = sum([individuo[i,:].sum() * costos_generacion[i] for i in range(4)])

    return -(costo_transporte + costo_generacion + penalizacion)  # Minimizar

def seleccion(poblacion, aptitudes):
    # Selección por ruleta
    probabilidades = aptitudes - aptitudes.min()
    if probabilidades.sum() == 0:
        return poblacion[np.random.choice(len(poblacion))]
    probabilidades /= probabilidades.sum()
    return poblacion[np.random.choice(len(poblacion), p=probabilidades)]

def cruce(padre1, padre2):
    # Cruce por intercambio de filas
    if np.random.random() > PROB_CRUCE:
        return padre1.copy(), padre2.copy()

    punto = np.random.randint(1,3)
    hijo1 = np.vstack((padre1[:punto], padre2[punto:]))
    hijo2 = np.vstack((padre2[:punto], padre1[punto:]))
    return hijo1, hijo2

def mutar(individuo):
    # Mutación aleatoria en una celda
    if np.random.random() < PROB_MUTACION:
        i, j = np.random.randint(4), np.random.randint(4)
        max_val = min(capacidad_plantas[i], demanda_ciudades[j])
        individuo[i,j] = np.random.uniform(0, max_val)
    return individuo

# Algoritmo principal
poblacion = [crear_individuo() for _ in range(POBLACION)]
mejor_aptitud = -float('inf')

for gen in range(GENERACIONES):
    aptitudes = np.array([calcular_aptitud(ind) for ind in poblacion])

    # Registro del mejor
    mejor_idx = aptitudes.argmax()
    if aptitudes[mejor_idx] > mejor_aptitud:
        mejor_aptitud = aptitudes[mejor_idx]
        mejor_individuo = poblacion[mejor_idx].copy()

    # Nueva generación
    nueva_poblacion = []
    for _ in range(POBLACION//2):
        padre1 = seleccion(poblacion, aptitudes)
        padre2 = seleccion(poblacion, aptitudes)
        hijo1, hijo2 = cruce(padre1, padre2)
        nueva_poblacion.extend([mutar(hijo1), mutar(hijo2)])

    poblacion = nueva_poblacion

print("Mejor solución encontrada:")
print(mejor_individuo)
print(f"Costo total: ${-mejor_aptitud:.2f}")
