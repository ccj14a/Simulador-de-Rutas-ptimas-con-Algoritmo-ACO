import random
import numpy as np
import pandas as pd
import folium
import time

# Lista de distritos de Lima con sus coordenadas
distritos_lima = {
    "Ate": (-12.0393, -76.9387),
    "Barranco": (-12.1439, -77.0204),
    "Breña": (-12.0532, -77.0469),
    "Carabayllo": (-11.8527, -77.0147),
    "Chaclacayo": (-11.9822, -76.7721),
    "Chorrillos": (-12.1641, -77.0212),
    "Cieneguilla": (-12.0374, -76.8432),
    "Comas": (-11.9574, -77.0451),
    "El Agustino": (-12.0517, -76.9944),
    "Independencia": (-11.9811, -77.0595),
    "Jesús María": (-12.0852, -77.0495),
    "La Molina": (-12.084, -76.9485),
    "La Victoria": (-12.0699, -77.0215),
    "Lima": (-12.0464, -77.0428),
    "Lince": (-12.0845, -77.0335),
    "Los Olivos": (-11.9724, -77.0921),
    "Lurigancho (Chosica)": (-11.9497, -76.7083),
    "Lurín": (-12.2754, -76.8699),
    "Magdalena del Mar": (-12.0913, -77.0727),
    "Miraflores": (-12.1218, -77.0297),
    "Pachacámac": (-12.1881, -76.8488),
    "Pucusana": (-12.4887, -76.7867),
    "Pueblo Libre": (-12.0756, -77.0707),
    "Puente Piedra": (-11.872, -77.0589),
    "Punta Hermosa": (-12.3102, -76.8393),
    "Punta Negra": (-12.3654, -76.7979),
    "Rímac": (-12.0285, -77.0274),
    "San Bartolo": (-12.3929, -76.7936),
    "San Borja": (-12.1064, -77.0015),
    "San Isidro": (-12.0975, -77.0364),
    "San Juan de Lurigancho": (-11.9674, -76.9918),
    "San Juan de Miraflores": (-12.1598, -76.9721),
    "San Luis": (-12.0759, -76.9968),
    "San Martín de Porres": (-12.0112, -77.0605),
    "San Miguel": (-12.0789, -77.0834),
    "Santa Anita": (-12.0521, -76.9647),
    "Santa María del Mar": (-12.4567, -76.7956),
    "Santa Rosa": (-11.8215, -77.1585),
    "Santiago de Surco": (-12.1372, -76.9893),
    "Surquillo": (-12.1179, -77.0219),
    "Villa El Salvador": (-12.1933, -76.9289),
    "Villa María del Triunfo": (-12.1523, -76.9454),
    "Ancón": (-11.7201, -77.1587),
    # Distritos de Callao
    "Callao": (-12.0566, -77.1181),
    "Bellavista": (-12.0679, -77.1288),
    "Carmen de la Legua Reynoso": (-12.0457, -77.1037),
    "La Perla": (-12.0725, -77.1127),
    "La Punta": (-12.0673, -77.1596),
    "Ventanilla": (-11.8775, -77.1481),
    "Mi Perú": (-11.8611, -77.1274),
}


# Función para seleccionar distritos aleatorios
def seleccionar_distritos_aleatorios(distritos, num_distritos):
    seleccionados = random.sample(list(distritos.keys()), num_distritos)
    return seleccionados


# Calcular distancias euclidianas entre distritos
def calcular_distancias(distritos):
    coordenadas = np.array(list(distritos.values()))
    num_distritos = len(coordenadas)
    distancias = np.zeros((num_distritos, num_distritos))

    for i in range(num_distritos):
        for j in range(num_distritos):
            if i != j:
                lat1, lon1 = coordenadas[i]
                lat2, lon2 = coordenadas[j]
                # Convertir distancias de grados a kilómetros
                distancia_km = np.sqrt(
                    (lat1 - lat2) ** 2 * (111.32**2) + (lon1 - lon2) ** 2 * (85.39**2)
                )
                distancias[i][j] = distancia_km

    return distancias


# Algoritmo de Colonia de Hormigas
class ACO:
    def __init__(
        self, distancias, num_hormigas, num_iteraciones, alfa=1, beta=2, rho=0.5
    ):
        self.distancias = distancias
        self.num_hormigas = num_hormigas
        self.num_iteraciones = num_iteraciones
        self.alfa = alfa
        self.beta = beta
        self.rho = rho
        self.num_ciudades = len(distancias)
        self.feromonas = np.ones((self.num_ciudades, self.num_ciudades))
        self.resultados = []

    def construir_ruta(self):
        ruta = []
        visitados = set()
        ciudad_inicial = random.randint(0, self.num_ciudades - 1)
        ruta.append(ciudad_inicial)
        visitados.add(ciudad_inicial)

        for _ in range(self.num_ciudades - 1):
            ciudad_actual = ruta[-1]
            probabilidades = self.calcular_probabilidades(ciudad_actual, visitados)
            if probabilidades.sum() == 0:  # Si no hay probabilidades válidas, salir
                break
            siguiente_ciudad = np.random.choice(
                range(self.num_ciudades), p=probabilidades
            )
            ruta.append(siguiente_ciudad)
            visitados.add(siguiente_ciudad)

        # Agregar el regreso a la ciudad inicial para cerrar el ciclo
        ruta.append(ciudad_inicial)
        return ruta

    def calcular_probabilidades(self, ciudad_actual, visitados):
        probabilidades = np.zeros(self.num_ciudades)

        for ciudad in range(self.num_ciudades):
            if ciudad not in visitados:
                if (
                    self.distancias[ciudad_actual][ciudad] > 0
                ):  # Asegurarse de que la distancia no sea cero
                    probabilidades[ciudad] = (
                        self.feromonas[ciudad_actual][ciudad] ** self.alfa
                    ) * ((1.0 / self.distancias[ciudad_actual][ciudad]) ** self.beta)

        # Normalizar las probabilidades
        probabilidades_sum = probabilidades.sum()
        if probabilidades_sum == 0:
            return np.zeros(
                self.num_ciudades
            )  # Retorna un array de ceros si no hay probabilidades válidas
        return probabilidades / probabilidades_sum

    def actualizar_feromonas(self, rutas):
        self.feromonas *= 1 - self.rho  # Evaporación de feromonas
        for ruta in rutas:
            distancia_total = sum(
                self.distancias[ruta[i], ruta[i + 1]] for i in range(len(ruta) - 1)
            )
            for i in range(len(ruta) - 1):
                self.feromonas[ruta[i], ruta[i + 1]] += 1 / distancia_total

    def ejecutar(self):
        mejor_ruta = None
        mejor_distancia = float("inf")

        for iteracion in range(self.num_iteraciones):
            print(f"\nIteración {iteracion + 1}:")
            rutas = [self.construir_ruta() for _ in range(self.num_hormigas)]
            for num_hormiga, ruta in enumerate(rutas):
                distancia_total = sum(
                    self.distancias[ruta[i], ruta[i + 1]] for i in range(len(ruta) - 1)
                )
                print(
                    f"  Hormiga {num_hormiga + 1}: Ruta: {[distritos_seleccionados[i] for i in ruta]}, Distancia: {distancia_total:.2f}"
                )
                self.resultados.append(
                    {
                        "Iteración": iteracion + 1,
                        "Hormiga": num_hormiga + 1,
                        "Ruta": [distritos_seleccionados[i] for i in ruta],
                        "Distancia": distancia_total,
                    }
                )
                if distancia_total < mejor_distancia:
                    mejor_distancia = distancia_total
                    mejor_ruta = ruta

            self.actualizar_feromonas(rutas)

        return mejor_ruta, mejor_distancia


# Seleccionar 10 distritos aleatorios
num_distritos = 15
distritos_seleccionados = seleccionar_distritos_aleatorios(
    distritos_lima, num_distritos
)

# Calcular distancias entre distritos seleccionados
distancias = calcular_distancias(
    {distrito: distritos_lima[distrito] for distrito in distritos_seleccionados}
)

# Ejecutar ACO
aco = ACO(distancias, num_hormigas=10, num_iteraciones=10)

start_time = time.time()
mejor_ruta, mejor_distancia = aco.ejecutar()  # Única ejecución
end_time = time.time()

# Guardar resultados en un archivo Excel
df_resultados = pd.DataFrame(aco.resultados)
df_resultados.to_excel("datos.xlsx", index=False)

# Imprimir resultados finales
print("\nMejor ruta:", [distritos_seleccionados[i] for i in mejor_ruta])
print("Distancia total:", mejor_distancia)
print(f"Tiempo de ejecución: {end_time - start_time:.2f} segundos")

# Crear un mapa centrado en Lima
# Crear un mapa centrado en Lima
mapa_lima = folium.Map(location=(-12.0464, -77.0428), zoom_start=12)

# Agregar marcadores para los distritos seleccionados
for distrito in distritos_seleccionados:
    folium.Marker(
        location=distritos_lima[distrito],
        popup=distrito,
    ).add_to(mapa_lima)

# Agregar marcadores para el inicio y fin de la mejor ruta
inicio_distrito = distritos_seleccionados[mejor_ruta[0]]
fin_distrito = distritos_seleccionados[mejor_ruta[-1]]

# Coordenadas del marcador de inicio
inicio_coord = distritos_lima[inicio_distrito]
# Desplazar ligeramente las coordenadas del fin
fin_coord = (
    distritos_lima[fin_distrito][0] + 0.001,
    distritos_lima[fin_distrito][1] + 0.001,
)

folium.Marker(
    location=inicio_coord,
    popup=f"Inicio: {inicio_distrito}",
    icon=folium.Icon(color="green", icon="flag"),
).add_to(mapa_lima)

folium.Marker(
    location=fin_coord,  # Usar las coordenadas ajustadas para el fin
    popup=f"Fin: {fin_distrito}",
    icon=folium.Icon(color="red", icon="flag"),
).add_to(mapa_lima)

# Agregar líneas entre las rutas en cada iteración
for resultado in aco.resultados:
    ruta_iteracion = resultado["Ruta"]
    ruta_coords = [distritos_lima[distrito] for distrito in ruta_iteracion] + [
        distritos_lima[ruta_iteracion[0]]
    ]
    # Cambiar el color de las líneas para las rutas generadas en iteraciones
    folium.PolyLine(
        locations=ruta_coords, color="blue", dash_array="5, 5", weight=1
    ).add_to(mapa_lima)

# Agregar la mejor ruta con un color diferente
mejor_ruta_coords = [distritos_lima[distritos_seleccionados[i]] for i in mejor_ruta] + [
    distritos_lima[distritos_seleccionados[mejor_ruta[0]]]
]
folium.PolyLine(locations=mejor_ruta_coords, color="red", weight=3).add_to(
    mapa_lima
)  # Resaltar la mejor ruta

# Guardar el mapa como un archivo HTML
mapa_lima.save("grafica.html")
