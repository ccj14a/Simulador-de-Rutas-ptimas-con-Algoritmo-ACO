# Simulador de Rutas Óptimas con Algoritmo ACO

Simulador diseñado para resolver el Problema del Viajero (TSP) aplicando el Algoritmo de Colonia de Hormigas (ACO), enfocado en optimizar rutas dentro de los distritos de Lima, Perú. Este proyecto permite encontrar las rutas más eficientes en una red urbana utilizando técnicas avanzadas de optimización inspiradas en el comportamiento natural de las colonias de hormigas.

## Archivos Generados

Cada vez que se ejecute el simulador, se generarán los siguientes archivos:

- **Archivo HTML**: Un archivo HTML con una representación visual de las rutas óptimas encontradas por el algoritmo.
- **Archivo Excel**: Un archivo Excel que contiene detalles de las rutas, tiempos de ejecución, y otros parámetros relevantes.

## Stack Tecnológico

En este proyecto utilicé las siguientes tecnologías y herramientas para desarrollar el simulador de rutas:

- Python: Lenguaje de programación principal utilizado para implementar la lógica del simulador y la ejecución del algoritmo ACO.
- NumPy: Biblioteca esencial para el manejo de grandes volúmenes de datos numéricos y cálculos matemáticos rápidos.
- Pandas: Herramienta utilizada para la manipulación y análisis de datos estructurados, como coordenadas y distancias entre puntos.
- Folium.js: Biblioteca para crear mapas interactivos donde se visualizan las rutas óptimas generadas por el algoritmo.
- OSMnx: Utilizada para descargar y analizar la red de calles de Lima a partir de OpenStreetMap, permitiendo la construcción de un grafo de las calles para el algoritmo ACO.

  <img src="resources/python.png" alt="python" width="80"> &nbsp; <img src="resources/numpy.png" alt="numpy" width="80"> &nbsp; <img src="resources/pandas.png" alt="pandas" width="80"> &nbsp; <img src="resources/folium.png" alt="folium" width="80"> &nbsp; <img src="resources/osmnx.jpg" alt="osmnx" width="80">

  ## Preview
![mapa1](https://github.com/user-attachments/assets/e05b9c18-e3ff-4b84-a8d4-6690be91ddd0)
  
![mapa2](https://github.com/user-attachments/assets/0d9caf32-6db0-4f75-b7ac-3ee72950c705)

![mejorRuta](https://github.com/user-attachments/assets/84e2db2b-969b-45b8-adeb-5c8976df3c1b)
