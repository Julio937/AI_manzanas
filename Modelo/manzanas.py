import cv2
import numpy as np
import pandas as pd
import os

# Definir la ubicación de tus imágenes de manzanas y pepinos
directorio_raiz = "D:/Universidad/IUE/IA/Modelo/manzanas_ai"

# Lista de nombres de características
nombres_caracteristicas = ["Color_Rojo", "Color_Verde", "Color_Azul", "Textura", "Forma", "Bordes", "Etiqueta"]

# Inicializar listas para características y etiquetas
caracteristicas = []
etiquetas = []

# Crear un DataFrame vacío
df = pd.DataFrame(columns=nombres_caracteristicas)

# Función para calcular características de una imagen
def calcular_caracteristicas(imagen):
    # Redimensionar la imagen a un tamaño manejable
    imagen = cv2.resize(imagen, (1000, 750))  # Ajusta el tamaño según tus necesidades
    
    # Calcular el promedio de intensidad en cada canal de color
    color_promedio = np.mean(imagen, axis=(0, 1))  # Color promedio
    
    # Ejemplo de cálculo de textura 
    textura = np.var(imagen)
    
    # Ejemplo de cálculo de forma 
    forma = imagen.shape[0] / imagen.shape[1]  # Relación de aspecto
    
    # Ejemplo de cálculo de bordes 
    imagen_gris = cv2.cvtColor(imagen, cv2.COLOR_BGR2GRAY)
    bordes = cv2.Canny(imagen_gris, 100, 200)
    
    # Concatenar todas las características en un vector
    caracteristicas_vector = np.concatenate([color_promedio, [textura, forma, bordes.mean()]])
    
    return caracteristicas_vector

# Iterar sobre las carpetas de manzanas y pepinos
for etiqueta in ["manzanas", "pepinos"]:
    directorio_datos = os.path.join(directorio_raiz, etiqueta)
    
    for archivo_imagen in os.listdir(directorio_datos):
        if archivo_imagen.endswith(".JPG"):
            imagen = cv2.imread(os.path.join(directorio_datos, archivo_imagen))
            
            # Calcular características para la imagen actual
            caracteristicas_vector = calcular_caracteristicas(imagen)
            
            # Agregar resultados al DataFrame con nombres de características
            datos_fila = dict(zip(nombres_caracteristicas, caracteristicas_vector))
            datos_fila["Etiqueta"] = etiqueta
            df = pd.concat([df, pd.DataFrame(datos_fila, index=[0])], ignore_index=True)

# Mostrar el DataFrame con las características separadas
print(df)

