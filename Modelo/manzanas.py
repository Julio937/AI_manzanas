import cv2
import numpy as np
import pandas as pd
import os
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.preprocessing import StandardScaler

# Definir la ubicación de tus imágenes
directorio_raiz = "D:/Universidad/IUE/IA/AI_manzanas/Modelo/manzanas_ai"

# Lista de nombres de características
nombres_caracteristicas = ["Color_Rojo", "Color_Verde", "Color_Azul", "Textura", "Forma", "Bordes", "Promedio_Grises", "Contornos", "Etiqueta"]

# Inicializar listas para características y etiquetas
caracteristicas = []
etiquetas = []

# Crear un DataFrame vacío
df = pd.DataFrame(columns=nombres_caracteristicas)

# Función para calcular características de una imagen
def calcular_caracteristicas(imagen):
    # Redimensionar la imagen a un tamaño manejable
    imagen = cv2.resize(imagen, (1000, 750))  
    
    # Calcular el promedio de intensidad en cada canal de color
    color_promedio = np.mean(imagen, axis=(0, 1))  # Color promedio
    
    # Cálculo de textura 
    textura = np.var(imagen)
    
    # Cálculo de forma 
    forma = imagen.shape[0] / imagen.shape[1]  # Relación de aspecto
    
    # Cálculo de bordes 
    imagen_gris = cv2.cvtColor(imagen, cv2.COLOR_BGR2GRAY)
    bordes = cv2.Canny(imagen_gris, 100, 200)
    
    # Cálculo del promedio de intensidad en escala de grises
    promedio_grises = np.mean(imagen_gris)
    
    # Cálculo de contornos
    contornos, _ = cv2.findContours(bordes, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    numero_contornos = len(contornos)
    
    # Concatenar todas las características en un vector
    caracteristicas_vector = np.concatenate([color_promedio, [textura, forma, bordes.mean(), promedio_grises, numero_contornos]])
    
    return caracteristicas_vector

# Iterar sobre las carpetas de manzanas y pepinos
for etiqueta in ["manzanas_rojas", "pepinos", "manzanas_verdes", "manzanas_rojas_verdes"]:
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

# Mostrar la cantidad de etiquetas de cada tipo
print("Cantidad de fotos de cada tipo:")
print(df["Etiqueta"].value_counts())

# Dividir el DataFrame en conjuntos de entrenamiento y pruebas
X_train, X_test, y_train, y_test = train_test_split(df.drop('Etiqueta', axis=1), df['Etiqueta'], test_size=0.3, random_state=42)

# Escalar las características
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Crear el modelo de clasificación
modelo = LogisticRegression(max_iter=10000)

# Entrenar el modelo
modelo.fit(X_train, y_train)

# Realizar predicciones en el conjunto de pruebas
y_pred = modelo.predict(X_test)

# Calcular la precisión del modelo
precisión = accuracy_score(y_test, y_pred)
print("Precisión del modelo:", precisión)

# Mostrar la matriz de confusión
matriz_confusión = confusion_matrix(y_test, y_pred)
print("Matriz de confusión:\n", matriz_confusión)

# Mostrar informe de clasificación
informe = classification_report(y_test, y_pred)
print("Informe de clasificación:\n", informe)

