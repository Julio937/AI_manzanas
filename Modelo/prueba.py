import cv2
import numpy as np

# Cargar una sola imagen
imagen = cv2.imread("manzanas_ai/manzanas/IMG_7857.JPG")  # Reemplaza "/ruta/a/tu/imagen.jpg" con la ruta a tu imagen

# Verificar si la imagen se cargó correctamente
if imagen is None:
    print("No se pudo cargar la imagen.")
else:
    # Calcular el histograma de color
    hist_color = cv2.calcHist([imagen], [0, 1, 2], None, [256, 256, 256], [0, 256, 0, 256, 0, 256])

    # Imprimir el histograma de color
    print("Histograma de Color:")
    print(hist_color)

    # Puedes acceder a valores específicos del histograma, por ejemplo, el valor en el canal R, G y B en el bin (0, 0, 0)
    valor_canal_R = hist_color[0, 0, 0]
    valor_canal_G = hist_color[0, 0, 1]
    valor_canal_B = hist_color[0, 0, 2]

    print("Valor del canal R en bin (0, 0, 0):", valor_canal_R)
    print("Valor del canal G en bin (0, 0, 0):", valor_canal_G)
    print("Valor del canal B en bin (0, 0, 0):", valor_canal_B)
