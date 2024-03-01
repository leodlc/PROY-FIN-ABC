import cv2
import os
import time

# Importar la clase
import SeguimientoManos as sm

# Crear la carpeta
nombre = 'letra_A'
direccion = 'C:/Users/Asrock/Desktop/proyectoFinal-ABC/vocales/data'  # Cambiar a su dirección local
carpeta = os.path.join(direccion, nombre)

# Si no existe la carpeta, crearla
if not os.path.exists(carpeta):
    print('Carpeta creada', carpeta)
    # Creando la carpeta
    os.makedirs(carpeta)

# Lectura de cámara
cap = cv2.VideoCapture(0)
# Cambiar la resolución
cap.set(3, 1280)
cap.set(4, 720)

# Declarar contador
cont = 0

# Declarar detector
detector = sm.detectormanos(Confdeteccion=0.9)

while True:
    # Realizar la lectura de la captura
    ret, frame = cap.read()

    # Extraer información de la mano
    frame = detector.encontrarmanos(frame, dibujar=False)  # Se puede poner en False para extraer los puntos

    # Posición de una sola mano
    lista1, bbox, mano = detector.encontrarposicion(frame, ManoNum=0, dibujarPuntos=False, dibujarBox=False,
                                                    color=[0, 255, 0])

    # Si hay mano y aún no se han capturado 100 imágenes
    if mano == 1 and cont < 100:
        # Extraer la información del recuadro que hay alrededor
        xmin, ymin, xmax, ymax = bbox

        # ASIGNAR MARGEN
        xmin = xmin - 40
        ymin = ymin - 40
        xmax = xmax + 40
        ymax = ymax + 40

        # Realizar recorte de nuestra mano
        recorte = frame[ymin:ymax, xmin:xmax]
        cv2.imshow("RECORTE", recorte)


        # Almacenar las imágenes
        cv2.imwrite(os.path.join(carpeta, "A_{}.jpg".format(cont)), recorte)
        # Aumentamos contador
        cont = cont + 1

    # Mostrar los fps
    cv2.imshow("LENGUAJE DE VOCALES", frame)

    # Leer el teclado
    t = cv2.waitKey(1)
    if t == 27 or cont == 100:
        break

cap.release()
cv2.destroyAllWindows()
