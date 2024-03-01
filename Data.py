import cv2
import os

#Importar la clase
import SeguimientoManos as sm

#Crear la carpeta
nombre = 'letra_A'
direccion = 'C:/Users/Asrock/Desktop/proyectoFinal-ABC/vocales/data' #Cambiar a su dirección local
carpeta = direccion + '/' + nombre

# Si no existe la carpeta, crearla
if not os.path.exists(carpeta):
    print('Carpeta creada', carpeta)
    #creando la carpeta
    os.makedirs(carpeta)
#Lectura de camara
cap = cv2.VideoCapture(0)
#Cambiar la resolución
cap.set(3, 1280)
cap.set(4, 720)

#Declarar contador
cont = 0

#Declarar detector
detector = sm.detectormanos(Confdeteccion=0.9)
while True:

    #Realizar la lectura de la captura
    ret, frame = cap.read()

    # Extrar información de la mano
    frame = detector.encontrarmanos(frame, dibujar=False) #Se puede poner en false para extraer los puntos

    #Posicion de una sola mano
    lista1, bbox, mano = detector.encontrarposicion(frame, ManoNum=0, dibujarPuntos=False, dibujarBox=False, color=[0,255,0])

    #Si hay mano
    if mano == 1:
        # Extrar la información del recuadro que hay alrededor
        xmin, ymin, xmax, ymax = bbox

        #ASIGNAR MARGEN
        xmin = xmin - 40
        ymin = ymin - 40
        xmax = xmax + 40
        ymax = ymax + 40

        #Realizar recorte de nuestra mano
        recorte = frame[ymin:ymax, xmin:xmax]
        cv2.imshow("RECORTE",recorte)

        #Redimensionamiento
        #recorte = cv2.resize(recorte, (640,640), interpolation=cv2.INTER_CUBIC)

        #Almacenar las imagenes
        cv2.imwrite(carpeta + "/A_{}.jpg".format(cont), recorte)
        #Aumentamos contador
        cont = cont + 1


        #cv2.rectangle(frame, (xmin, ymin ), (xmax , ymax ), [0,255,0], 2) #Verificar que en el recuadro la mano quepa correctamente

    #Mostrar los fps
    cv2.imshow("LENGUAJE DE VOCALES", frame)

    #Leer el teclado
    t = cv2.waitKey(1)
    if t == 27 or cont == 100:
        break
cap.release()
cv2.destroyAllWindows()

