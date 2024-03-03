import tkinter as tk
from PIL import ImageTk, Image
import cv2
from ultralytics import YOLO
import SeguimientoManos as sm

class VentanaPrincipal(tk.Tk):
    def __init__(self):
        super().__init__()
        self.geometry('600x400')
        self.title('Reconocimiento de Señas')
        self.iconbitmap('media/icono.ico')  # Cambia la ruta al icono de tu aplicación

        # Cargamos la imagen con Pillow
        imagen_fondo = Image.open("media/fondo2.jpg")  # Cambia la ruta a tu imagen de fondo
        imagen_fondo = imagen_fondo.resize((600, 400), Image.BILINEAR)

        self.imagen_fondo_tk = ImageTk.PhotoImage(imagen_fondo)

        # Logo en ventana principal
        imagen_logo = Image.open("media/logo.jpG").convert('RGBA')  # Cambia la ruta a tu logo
        # Redimensionar la imagen a un tamaño menor
        imagen_logo = imagen_logo.resize((200, 200), Image.BILINEAR)
        # Crear un objeto PhotoImage a partir de la imagen
        self.imagen_logo_tk = ImageTk.PhotoImage(imagen_logo)

        # Creamos un label con la imagen y lo agregamos a la ventana
        label_fondo = tk.Label(self, image=self.imagen_fondo_tk)
        label_fondo.place(x=0, y=0, relwidth=1, relheight=1)
        label_logo = tk.Label(self, image=self.imagen_logo_tk)
        label_logo.grid(row=1, column=2)

        # Crear una cuadrícula 5x5
        for fila in range(0, 5):
            self.rowconfigure(fila, weight=1)
            for columna in range(0, 5):
                self.columnconfigure(columna, weight=5)

        self.boton_iniciar_deteccion = tk.Button(self, text='Iniciar Detección', command=self.iniciar_deteccion, bg='#41A7BE',
                                        fg='black', font=('DynaPuff', 12))
        self.boton_iniciar_deteccion.grid(row=2, column=2)

        self.boton_salir = tk.Button(self, text='Salir', command=self.salir,
                                bg='#41A7BE', fg='black', font=('DynaPuff', 12))
        self.boton_salir.grid(row=3, column=2)

    def iniciar_deteccion(self):
        # Inicializar aquí la detección de señas
        cap = cv2.VideoCapture(0)
        cap.set(3, 1280)
        cap.set(4, 720)

        model = YOLO('best3.pt')
        detector = sm.detectormanos(Confdeteccion=0.9)

        while True:
            ret, frame = cap.read()

            frame = detector.encontrarmanos(frame, dibujar=False)

            lista1, bbox, mano = detector.encontrarposicion(frame, ManoNum=0, dibujarPuntos=False, dibujarBox=False,
                                                            color=[0, 255, 0])

            if mano == 1:
                xmin, ymin, xmax, ymax = bbox
                xmin = xmin - 40
                ymin = ymin - 40
                xmax = xmax + 40
                ymax = ymax + 40

                recorte = frame[ymin:ymax, xmin:xmax]

                recorte = cv2.resize(recorte, (640, 640), interpolation=cv2.INTER_CUBIC)
                # Mostrar mensaje en la ventana de recorte
                cv2.putText(recorte, "Presione ESC para salir", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)

                resultados = model.predict(recorte, conf=0.55)

                if len(resultados) != 0:
                    for results in resultados:
                        masks = results.masks
                        coordenadas = masks

                        # Se define la variable 'anotaciones' fuera del bloque 'if'
                        anotaciones = resultados[0].plot()
                cv2.imshow("RECORTE", anotaciones)

            cv2.imshow("LENGUAJE DE VOCALES", frame)



            t = cv2.waitKey(1)
            if t == 27:
                break

        cap.release()
        cv2.destroyAllWindows()

    def salir(self):
        self.quit()
        self.destroy()


if __name__ == '__main__':
    VentanaPrincipal().mainloop()
