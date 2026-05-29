import cv2
import numpy as np
import matplotlib.pyplot as plt

def cuantificar_y_comparar(video_sin_persistencia, video_con_persistencia):
    cap_np = cv2.VideoCapture(video_sin_persistencia)
    cap_p = cv2.VideoCapture(video_con_persistencia)
    
    historial_instantaneo = []  # Para el video normal (sin memoria)
    historial_estela = []       # Para la memoria pura (la resta)
    
    if not cap_np.isOpened() or not cap_p.isOpened():
        print("error: can't open video.")
        return
        
    while True:
        ret_np, frame_np = cap_np.read()
        ret_p, frame_p = cap_p.read()
        
        if not ret_np or not ret_p:
            break
            
        # 1. Escala de grises
        gris_np = cv2.cvtColor(frame_np, cv2.COLOR_BGR2GRAY)
        gris_p = cv2.cvtColor(frame_p, cv2.COLOR_BGR2GRAY)
        
        # 2. Invertir colores (Fondo negro, borde blanco)
        inv_np = 255 - gris_np
        inv_p = 255 - gris_p
        
        # 3. Binarizar
        _, bin_np = cv2.threshold(inv_np, 127, 255, cv2.THRESH_BINARY)
        _, bin_p = cv2.threshold(inv_p, 127, 255, cv2.THRESH_BINARY)
        
        # 4. Extraer métricas
        estela = cv2.subtract(bin_p, bin_np)
        
        energia_np = np.sum(bin_np) / 255.0      # Movimiento en T=0
        energia_estela = np.sum(estela) / 255.0  # Memoria acumulada de T-n
        
        historial_instantaneo.append(energia_np)
        historial_estela.append(energia_estela)
        
    cap_np.release()
    cap_p.release()
    
    # 5. Graficar la comparativa
    plt.figure(figsize=(12, 6))
    
    # Línea de Actividad Instantánea (El estímulo)
    plt.plot(historial_instantaneo, label='No persistence', 
             color='gray', linestyle='--', linewidth=1.5, alpha=0.8)
    
    # Línea de la Memoria (La estela)
    plt.plot(historial_estela, label='Persistence', 
             color='purple', linewidth=2.5)
    plt.fill_between(range(len(historial_estela)), historial_estela, color='purple', alpha=0.2)
    
    #plt.title('Estímulo Instantáneo vs. Memoria Neuromórfica Acumulada', fontsize=14)
    plt.xlabel('Time (Frames)', fontsize=15)
    plt.ylabel('Active Pixels', fontsize=15)
    plt.grid(True, linestyle=':', alpha=0.7)
    plt.legend(loc='upper left', fontsize=13)
    plt.tight_layout()
    
    plt.show()

# --- EJECUCIÓN ---
ruta_video_1 = 'no-perisistencia.mp4'  
ruta_video_2 = 'persistencia.mp4'   

cuantificar_y_comparar(ruta_video_1, ruta_video_2)