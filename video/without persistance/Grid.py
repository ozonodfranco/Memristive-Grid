import numpy as np
import cv2
import os
import time
import glob
import shutil
from numba import njit, prange

# ==============================================================================
# HERRAMIENTAS DE VIDEO
# ==============================================================================
def VideoToImagen(video_path, output_folder):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    cap = cv2.VideoCapture(video_path)
    frame_count = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        frame_filename = os.path.join(output_folder, f'frame_{frame_count:04d}.png')
        cv2.imwrite(frame_filename, gray_frame)
        frame_count += 1
    cap.release()
    print(f"[INFO] extract {frame_count} photograms.")

def contruirVideo(input_folder, output_video):
    image_files = sorted([img for img in os.listdir(input_folder) if img.endswith(".png")])
    if not image_files: return
    frame_example = cv2.imread(os.path.join(input_folder, image_files[0]))
    height, width = frame_example.shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_video, fourcc, 30, (width, height), isColor=False)
    for image_file in image_files:
        img_path = os.path.join(input_folder, image_file)
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        out.write(img)
    out.release()
    print(f"[INFO] output video save: {output_video}")

# ==============================================================================
# SECCIÓN 1: NÚCLEO MATEMÁTICO (JIT) - FLUJO (Orden 1, k=1)
# ==============================================================================
@njit(fastmath=True)
def calculate_memristance_flux(phi, eta, X0, Ron, Roff, kappa):
    """
    Calcula la memristancia basada en el flujo. 
    Nota: Se omitió el inverso global de la fórmula original de Maple 
    para retornar Memristancia (Ohms) en lugar de Conductancia (Siemens).
    """
    exp_p = np.exp(4.0 * kappa * phi)
    exp_m = np.exp(-4.0 * kappa * phi)

    if eta == -1:
        if phi <= 0.0:
            term = (1.0 - X0) * exp_p - ((1.0 - X0)**2) * (exp_p - 1.0) * exp_p
            val = (1.0 / Roff) + (1.0 / Ron - 1.0 / Roff) * term
        else:
            term = X0 * exp_m - (X0**2) * (exp_m - 1.0) * exp_m
            val = (1.0 / Ron) + (1.0 / Roff - 1.0 / Ron) * term
    else: # eta == 1
        if phi <= 0.0:
            term = X0 * exp_p - (X0**2) * (exp_p - 1.0) * exp_p
            val = (1.0 / Ron) + (1.0 / Roff - 1.0 / Ron) * term
        else:
            term = (1.0 - X0) * exp_m - ((1.0 - X0)**2) * (exp_m - 1.0) * exp_m
            val = (1.0 / Roff) + (1.0 / Ron - 1.0 / Roff) * term

    # Retornamos M(phi) = Ron * Roff * val
    return Ron * Roff * val
#simulation_step(u_curr, u_next, Mp, Mq, phi_p, phi_q, Mp1, Mp2, Mq1, Mq2, Pimagnoise, Rin, Roff, Ron, kappa, dt, tn, X0, phi_p2,phi_q2)
@njit(parallel=True, fastmath=True)
def simulation_step(u_curr, u_next, Mp, Mq, phi_p, phi_q, Mp1, Mp2, Mq1, Mq2, 
                    Pimagnoise, Rin, Roff, Ron, kappa, dt,tn, X0,phi_p2,phi_q2):
    
    M, N = u_curr.shape
    inv_Rin = 1.0 / Rin
    
    # --- PASO 1: CALCULAR NUEVO VOLTAJE ---
    for i in prange(M):
        for j in range(N):
            numerator = Pimagnoise[i, j] * inv_Rin
            denominator = inv_Rin
            
            if i > 0:
                g = 1.0 / Mp[i-1, j]
                numerator += u_curr[i-1, j] * g
                denominator += g
            if i < M - 1:
                g = 1.0 / Mp[i, j]
                numerator += u_curr[i+1, j] * g
                denominator += g
            if j > 0:
                g = 1.0 / Mq[i, j-1]
                numerator += u_curr[i, j-1] * g
                denominator += g
            if j < N - 1:
                g = 1.0 / Mq[i, j]
                numerator += u_curr[i, j+1] * g
                denominator += g
                
            u_next[i, j] = numerator / denominator

    # --- PASO 2: ACTUALIZAR MEMRISTORES (BACKWARD EULER) ---
    for i in prange(M):
        for j in range(N):
            
            # --- Vertical (Mp) ---
            # Topología Qucs-S: N_Arriba(N-) --- [FLUX] ---(N+) Centro (N+)--- [INV] --- (N-)N_Abajo
            if i < M - 1:
                v_drop = u_next[i+1, j] - u_next[i, j]
                
                sum_M = Mp1[i, j] + Mp2[i, j]
                v1_new = v_drop * (Mp1[i, j] / sum_M)
                v2_new = v_drop - v1_new
                
                # CORREGIDO: M1 ve voltaje negativo (N+ en el centro, corriente entra por N-)
                # M2 ve voltaje positivo (corriente sale hacia N-)
                phi_p[i, j] += dt * v1_new
                phi_p2[i, j] += dt * v2_new
                
                # CORREGIDO: Qucs usa FLUX (1) primero y INV (-1) después
                Mp1[i, j] = calculate_memristance_flux(phi_p[i, j], -1, X0, Ron, Roff, kappa)
                Mp2[i, j] = calculate_memristance_flux(phi_p2[i, j], 1, X0, Ron, Roff, kappa)
                Mp[i, j] = Mp1[i, j] + Mp2[i, j]

            # --- Horizontal (Mq) ---
            # Topología Qucs-S: N_Izq(N-) --- [INV] ---(N+) Centro (N+)--- [FLUX] --- (N-)N_Der
            if j < N - 1:
                v_drop = u_next[i, j+1] - u_next[i, j]
                
                sum_M = Mq1[i, j] + Mq2[i, j]
                v1_new = v_drop * (Mq1[i, j] / sum_M)
                v2_new = v_drop - v1_new
                
                # CORREGIDO: M1 ve voltaje negativo, M2 ve voltaje positivo
                phi_q[i, j] += dt * v1_new
                phi_q2[i, j] += dt * v2_new
                
                # CORREGIDO: Qucs usa INV (-1) primero y FLUX (1) después en horizontales
                Mq1[i, j] = calculate_memristance_flux(phi_q[i, j], -1, X0, Ron, Roff, kappa)
                Mq2[i, j] = calculate_memristance_flux(phi_q2[i, j], 1, X0, Ron, Roff, kappa)
                Mq[i, j] = Mq1[i, j] + Mq2[i, j]

    # --- PASO 3: PREPARAR SIGUIENTE CICLO ---
    u_curr[:] = u_next[:]





# ==============================================================================
# SECCIÓN 2: PROCESAMIENTO ESTÁTICO DE FOTOGRAMAS (SIN MEMORIA)
# ==============================================================================
def process_memristive_frame_static(image_path, output_dir, params):
    Rinit, Roff, Ron, Rin, kappa, X0, dt, tn, Mum = params

    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None: return

    # Inyectamos 3V para máxima velocidad
    Pimagnoise = cv2.normalize(img.astype('float64'), None, 0.0, 3.0, cv2.NORM_MINMAX)
    M, N = Pimagnoise.shape

    # ¡LA CLAVE DEL ENFOQUE NATURAL!
    # Las matrices nacen con valores cero (o Rinit) puras y frescas en cada fotograma.
    u_curr = np.zeros((M, N), dtype=np.float64)
    u_next = np.empty_like(u_curr)
    Mp1 = np.full((M-1, N), Rinit, dtype=np.float64)
    Mp2 = np.full((M-1, N), Rinit, dtype=np.float64)
    Mp = Mp1 + Mp2
    phi_p = np.zeros((M-1, N), dtype=np.float64) 
    phi_p2 = np.zeros((M-1, N), dtype=np.float64)
    Mq1 = np.full((M, N-1), Rinit, dtype=np.float64)
    Mq2 = np.full((M, N-1), Rinit, dtype=np.float64)
    Mq = Mq1 + Mq2
    phi_q = np.zeros((M, N-1), dtype=np.float64)
    phi_q2 = np.zeros((M, N-1), dtype=np.float64)

    start_time = time.time()
    
    for ts in range(tn):
        # Corremos la simulación completa para este frame independiente
        simulation_step(u_curr, u_next, Mp, Mq, phi_p, phi_q, Mp1, Mp2, Mq1, Mq2, Pimagnoise, Rin, Roff, Ron, kappa, dt, tn, X0, phi_p2,phi_q2)
    
    print(f"[{os.path.basename(image_path)}] Time processing: {time.time() - start_time:.4f}s")

    # Extracción de Bordes
    edge_map = np.ones((M, N), dtype=np.uint8) * 255
    mask_p = Mp >= Mum
    mask_q = Mq >= Mum
    Mp_pad = np.pad(mask_p, ((0, 1), (0, 0)), mode='constant')
    Mq_pad = np.pad(mask_q, ((0, 0), (0, 1)), mode='constant')
    edge_map[Mp_pad | Mq_pad] = 0

    save_path = os.path.join(output_dir, f"Proc_{os.path.basename(image_path)}")
    cv2.imwrite(save_path, edge_map)

# ==============================================================================
# EJECUCIÓN PRINCIPAL LOCAL
# ==============================================================================
if __name__ == "__main__":
    
    input_videos_folder = "input_videos"
    output_videos_folder = "output_videos"
    workspace_folder = "workspace_memristivo"

    os.makedirs(input_videos_folder, exist_ok=True)
    os.makedirs(output_videos_folder, exist_ok=True)
    os.makedirs(workspace_folder, exist_ok=True)

    formatos = ('*.mp4', '*.mov', '*.avi', '*.mkv')
    videos_encontrados = []
    for ext in formatos:
        #videos_encontrados.extend(glob.glob(os.path.join(input_videos_folder, ext)))
        videos_encontrados.extend(glob.glob(os.path.join(input_videos_folder, ext.upper())))

    if not videos_encontrados:
        print(f"¡Atention! no videos found.")
        print(f"please, only videos (.mp4, .avi, etc.) in the folder '{input_videos_folder}' and run again.")
        exit()

    print(f"[INFO] ¡found {len(videos_encontrados)} videos!")

    # PARÁMETROS GLOBALES DE LA RED
    Rinit = 1.1; alpha_param = 500; Roff = 2 * alpha_param * Rinit
    Ron = 1.0; beta_param = 22; Rin = Roff / beta_param
    mu = 1e-14; Delta = 10e-9
    X0 = (Roff - Rinit) / (Roff - Ron)
    kappa = (mu * Ron) / (Delta**2 * (Ron * X0 + Roff * (1.0 - X0)))
    
    # ¡TIEMPO DE INTEGRACIÓN ULTRA CORTO! (La magia del k=5 + 3V sin memoria)
    tfin = 0.018 # Solo toma 15 iteraciones llegar al borde
    dt = 0.0001
    tn = int(tfin / dt)
    
    porcMoff = 2.0
    Mum = (porcMoff / 100.0) * Roff
    params = (Rinit, Roff, Ron, Rin, kappa, X0, dt, tn, Mum)

    for local_video_in in videos_encontrados:
        nombre_base = os.path.splitext(os.path.basename(local_video_in))[0]
        local_video_out = os.path.join(output_videos_folder, f"{nombre_base}_reconstruido.mp4")

        print(f"\n=======================================================")
        print(f"[INFO] START PROCESSING: {nombre_base}")
        print(f"=======================================================")

        local_input_folder = os.path.join(workspace_folder, "separado")
        local_output_folder = os.path.join(workspace_folder, "Procesado")

        if os.path.exists(local_input_folder): shutil.rmtree(local_input_folder)
        if os.path.exists(local_output_folder): shutil.rmtree(local_output_folder)
        os.makedirs(local_input_folder, exist_ok=True)
        os.makedirs(local_output_folder, exist_ok=True)

        print("\n--- PHASE 1: EXTRACTION ---")
        VideoToImagen(local_video_in, local_input_folder)

        print("\n--- PHASE 2: MATH PROCESSING  ---")
        files = sorted(glob.glob(os.path.join(local_input_folder, '*.png')))
        
        # Procesamos cada frame mandándolo limpio desde cero
        for f in files:
            process_memristive_frame_static(f, local_output_folder, params)

        print("\n--- PHASE 3: reconstruction ---")
        contruirVideo(local_output_folder, local_video_out)
        print(f"video save: {local_video_out}")

    print("\n[INFO] Processing END. ")