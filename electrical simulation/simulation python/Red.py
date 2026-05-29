import numpy as np
import cv2
import os
import time
from numba import njit, prange
import glob

# ==============================================================================
# Autor: Juan Manuel Ugalde Franco
# ==============================================================================

# ==============================================================================
# SECCIÓN 1: NÚCLEO MATEMÁTICO (JIT) - MODELO CONTROLADO POR FLUJO
# ==============================================================================


@njit(fastmath=True)
def calculate_memristance_flux(phi, eta, X0, Ron, Roff, kappa):
    exp_p = np.exp(4.0 * kappa * phi)
    exp_m = np.exp(-4.0 * kappa * phi)

    if eta == -1:
        if phi <= 0.0:
            term = (1.0 - X0) * exp_p - ((1.0 - X0)**2) * (exp_p - 1.0) * exp_p
            # ¡Base correcta para eta=-1 (Arranca en Rinit = 1.1 Ohms)!
            val = (1.0 / Roff) + (1.0 / Ron - 1.0 / Roff) * term
        else:
            term = X0 * exp_m - (X0**2) * (exp_m - 1.0) * exp_m
            val = (1.0 / Ron) + (1.0 / Roff - 1.0 / Ron) * term
    else: # eta == 1
        if phi <= 0.0:
            term = X0 * exp_p - (X0**2) * (exp_p - 1.0) * exp_p
            # ¡Base correcta para eta=1 (Arranca en Rinit = 1.1 Ohms)!
            val = (1.0 / Ron) + (1.0 / Roff - 1.0 / Ron) * term
        else:
            term = (1.0 - X0) * exp_m - ((1.0 - X0)**2) * (exp_m - 1.0) * exp_m
            val = (1.0 / Roff) + (1.0 / Ron - 1.0 / Roff) * term

    return Ron * Roff * val

@njit(parallel=True, fastmath=True)
def simulation_step(u_curr, u_next, Mp, Mq, phi_p, phi_q, Mp1, Mp2, Mq1, Mq2, 
                    Pimagnoise, Rin, Roff, Ron, kappa, dt, X0, phi_p2, phi_q2):
    
    M, N = u_curr.shape
    inv_Rin = 1.0 / Rin

    # --- PASO 1: CALCULAR NUEVO VOLTAJE ---
    for _ in range(200): # <-- ¡ESTE BUCLE ES VITAL!
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
                
        # Sincronización dentro del solver
        for i in prange(M):
            for j in range(N):
                u_curr[i, j] = u_next[i, j]


    # --- PASO 2: ACTUALIZAR MEMRISTORES (BACKWARD EULER) ---
    for i in prange(M):
        for j in range(N):
            
            # --- Vertical (Mp) ---
            # Topología Qucs-S: N_Arriba(N-) --- [FLUX] ---(N+) Centro (N+)--- [INV] --- (N-)N_Abajo
            if i < M - 1:
                v_drop_new = u_next[i, j] - u_next[i+1, j] 
                
                sum_M = Mp1[i, j] + Mp2[i, j]
                v1_new = v_drop_new * (Mp1[i, j] / sum_M)
                v2_new = v_drop_new - v1_new
                
                # CORREGIDO: M1 ve voltaje negativo (N+ en el centro, corriente entra por N-)
                # M2 ve voltaje positivo (corriente sale hacia N-)
                phi_p[i, j] -= dt * v1_new
                phi_p2[i, j] += dt * v2_new
                
                # CORREGIDO: Qucs usa FLUX (1) primero y INV (-1) después
                Mp1[i, j] = calculate_memristance_flux(phi_p[i, j], -1, X0, Ron, Roff, kappa)
                Mp2[i, j] = calculate_memristance_flux(phi_p2[i, j], 1, X0, Ron, Roff, kappa)
                Mp[i, j] = Mp1[i, j] + Mp2[i, j]

            # --- Horizontal (Mq) ---
            # Topología Qucs-S: N_Izq(N-) --- [INV] ---(N+) Centro (N+)--- [FLUX] --- (N-)N_Der
            if j < N - 1:
                v_drop_new = u_next[i, j] - u_next[i, j+1]
                
                sum_M = Mq1[i, j] + Mq2[i, j]
                v1_new = v_drop_new * (Mq1[i, j] / sum_M)
                v2_new = v_drop_new - v1_new
                
                # CORREGIDO: M1 ve voltaje negativo, M2 ve voltaje positivo
                phi_q[i, j] -= dt * v1_new
                phi_q2[i, j] += dt * v2_new
                
                # CORREGIDO: Qucs usa INV (-1) primero y FLUX (1) después en horizontales
                Mq1[i, j] = calculate_memristance_flux(phi_q[i, j], -1, X0, Ron, Roff, kappa)
                Mq2[i, j] = calculate_memristance_flux(phi_q2[i, j], 1, X0, Ron, Roff, kappa)
                Mq[i, j] = Mq1[i, j] + Mq2[i, j]

    # --- PASO 3: PREPARAR SIGUIENTE CICLO ---
    u_curr[:] = u_next[:]


# ==============================================================================
# SECCIÓN 1.5: EXPORTACIÓN DE DATOS TIPO QUCS
# ==============================================================================
def export_to_ngspice_format(time_history, voltage_history, Rp_history,Rq_history, filepath):
    """
    Exporta el historial de voltajes de la matriz en formato .dat.ngspice
    compatible con Qucs-S y el script de ploteo plot2E.py.
    """
    tn = len(time_history)
    M, N = voltage_history[0].shape
    
    print(f"   💾 Write simulation file: {os.path.basename(filepath)}...")
    with open(filepath, 'w', encoding='utf-8') as f:
        # Encabezado estándar
        f.write("<Python Dataset  1.0>\n")
        
        # Bloque de tiempo independiente
        f.write(f"<indep time {tn}>\n")
        for t in time_history:
            f.write(f"{t:.12e}\n")
        f.write("</indep>\n")
        
        # Bloques de variables dependientes (Voltajes nodales)
        for i in range(M):
            for j in range(N):
                node_name = f"v{i}{j}"
                f.write(f"<dep tran.v({node_name}) time>\n")
                for t_idx in range(tn):
                    f.write(f"{voltage_history[t_idx][i, j]:.12e}\n")
                f.write("</dep>\n")
                
        print(f"   ✅ Export: {M * N} nodes.")
    
        print(f"   💾 Resistances history: {os.path.basename(filepath)}...")
        
        # 1. Escribir todos los Fusibles Verticales (Mp)
        for i in range(M-1):
            for j in range(N):
                node_name = f"Mp{i}{j}"
                f.write(f"<dep tran.R({node_name}) time>\n")
                for t_idx in range(tn):
                    f.write(f"{Rp_history[t_idx][i, j]:.12e}\n")
                f.write("</dep>\n")
                
        # 2. Escribir todos los Fusibles Horizontales (Mq)
        for i in range(M):
            for j in range(N-1):
                node_name = f"Mq{i}{j}"
                f.write(f"<dep tran.R({node_name}) time>\n")
                for t_idx in range(tn):
                    f.write(f"{Rq_history[t_idx][i, j]:.12e}\n")
                f.write("</dep>\n")
                
    print(f"   ✅ Export {(M-1 * N) + (M * N-1)} fuses complete.")

# ==============================================================================
# SECCIÓN 2: PROCESAMIENTO Y CONTROL
# ==============================================================================

def process_memristive_grid(image_path, output_dir):
    # Parámetros base
    Rinit = 1.1; alpha_param = 500; Roff = 2 * alpha_param * Rinit
    Ron = 1.0; beta_param = 22; Rin = Roff / beta_param
    mu = 1e-14; Delta = 10e-9
    
    # Nuevos parámetros para el modelo de flujo
    X0 = (Roff - Rinit) / (Roff - Ron)
    kappa = (mu * Ron) / (Delta**2 * (Ron * X0 + Roff * (1.0 - X0)))
    
    tfin = 0.018 
    #dt = 0.000001
    dt = 100e-6
    tn = int(tfin / dt)
    
    img = cv2.imread(image_path)
    if img is None: return

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    Pimagnoise = cv2.normalize(gray.astype('float64'), None, 0.0, 1.0, cv2.NORM_MINMAX)
    
    print(Pimagnoise)
    
    M, N = Pimagnoise.shape
    
    # Inicialización de matrices
    u_curr = np.zeros((M, N), dtype=np.float64)
    u_next = np.empty_like(u_curr)
    
    Mp1 = np.full((M-1, N), Rinit, dtype=np.float64); Mp2 = np.full((M-1, N), Rinit, dtype=np.float64)
    Mp = Mp1 + Mp2; phi_p = np.zeros((M-1, N), dtype=np.float64) 
    
    phi_p2 = np.zeros((M-1, N), dtype=np.float64)
    
    Mq1 = np.full((M, N-1), Rinit, dtype=np.float64); Mq2 = np.full((M, N-1), Rinit, dtype=np.float64)
    Mq = Mq1 + Mq2; phi_q = np.zeros((M, N-1), dtype=np.float64)
    
    phi_q2 = np.zeros((M, N-1), dtype=np.float64)
    
    base_name = os.path.basename(image_path).split('.')[0]
    history_dir = os.path.join(output_dir, f"historial_{base_name}")
    os.makedirs(history_dir, exist_ok=True)
    
    porcMoff = 2.0
    Mum = (porcMoff / 100.0) * Roff

    # Listas para almacenar el historial de voltajes y tiempos
    time_history = []
    voltage_history = []
    Rp_history = []
    Rq_history = []

    print(f"Start flux simulation... {os.path.basename(image_path)}")
    start_time = time.time()
    
    for ts in range(tn):
        # Guardar estado actual (antes del paso de simulación)
        current_time = ts * dt
        time_history.append(current_time)
        voltage_history.append(u_curr.copy())
        Rp_history.append(Mp.copy())
        Rq_history.append(Mq.copy())
        
        simulation_step(u_curr, u_next, Mp, Mq, phi_p, phi_q, Mp1, Mp2, Mq1, Mq2,
                        Pimagnoise, Rin, Roff, Ron, kappa, dt, X0,phi_p2,phi_q2)
        
        # Extracción de bordes para la iteración actual
        edge_map = np.ones((M, N), dtype=np.uint8) * 255
        
        mask_p = Mp >= Mum
        mask_q = Mq >= Mum
        
        Mp_pad = np.pad(mask_p, ((1, 1), (0, 0)), mode='constant')
        Mq_pad = np.pad(mask_q, ((0, 0), (1, 1)), mode='constant')
        
        is_edge = (Mp_pad[:-1, :] | Mp_pad[1:, :] | Mq_pad[:, :-1] | Mq_pad[:, 1:])
        edge_map[is_edge] = 0
        
        iter_path = os.path.join(history_dir, f"iter_{ts:04d}.jpg")
        cv2.imwrite(iter_path, edge_map)

    print(f"Simulation time: {time.time() - start_time:.4f}s")
    
    # Guardar el dataset
    dat_filepath = os.path.join(output_dir, f"Nodes_{base_name}.dat.ngspice")
    export_to_ngspice_format(time_history, voltage_history,Rp_history,Rq_history, dat_filepath)

    print(f"History save in: {history_dir}")
    final_path = os.path.join(output_dir, f"Final_Flux_{os.path.basename(image_path)}")
    cv2.imwrite(final_path, edge_map)

if __name__ == "__main__":
    input_folder = "input"  
    output_folder = "output"
    
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
        
    extensions = ['*.jpg', '*.png']
    files = []
    for ext in extensions:
        files.extend(glob.glob(os.path.join(input_folder, ext)))
        
    if not files:
        print("No se encontraron imágenes. Creando imagen dummy para prueba (10x10 para no saturar los nodos)...")
        # Cambiado a 10x10 para que los archivos .dat no sean inmensos durante pruebas
        dummy = np.random.randint(0, 255, (10, 10), dtype=np.uint8) 
        cv2.imwrite(os.path.join(input_folder, "dummy_test.jpg") if os.path.exists(input_folder) else "dummy_test.jpg", dummy)
        files = ["dummy_test.jpg" if not os.path.exists(input_folder) else os.path.join(input_folder, "dummy_test.jpg")]

    for f in files:
        process_memristive_grid(f, output_folder)