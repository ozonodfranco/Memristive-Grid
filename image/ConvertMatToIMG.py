import scipy.io as sio
import numpy as np
import cv2
import os
import glob

# ================= CONFIGURACIÓN =================
INPUT_DIR = "ground_truth_mat"
OUTPUT_DIR = "ground_truth_img"

os.makedirs(OUTPUT_DIR, exist_ok=True)

# Buscar todos los archivos .mat
mat_files = glob.glob(os.path.join(INPUT_DIR, "*.mat"))

if not mat_files:
    print(f"❌ No found  .mat files in '{INPUT_DIR}'")
    exit()

print(f"📂 Found {len(mat_files)}  files .mat\n")

# ================= PROCESAMIENTO =================
processed_count = 0
error_count = 0

for mat_path in mat_files:
    filename = os.path.basename(mat_path)
    name_no_ext = os.path.splitext(filename)[0]
    
    try:
        # Cargar archivo .mat
        mat = sio.loadmat(mat_path)
        gt = mat['groundTruth'][0]  # Array de 5 observadores
        
        print(f"🔍 {name_no_ext}: {len(gt)} observers")
        
        # Procesar cada observador humano
        for i, g in enumerate(gt):
            try:
                # Extraer boundaries (tu código que funciona)
                boundaries = g[0][0]['Boundaries']
                
                # Convertir correctamente a array
                boundaries = np.array(boundaries, dtype=np.float32)
                
                # Invertir: fondo blanco (255), bordes negros (0)
                # boundaries: 1 = borde, 0 = fondo
                # (1-boundaries): 0 = borde, 1 = fondo
                # * 255: 0 = borde, 255 = fondo
                img = ((1 - boundaries) * 255).astype(np.uint8)
                
                # Guardar con formato: {nombre}_h{i}.png
                out_path = os.path.join(OUTPUT_DIR, f"{name_no_ext}_h{i}.png")
                cv2.imwrite(out_path, img)
                
                # Estadísticas
                edge_pixels = np.sum(boundaries > 0)
                total_pixels = boundaries.size
                edge_pct = (edge_pixels / total_pixels) * 100
                
                print(f"   ✅ h{i}: {boundaries.shape} | Bordes: {edge_pct:.2f}%")
                processed_count += 1
                
            except Exception as e:
                print(f"   ❌ Error in h{i}: {e}")
                error_count += 1
        
        print()  # Línea en blanco entre archivos
        
    except Exception as e:
        print(f"❌ Error load {filename}: {e}\n")
        error_count += 1

# ================= RESUMEN =================
print("=" * 60)
print(f"📊 RESUME:")
print(f"   Processed files: {len(mat_files)}")
print(f"   image GT created: {processed_count}")
print(f"   Errors: {error_count}")
print(f"   Output folder: '{OUTPUT_DIR}'")
print("=" * 60)

if processed_count > 0:
    print(f"\n🛡️ end")
    print(f"💡 Ejemplo: 3063_h0.png, 3063_h1.png, ..., 3063_h4.png")
else:
    print(f"\n⚠️ Check, errors found!.")