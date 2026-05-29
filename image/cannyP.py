import cv2
import os
import glob

INPUT_DIR = "input"
OUTPUT_DIR = "output_canny"

# Parámetros para obtener bordes limpios (tipo 3063_edgeC)
GAUSSIAN_BLUR = (7, 7)  # Kernel de blur - AJUSTA ESTO
low_norm =  0.4     # Umbral alto normalizado
high_norm = 0.8       # Ratio para umbral bajo

os.makedirs(OUTPUT_DIR, exist_ok=True)

for img_path in glob.glob(os.path.join(INPUT_DIR, "*.jpg")):
    img = cv2.imread(img_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # 1. Aplicar Gaussian Blur (CRÍTICO para bordes limpios)
    blurred = cv2.GaussianBlur(gray, GAUSSIAN_BLUR, 0)
    
    # 2. Calcular umbrales
    # Calcular umbrales absolutos
    low_thresh = int(low_norm * 255)
    high_thresh = int(high_norm * 255)
    
    # 3. Canny
    edges = cv2.Canny(blurred, low_thresh, high_thresh)
    
    # 4. Invertir
    binary_edges = cv2.bitwise_not(edges)
    
    # 5. Guardar
    filename = os.path.splitext(os.path.basename(img_path))[0]
    out_path = os.path.join(OUTPUT_DIR, f"{filename}_edgeC.png")
    cv2.imwrite(out_path, binary_edges)
    print(f"✅ {out_path} (blur={GAUSSIAN_BLUR}, thresh=[{low_thresh},{high_thresh}])")

print("\n🛡️ ¡end!")