import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import distance_transform_edt
from glob import glob

# ================= CONFIGURATION =================
DIR_MEMRISTOR = 'output'
DIR_CANNY = 'output_canny'
DIR_GT = 'ground_truth_img'
MAX_DIST = 2.0  # Spatial tolerance (standard for BSDS benchmark)
NUM_HUMANS = 3  # Using _h0, _h1, _h2

# ================= CORE FUNCTIONS =================
def load_edge_image(path):
    """Loads image, assumes white background and black edges. Returns binary matrix (1=edge, 0=bg)."""
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if img is None: return None
    return (img < 128).astype(np.uint8)

def evaluate_single_image(pred_edges, gt_edges_list, max_dist):
    """Calculates True Positives, False Positives, and False Negatives using spatial tolerance."""
    if pred_edges is None or pred_edges.sum() == 0:
        return 0, pred_edges.sum() if pred_edges is not None else 0, sum([gt.sum() for gt in gt_edges_list])/len(gt_edges_list)

    # Union of all humans for Precision calculation
    gt_union = np.logical_or.reduce(gt_edges_list)
    gt_dist_map = distance_transform_edt(1 - gt_union)
    
    match_pred = (gt_dist_map <= max_dist) & pred_edges
    tp_precision = match_pred.sum()
    fp = pred_edges.sum() - tp_precision
    
    # Recall calculated per human, then averaged
    pred_dist_map = distance_transform_edt(1 - pred_edges)
    recalls = []
    tp_recall_sum = 0
    fn_sum = 0
    
    for gt in gt_edges_list:
        if gt.sum() == 0: continue
        match_gt = (pred_dist_map <= max_dist) & gt
        tp_recall_sum += match_gt.sum()
        fn_sum += (gt.sum() - match_gt.sum())
        
    return tp_precision, fp, fn_sum / len(gt_edges_list)

def get_metrics(tp, fp, fn):
    p = tp / (tp + fp) if (tp + fp) > 0 else 0
    r = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * (p * r) / (p + r) if (p + r) > 0 else 0
    return p, r, f1

# ================= MAIN PIPELINE =================
# Stores data for ODS (Global sums)
mem_global = {'tp': 0, 'fp': 0, 'fn': 0}
can_global = {'tp': 0, 'fp': 0, 'fn': 0}

# Stores data for OIS (Per-image metrics)
mem_img_f1, can_img_f1 = [], []
mem_scatter_P, mem_scatter_R = [], []
can_scatter_P, can_scatter_R = [], []

gt_files = glob(os.path.join(DIR_GT, '*_h0.png'))
image_numbers = [os.path.basename(f).replace('_h0.png', '') for f in gt_files]

print(f"Evaluating {len(image_numbers)} images against {NUM_HUMANS} human annotators...")

for num in image_numbers:
    # Load up to NUM_HUMANS ground truths
    gts = []
    for i in range(NUM_HUMANS):
        gt_path = os.path.join(DIR_GT, f"{num}_h{i}.png")
        gt = load_edge_image(gt_path)
        if gt is not None:
            gts.append(gt)
            
    if not gts: continue
        
    # Memristor Evaluation
    mem_edges = load_edge_image(os.path.join(DIR_MEMRISTOR, f"Final_Flux_{num}.jpg"))
    if mem_edges is not None:
        tp_p, fp, fn = evaluate_single_image(mem_edges, gts, MAX_DIST)
        mem_global['tp'] += tp_p; mem_global['fp'] += fp; mem_global['fn'] += fn
        p, r, f1 = get_metrics(tp_p, fp, fn)
        mem_img_f1.append(f1)
        mem_scatter_P.append(p); mem_scatter_R.append(r)
        
    # Canny Evaluation
    can_edges = load_edge_image(os.path.join(DIR_CANNY, f"{num}_edgeC.png"))
    if can_edges is not None:
        tp_p, fp, fn = evaluate_single_image(can_edges, gts, MAX_DIST)
        can_global['tp'] += tp_p; can_global['fp'] += fp; can_global['fn'] += fn
        p, r, f1 = get_metrics(tp_p, fp, fn)
        can_img_f1.append(f1)
        can_scatter_P.append(p); can_scatter_R.append(r)

# ================= CALCULATE ODS, OIS, AP =================
# ODS: Metric calculated over the entire dataset accumulation
mem_ODS_P, mem_ODS_R, mem_ODS_F1 = get_metrics(mem_global['tp'], mem_global['fp'], mem_global['fn'])
can_ODS_P, can_ODS_R, can_ODS_F1 = get_metrics(can_global['tp'], can_global['fp'], can_global['fn'])

# OIS: Average of the maximum F1 per image (since it's binary, it's just the mean F1)
mem_OIS_F1 = np.mean(mem_img_f1) if mem_img_f1 else 0
can_OIS_F1 = np.mean(can_img_f1) if can_img_f1 else 0

# AP: Average Precision (For binary outputs, it equals the ODS Precision)
mem_AP = mem_ODS_P
can_AP = can_ODS_P

print("\n--- BENCHMARK RESULTS ---")
print(f"Method       |  ODS (F1)  |  OIS (F1)  |    AP      |")
print(f"-----------------------------------------------------")
print(f"Memristive   |   {mem_ODS_F1:.4f}   |   {mem_OIS_F1:.4f}   |   {mem_AP:.4f}   |")
print(f"Canny OpenCV |   {can_ODS_F1:.4f}   |   {can_OIS_F1:.4f}   |   {can_AP:.4f}   |")

# ================= PLOTTING THE PR BENCHMARK GRAPH =================
plt.figure(figsize=(12, 10))

# Background Iso-F1 curves
f_scores = np.linspace(0.1, 0.9, num=9)
for f_score in f_scores:
    x = np.linspace(0.01, 1)
    y = f_score * x / (2 * x - f_score)
    valid = (y >= 0) & (y <= 1)
    plt.plot(x[valid], y[valid], color='gray', alpha=0.3, linestyle='--')
    if valid.any():
        plt.text(x[valid][-1] + 0.02, y[valid][-1], f'F={f_score:.1f}', color='gray', fontsize=10)

# Scatter plot for individual images (OIS distribution)
plt.scatter(mem_scatter_R, mem_scatter_P, alpha=0.3, color='#1f77b4', edgecolor='none', s=40)
plt.scatter(can_scatter_R, can_scatter_P, alpha=0.3, color='#ff7f0e', edgecolor='none', s=40)

# ODS Main Points
plt.plot(mem_ODS_R, mem_ODS_P, marker='o', color='#1f77b4', markersize=12, markeredgecolor='black', markeredgewidth=1.5,
         label=f'Memristive (ODS={mem_ODS_F1:.3f}, OIS={mem_OIS_F1:.3f}, AP={mem_AP:.3f})')
plt.plot(can_ODS_R, can_ODS_P, marker='s', color='#ff7f0e', markersize=12, markeredgecolor='black', markeredgewidth=1.5,
         label=f'Canny (ODS={can_ODS_F1:.3f}, OIS={can_OIS_F1:.3f}, AP={can_AP:.3f})')

plt.xlim([0.0, 1.08])
plt.ylim([0.0, 1.05])
plt.title('Precision-Recall Evaluation on Boundary Detection', fontsize=16, pad=15)
plt.xlabel('Recall', fontsize=14)
plt.ylabel('Precision', fontsize=14)

# Legend formatting exactly like official papers
plt.legend(loc='lower left', fontsize=11, frameon=True, framealpha=0.9, edgecolor='black')
plt.grid(True, linestyle=':', alpha=0.7)

# <--- 4. CLAVE: Ajuste manual del margen derecho. 
#    right=0.92 significa que los ejes ocuparán el 92% izquierdo, dejando el 8% derecho para las etiquetas F=
plt.subplots_adjust(right=0.92) 

#plt.tight_layout()
plt.savefig('PR_Benchmark_Evaluation.png', dpi=300, bbox_inches='tight')
print("\nPlot saved successfully as 'PR_Benchmark_Evaluation.png'")
plt.show()