import re
import os
from pathlib import Path
import polars as pl
import matplotlib.pyplot as plt
import numpy as np

def get_si_prefix(value: float) -> tuple[float, str, str]:
    """
    Determina el prefijo SI adecuado para un valor numérico.
    Retorna: (factor_de_escala, símbolo_del_prefijo, nombre_legible)
    """
    if value == 0:
        return 1.0, "", ""
    
    abs_val = abs(value)
    
    prefixes = [
        (1e-12, 1e-9, 1e12, 'p', 'pico'),
        (1e-9, 1e-6, 1e9, 'n', 'nano'),
        (1e-6, 1e-3, 1e6, 'µ', 'micro'),
        (1e-3, 1e0, 1e3, 'm', 'mili'),
        (1e0, 1e3, 1e0, '', ''),
        (1e3, 1e6, 1e-3, 'k', 'kilo'),
        (1e6, 1e9, 1e-6, 'M', 'mega'),
        (1e9, float('inf'), 1e-9, 'G', 'giga'),
    ]
    
    for low, high, factor, symbol, name in prefixes:
        if low <= abs_val < high:
            return factor, symbol, name
    
    return 1.0, "", ""

def auto_scale_axis(data: np.ndarray, unit: str = '', label_base: str = '') -> tuple[np.ndarray, str, str]:
    """
    Escala automáticamente un array de datos y prepara la etiqueta del eje.
    """
    clean_data = data[~np.isnan(data) & np.isfinite(data)]
    if len(clean_data) == 0:
        return data, "", f"{label_base} ({unit})"
    
    max_val = np.max(np.abs(clean_data))
    factor, symbol, _ = get_si_prefix(max_val)
    
    data_scaled = data * factor if factor != 1.0 else data
    
    prefix_display = f"{symbol}" if symbol else ""
    axis_label = f"{label_base} ({prefix_display}{unit})" if unit else f"{label_base} ({prefix_display})"
    
    return data_scaled, symbol, axis_label

def parse_ngspice_file(filepath: str) -> tuple[dict, int]:
    """Lee el archivo .dat.ngspice y extrae time y variables dependientes."""
    #print("[1/4] Leyendo y parseando el archivo de simulación...")
    print("[1/4] Read and parse file of simulation...")
    file_path = Path(filepath)
    if not file_path.exists():
        raise FileNotFoundError(f"No file found: {filepath}")

    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()

    time_match = re.search(r'<indep\s+time\s+(\d+)>(.*?)</indep>', content, re.DOTALL)
    if not time_match:
        raise ValueError("No found block <indep time N>. verify format file.")

    n_rows = int(time_match.group(1))
    time_data = [float(x) for x in time_match.group(2).split()]
    print(f"   ✅ Variable  'time' extract from {n_rows} points.")

    dep_pattern = re.compile(r'<dep\s+.*?\(([^)]+)\).*?>(.*?)</dep>', re.DOTALL)
    variables_data = {'time': time_data}

    for match in dep_pattern.finditer(content):
        var_name = match.group(1)
        values = [float(x) for x in match.group(2).split()]
        variables_data[var_name] = values
        print(f"   📦 Found: '{var_name}' ({len(values)} points)")

    return variables_data, n_rows

def construir_dataframe(data_dict: dict, variables_interes: list) -> pl.DataFrame:
    """Construye un DataFrame de Polars filtrando solo las variables solicitadas."""
    #print("[2/4] Construyendo DataFrame con Polars y filtrando variables de interés...")
    print("[2/4] Built DataFrame with Polars and filtering variables of interest...")
    
    disponibles = [v for v in variables_interes if v in data_dict]
    no_encontradas = [v for v in variables_interes if v not in data_dict]

    if no_encontradas:
        print(f"   ⚠️ [WARNING] Variables not present in the file: {no_encontradas}")
    if not disponibles:
        raise ValueError("None variables of interest present.")

    df = pl.DataFrame({v: data_dict[v] for v in ['time'] + disponibles})
    print(f"   📊 DataFrame created with the form: {df.shape}")
    return df

def generar_graficas(df: pl.DataFrame, variables_v: list, variables_r: list, output_dir: str = 'IMG', dpi: int = 300):
    """Genera gráficas individuales y combinadas separando Voltajes y Resistencias."""
    print(f"[3/4] Generate graph and save in '{output_dir}/'...")
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    plt.style.use('seaborn-v0_8-whitegrid')
    plt.rcParams.update({
        'font.family': 'sans-serif',
        'font.size': 11,
        'axes.titlesize': 13,
        'axes.labelsize': 12,
        'legend.fontsize': 10,
        'figure.dpi': dpi,
        'savefig.bbox': 'tight'
    })

    time_np = df['time'].to_numpy()
    time_scaled, time_prefix, time_label = auto_scale_axis(time_np, unit='s', label_base='Time')

    

    # Función auxiliar para gráficas combinadas (CORREGIDA)
    # Función auxiliar para gráficas combinadas (ESCALA REAL CRUDA)
    def graficar_combinada(vars_list, unit_str, label_str, filename,threshold=None):
        disponibles = [v for v in vars_list if v in df.columns]
        if not disponibles: return
        
        print(f"   🖌️ Graph {label_str} in lineal scale...")
        
        fig, ax = plt.subplots(figsize=(10, 6))
        palette = plt.cm.tab10.colors

        # Graficamos los datos crudos directos de la matriz, 
        # sin ningún factor engañoso que infle las variables
        for i, var in enumerate(disponibles):
            var_np = df[var].to_numpy()
            ax.plot(time_scaled, var_np, color=palette[i % 10], linewidth=2.5, label=var)
        
        # =========================================================
        # NUEVA LÍNEA: DIBUJAR EL UMBRAL (Mth)
        # =========================================================
        if threshold is not None:
            ax.axhline(y=threshold, color='red', linestyle='--', linewidth=2, alpha=0.7, label=f'Mth ({threshold} {unit_str})')
        # =========================================================
        
        ax.set_xlabel(time_label)
        ax.set_ylabel(f'{label_str} ({unit_str})')
        ax.set_title(f'Response: {label_str} Fuses')
        ax.legend(loc='best', frameon=True, shadow=False)
        ax.grid(True, alpha=0.3, linestyle='--')
        
        # Obligamos a matplotlib a mostrar números normales (ej. 1000) 
        # en lugar de notación científica o prefijos automáticos
        ax.ticklabel_format(axis='both', style='plain', useOffset=False)

        out_path = Path(output_dir) / filename
        fig.savefig(out_path, dpi=dpi)
        plt.close(fig)
        print(f"   ✅ Save: {out_path} | axe Y in [{unit_str}]")
        
    # (Al final de generar_graficas, llamas a la función sin parámetros extra)
    graficar_combinada(variables_v, 'V', 'Voltaje', 'combined_v_vs_time.png')
    graficar_combinada(variables_r, r'$\Omega$', 'Memristance', 'combined_r_vs_time.png',threshold=22.0)
    # Función auxiliar para gráficas combinadas (REEMPLAZA ESTA FUNCIÓN)
    def graficar_combinadaLoga(vars_list, unit_str, label_str, filename, use_log=False,threshold=None):
        disponibles = [v for v in vars_list if v in df.columns]
        if not disponibles: return
        
        print(f"   🖌️ Graph {label_str}...")
        
        global_max = 0.0
        for var in disponibles:
            var_np = df[var].to_numpy()
            clean_data = var_np[~np.isnan(var_np) & np.isfinite(var_np)]
            if len(clean_data) > 0:
                current_max = np.max(np.abs(clean_data))
                if current_max > global_max:
                    global_max = current_max
        
        factor, common_prefix, _ = get_si_prefix(global_max)
        
        # En escala logarítmica, forzamos la unidad base para ver el rango real (10^0 a 10^3)
        if use_log:
            factor = 1.0
            common_prefix = ""
            
        prefix_display = f"{common_prefix}" if common_prefix else ""
        y_label = f"{label_str} ({prefix_display}{unit_str})"
        if use_log:
            y_label += " (Log)"
        
        fig, ax = plt.subplots(figsize=(10, 6))
        palette = plt.cm.tab10.colors

        for i, var in enumerate(disponibles):
            var_np = df[var].to_numpy()
            var_scaled = var_np * factor if factor != 1.0 else var_np
            ax.plot(time_scaled, var_scaled, color=palette[i % 10], linewidth=2.5, label=var)
        # =========================================================
        # NUEVA LÍNEA: DIBUJAR EL UMBRAL (Mth)
        # =========================================================
        if threshold is not None:
            ax.axhline(y=threshold, color='red', linestyle='--', linewidth=2, alpha=0.7, label=f'Mth ({threshold} {unit_str})')
        # =========================================================
        ax.set_xlabel(time_label)
        ax.set_ylabel(y_label)
        ax.set_title(f'Response: {label_str} Fuses')
        ax.legend(loc='best', frameon=True, shadow=False)
        ax.grid(True, alpha=0.3, linestyle='--')
        
        # ACTIVAR ESCALA LOGARÍTMICA SI SE PIDE
        if use_log:
            ax.set_yscale('log')
        else:
            ax.ticklabel_format(axis='both', style='plain', useOffset=False)

        out_path = Path(output_dir) / filename
        fig.savefig(out_path, dpi=dpi)
        plt.close(fig)
        print(f"   ✅ Save: {out_path} | axe Y: [{prefix_display}{unit_str}]")

    # 2. Gráficas Combinadas (ACTUALIZA ESTAS DOS LÍNEAS AL FINAL DE generar_graficas)
    graficar_combinadaLoga(variables_v, 'V', 'Voltaje', 'combined_v_vs_timeLoga.png')
    graficar_combinadaLoga(variables_r, r'$\Omega$', 'Memristance', 'combined_r_vs_timeLoga.png', use_log=True,threshold=22.0)

    # Función auxiliar para graficar individualmente
    def graficar_individual(var, unit_str, label_str):
        if var not in df.columns: return
        print(f"   🖌️ Graph '{var}' ({label_str})...")
        var_np = df[var].to_numpy()
        var_scaled, var_prefix, var_label = auto_scale_axis(var_np, unit=unit_str, label_base=var)
        
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.plot(time_scaled, var_scaled, color='tab:blue', linewidth=2, marker='.', markersize=4, linestyle='-')
        ax.set_xlabel(time_label)
        ax.set_ylabel(var_label)
        ax.set_title(f'Response: {var}')
        ax.grid(True, alpha=0.3, linestyle='--')
        ax.ticklabel_format(axis='both', style='plain', useOffset=False)
        
        out_path = Path(output_dir) / f'{var}_vs_time.png'
        fig.savefig(out_path, dpi=dpi)
        plt.close(fig)

    # 1. Gráficas individuales
    for var in variables_v:
        graficar_individual(var, 'V', 'Voltaje')
        
    for var in variables_r:
        graficar_individual(var, r'$\Omega$', 'Memristance') # Usa el símbolo Omega

    

def main():
    ARCHIVO_ENTRADA = 'Nodes_probe.dat.ngspice' # Cambia por tu archivo real
    
    # Define las variables a extraer
    VARIABLES_INTERES_V = ['v01', 'v11', 'v21']
    VARIABLES_INTERES_R = ['Mp11', 'Mp01', 'Mq10']
    
    # Se combinan para procesar el DataFrame en una sola pasada
    TODAS_LAS_VARIABLES = VARIABLES_INTERES_V + VARIABLES_INTERES_R
    
    CARPETA_SALIDA = 'IMG'
    DPI_SALIDA = 300

    print("="*60)
    print("🚀 start processing NGSPICE/QUCS + scale SI")
    print("="*60)

    try:
        data_dict, n_rows = parse_ngspice_file(ARCHIVO_ENTRADA)
        df = construir_dataframe(data_dict, TODAS_LAS_VARIABLES)
        generar_graficas(df, VARIABLES_INTERES_V, VARIABLES_INTERES_R, CARPETA_SALIDA, DPI_SALIDA)

        print("\n" + "="*60)
        print("🎉 [COMPLETED] Graph")
        print("💡 Generate img.")
        print("="*60)
    except Exception as e:
        print(f"\n❌ Error in: {e}")
        import traceback
        traceback.print_exc()

if __name__ == '__main__':
    main()