import os
import json
import glob
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def load_reprojection_data(base_dir):
    """Carrega os erros de reprojeção dos arquivos sequence_log.json."""
    data_list = []
    
    target_methods = ['aruco', 'hsv', 'yolo', 'auto']
    
    translation_map = {
        "caixa_marcador": "Marker Box",
        "caixa_padrao": "Standard Box",
        "multiplas_caixas": "Multiple Boxes"
    }
    
    json_files = glob.glob(os.path.join(base_dir, "**", "sequence_log.json"), recursive=True)
    
    if not json_files:
        print(f"[WARN] No sequence_log.json files found in {base_dir}")
        return pd.DataFrame()

    for file_path in json_files:
        parts = os.path.normpath(file_path).split(os.sep)
        
        if len(parts) < 4:
            continue
            
        method = parts[-3].lower()
        raw_test_case = parts[-4]
        
        test_case = translation_map.get(raw_test_case, raw_test_case.replace("_", " ").title())
        
        if method not in target_methods:
            continue
            
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = json.load(f)
                
                if "phases" in content:
                    for phase_key, phase_data in content["phases"].items():
                        if "reproj_errors_px_vector" in phase_data:
                            errors = phase_data["reproj_errors_px_vector"]
                            
                            for err in errors:
                                data_list.append({
                                    "Test Case": test_case,
                                    "Method": method.upper(),
                                    "Reprojection Error": err
                                })
        except Exception as e:
            print(f"[ERROR] Failed to read {file_path}: {e}")
            
    return pd.DataFrame(data_list)

def plot_reprojection_errors(df, output_path=None):
    if df.empty:
        print("[ERROR] No data available to plot.")
        return

    sns.set_theme(style="ticks", font="serif")
    plt.rcParams.update({'font.size': 11})
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    method_palette = {
        'ARUCO': '#0072B2',
        'HSV':   '#D55E00',
        'YOLO':  '#009E73',
        'AUTO':  '#CC79A7',
    }

    sns.boxplot(
        data=df,
        x="Test Case",
        y="Reprojection Error",
        hue="Method",
        order=['Marker Box', 'Standard Box', 'Multiple Boxes'],
        hue_order=['ARUCO', 'HSV', 'YOLO', 'AUTO'],
        palette=method_palette,
        linewidth=1.2,
        fliersize=3,
        showmeans=True, 
        meanprops={"marker": "x", "markeredgecolor": "black", "markersize": 5},
        ax=ax
    )

    ax.set_xlabel("Evaluation Scenarios", fontweight='bold', labelpad=10)
    ax.set_ylabel("Reprojection Error (pixels)", fontweight='bold', labelpad=10)
    
    for spine in ax.spines.values():
        spine.set_edgecolor('black')
        spine.set_linewidth(1.0)
        
    ax.yaxis.grid(True, linestyle='--', alpha=0.6, color='gray')
    ax.xaxis.grid(False)

    handles, labels = ax.get_legend_handles_labels()
    ax.legend(
        handles=handles, 
        labels=labels, 
        title="Detection Method", 
        frameon=True, 
        edgecolor='black',
        fancybox=False
    )
    
    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, format=output_path.split('.')[-1], bbox_inches='tight', dpi=300)
        print(f"[SUCCESS] Vectorized figure saved to {output_path}")
    else:
        plt.show()

if __name__ == "__main__":
    DATA_ROOT_FOLDER = "./logs"
    OUTPUT_IMAGE_PATH = "reprojection_errors_comparison.pdf"

    print("[INFO] Parsing tracking logs...")
    data_df = load_reprojection_data(DATA_ROOT_FOLDER)
    
    print(f"[INFO] Total data points loaded: {len(data_df)}")
    if not data_df.empty:
        print("\nData Summary:")
        print(data_df.groupby(['Test Case', 'Method']).size().unstack(fill_value=0))
        
        print("\n[INFO] Rendering plot...")
        plot_reprojection_errors(data_df, output_path=OUTPUT_IMAGE_PATH)