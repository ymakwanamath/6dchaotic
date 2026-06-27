import numpy as np
import cv2
import tkinter as tk
from tkinter import filedialog

def calculate_correlation_all_pairs(img):
    if img is None:
        return 0.0, 0.0, 0.0
    
    # Convert to float64 to prevent arithmetic overflow
    img = img.astype(np.float64)
    
    # 1. Horizontal Pairs: Compare all pixels except the last column with their right neighbor
    u_h = img[:, :-1].ravel()
    v_h = img[:, 1:].ravel()
    r_h = np.corrcoef(u_h, v_h)[0, 1]
    
    # 2. Vertical Pairs: Compare all pixels except the last row with their bottom neighbor
    u_v = img[:-1, :].ravel()
    v_v = img[1:, :].ravel()
    r_v = np.corrcoef(u_v, v_v)[0, 1]
    
    # 3. Diagonal Pairs: Compare all pixels except last row/col with bottom-right neighbor
    u_d = img[:-1, :-1].ravel()
    v_d = img[1:, 1:].ravel()
    r_d = np.corrcoef(u_d, v_d)[0, 1]
    
    # Handle NaN values
    r_h = 0.0 if np.isnan(r_h) else r_h
    r_v = 0.0 if np.isnan(r_v) else r_v
    r_d = 0.0 if np.isnan(r_d) else r_d
    
    return r_h, r_v, r_d

def main():
    root = tk.Tk()
    root.withdraw()

    print("--- Correlation (All Pairs) File Selection Phase ---")
    
    plain_paths = []
    enc_paths = []
    dec_paths = []
    
    for i in range(1, 5):
        print(f"\nSelect Plain Image {i}...")
        p_path = filedialog.askopenfilename(title=f"Select Plain Image {i}")
        plain_paths.append(p_path)
        
        print(f"Select Encrypted Image {i}...")
        e_path = filedialog.askopenfilename(title=f"Select Encrypted Image {i}")
        enc_paths.append(e_path)
        
        print(f"Select Decrypted Image {i}...")
        d_path = filedialog.askopenfilename(title=f"Select Decrypted Image {i}")
        dec_paths.append(d_path)

    results = []

    for i in range(4):
        img_p = cv2.imread(plain_paths[i], cv2.IMREAD_GRAYSCALE)
        img_e = cv2.imread(enc_paths[i], cv2.IMREAD_GRAYSCALE)
        img_d = cv2.imread(dec_paths[i], cv2.IMREAD_GRAYSCALE)
        
        p_h, p_v, p_d = calculate_correlation_all_pairs(img_p)
        e_h, e_v, e_d = calculate_correlation_all_pairs(img_e)
        d_h, d_v, d_d = calculate_correlation_all_pairs(img_d)
        
        results.append({
            'index': i + 1,
            'plain': (p_h, p_v, p_d),
            'cipher': (e_h, e_v, e_d),
            'decrypted': (d_h, d_v, d_d)
        })

    latex_str = (
        "\\begin{table}[htbp]\n"
        "\\centering\n"
        "\\caption{Global Adjacent Pixel Correlation Coefficients (All Pairs Analysis)}\n"
        "\\label{tab:correlation_all_pairs}\n"
        "\\begin{tabular}{|c|l|c|c|c|}\n"
        "\\hline\n"
        "Image No. & Image State & Horizontal & Vertical & Diagonal \\\\\n"
        "\\hline\n"
    )

    for res in results:
        idx = res['index']
        p = res['plain']
        c = res['cipher']
        d = res['decrypted']
        
        latex_str += (
            f"\\multirow{{3}}{{*}}{{Input {idx}}} & Plain Image & {p[0]:.4f} & {p[1]:.4f} & {p[2]:.4f} \\\\\n"
            f" & Encrypted Image & {c[0]:.4f} & {c[1]:.4f} & {c[2]:.4f} \\\\\n"
            f" & Decrypted Image & {d[0]:.4f} & {d[1]:.4f} & {d[2]:.4f} \\\\\n"
            "\\hline\n"
        )

    latex_str += "\\end{tabular}\n\\end{table}\n"

    output_filename = "correlation_all_pairs_results.txt"
    with open(output_filename, "w") as text_file:
        text_file.write(latex_str)

    print(f"\nProcessing complete. All-pairs LaTeX table written to '{output_filename}'.")

if __name__ == "__main__":
    main()