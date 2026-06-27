import numpy as np
import cv2
import tkinter as tk
from tkinter import filedialog

def calculate_channel_chi_square(channel_data):
    total_pixels = channel_data.size
    expected_frequency = total_pixels / 256.0
    
    # Calculate observed frequencies for bins 0 to 255
    observed_frequencies, _ = np.histogram(channel_data.ravel(), bins=256, range=(0, 256))
    
    O = observed_frequencies.astype(np.float64)
    E = float(expected_frequency)
    
    # Chi-Square Formula
    chi_square_stat = np.sum(((O - E) ** 2) / E)
    return chi_square_stat

def main():
    root = tk.Tk()
    root.withdraw()

    print("--- Multi-Channel Chi-Square Uniformity Test ---")
    
    enc_paths = []
    for i in range(1, 5):
        print(f"Select Encrypted Image {i} (PNG/BMP preferred)...")
        e_path = filedialog.askopenfilename(title=f"Select Encrypted Image {i}")
        if not e_path:
            print(f"Error: Image {i} selection canceled.")
            return
        enc_paths.append(e_path)

    results = []
    critical_value = 293.2478

    for i, path in enumerate(enc_paths):
        # Read image unchanged to retain all color/alpha channels
        img_e = cv2.imread(path, cv2.IMREAD_UNCHANGED)
        
        if img_e is None:
            print(f"Error loading image {i+1}")
            return
        
        # Determine channels
        if len(img_e.shape) == 2:
            # Grayscale fallback
            channels = [img_e]
            channel_names = ['Grayscale']
        elif img_e.shape[2] == 3:
            # OpenCV reads as BGR
            channels = [img_e[:,:,2], img_e[:,:,1], img_e[:,:,0]]
            channel_names = ['R', 'G', 'B']
        elif img_e.shape[2] == 4:
            # BGRA image
            channels = [img_e[:,:,2], img_e[:,:,1], img_e[:,:,0], img_e[:,:,3]]
            channel_names = ['R', 'G', 'B', 'Alpha']
        else:
            print(f"Unsupported channel dimensions in image {i+1}")
            return
            
        img_results = {}
        for name, ch in zip(channel_names, channels):
            img_results[name] = calculate_channel_chi_square(ch)
            
        results.append({
            'index': i + 1,
            'ch_stats': img_results,
            'critical': critical_value
        })

    # Generate LaTeX Table Code
    latex_str = (
        "\\begin{table}[htbp]\n"
        "\\centering\n"
        "\\caption{Multi-Channel Chi-Square ($\\chi^2$) Uniformity Test Results for Cipher Images}\n"
        "\\label{tab:multichannel_chi_square}\n"
        "\\begin{tabular}{|c|c|c|c|c|c|}\n"
        "\\hline\n"
        "Cipher Image & R Channel & G Channel & B Channel & Alpha Channel & Critical Value \\\\\n"
        "\\hline\n"
    )

    for res in results:
        stats = res['ch_stats']
        r_val = f"{stats['R']:.4f}" if 'R' in stats else "N/A"
        g_val = f"{stats['G']:.4f}" if 'G' in stats else "N/A"
        b_val = f"{stats['B']:.4f}" if 'B' in stats else "N/A"
        a_val = f"{stats['Alpha']:.4f}" if 'Alpha' in stats else "N/A"
        
        # Handle Grayscale image edge-case formatting if necessary
        if 'Grayscale' in stats:
            r_val = g_val = b_val = f"{stats['Grayscale']:.4f}"
            
        latex_str += f"Cipher Input {res['index']} & {r_val} & {g_val} & {b_val} & {a_val} & {res['critical']:.4f} \\\\\n\\hline\n"

    latex_str += "\\end{tabular}\n\\end{table}\n"

    output_filename = "multichannel_chi_square_results.txt"
    with open(output_filename, "w") as text_file:
        text_file.write(latex_str)

    print(f"\nCompleted successfully. Structured LaTeX table saved to '{output_filename}'.")

if __name__ == "__main__":
    main()