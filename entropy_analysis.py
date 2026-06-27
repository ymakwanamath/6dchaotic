import numpy as np
import cv2
import tkinter as tk
from tkinter import filedialog

def calculate_global_entropy(img):
    if img is None:
        return 0.0
    # Calculate histogram and probabilities
    hist, _ = np.histogram(img.ravel(), bins=256, range=(0, 256))
    prob = hist / hist.sum()
    # Filter out zero probabilities to avoid log2(0) error
    prob = prob[prob > 0]
    global_entropy = -np.sum(prob * np.log2(prob))
    return global_entropy

def calculate_local_entropy(img, num_blocks=10, block_size=16):
    if img is None:
        return 0.0
    h, w = img.shape
    block_entropies = []
    
    # Seed for reproducibility of random block selection
    np.random.seed(42) 
    
    # Pick random non-overlapping blocks
    for _ in range(num_blocks):
        # Ensure block fits within dimensions
        max_y = h - block_size
        max_x = w - block_size
        if max_y <= 0 or max_x <= 0:
            return 0.0
            
        y = np.random.randint(0, max_y)
        x = np.random.randint(0, max_x)
        
        block = img[y:y+block_size, x:x+block_size]
        
        hist, _ = np.histogram(block.ravel(), bins=256, range=(0, 256))
        prob = hist / hist.sum()
        prob = prob[prob > 0]
        
        if len(prob) > 0:
            block_entropies.append(-np.sum(prob * np.log2(prob)))
            
    return np.mean(block_entropies) if block_entropies else 0.0

def main():
    # Hide the main tkinter root window
    root = tk.Tk()
    root.withdraw()

    print("--- File Selection Phase ---")
    
    # Dynamic inputs for Plain, Encrypted, and Decrypted paths
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

    # Initialize data dictionary for the results
    results = []

    for i in range(4):
        # Read images in grayscale
        img_p = cv2.imread(plain_paths[i], cv2.IMREAD_GRAYSCALE)
        img_e = cv2.imread(enc_paths[i], cv2.IMREAD_GRAYSCALE)
        img_d = cv2.imread(dec_paths[i], cv2.IMREAD_GRAYSCALE)
        
        # Calculations
        p_global = calculate_global_entropy(img_p)
        p_local  = calculate_local_entropy(img_p)
        
        e_global = calculate_global_entropy(img_e)
        e_local  = calculate_local_entropy(img_e)
        
        d_global = calculate_global_entropy(img_d)
        d_local  = calculate_local_entropy(img_d)
        
        results.append({
            'index': i + 1,
            'p_glob': p_global, 'p_loc': p_local,
            'e_glob': e_global, 'e_loc': e_local,
            'd_glob': d_global, 'd_loc': d_local
        })

    # Generate LaTeX code string
    latex_str = (
        "\\begin{table}[htbp]\n"
        "\\centering\n"
        "\\caption{Global and Local Shannon Entropy Analysis results}\n"
        "\\label{tab:entropy_analysis}\n"
        "\\begin{tabular}{|c|c|c|c|c|c|c|}\n"
        "\\hline\n"
        "Image & \\multicolumn{2}{c|}{Plain Image} & \\multicolumn{2}{c|}{Encrypted Image} & \\multicolumn{2}{c|}{Decrypted Image} \\\\\n"
        "\\cline{2-7}\n"
        "No. & Global Entropy & Local Entropy & Global Entropy & Local Entropy & Global Entropy & Local Entropy \\\\\n"
        "\\hline\n"
    )

    for res in results:
        latex_str += (
            f"Input {res['index']} & {res['p_glob']:.4f} & {res['p_loc']:.4f} & "
            f"{res['e_glob']:.4f} & {res['e_loc']:.4f} & "
            f"{res['d_glob']:.4f} & {res['d_loc']:.4f} \\\\\n"
            "\\hline\n"
        )

    latex_str += "\\end{tabular}\n\\end{table}\n"

    # Write LaTeX content to a text file
    output_filename = "entropy_results_table.txt"
    with open(output_filename, "w") as text_file:
        text_file.write(latex_str)

    print(f"\nProcessing complete. LaTeX code successfully written to '{output_filename}'.")

if __name__ == "__main__":
    main()