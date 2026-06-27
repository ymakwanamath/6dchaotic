import numpy as np
import cv2
import tkinter as tk
from tkinter import filedialog
import os

def export_nist_bitstream(img_path, output_dir, index):
    # Read image exactly as it is
    img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
    if img is None:
        print(f"Failed to load image {index}")
        return False
        
    # Flatten the image matrix to a 1D array of bytes
    flat_bytes = img.ravel()
    
    # Unpack bytes into an array of bits
    bits = np.unpackbits(flat_bytes)
    
    # Convert the integer array to a single giant string of '0' and '1' characters
    # This is the safest format for the official NIST software to read
    bit_string = ''.join(bits.astype(str))
    
    # Save to a text file
    filename = os.path.join(output_dir, f"nist_bitstream_input_{index}.txt")
    with open(filename, 'w') as f:
        f.write(bit_string)
        
    print(f"Image {index} exported: {len(bit_string)} bits saved to {filename}")
    return True

def main():
    root = tk.Tk()
    root.withdraw()

    print("--- NIST Bitstream Extractor ---")
    
    # Ask for output directory first
    print("Select a folder to save the NIST text files...")
    output_dir = filedialog.askdirectory(title="Select Output Folder")
    if not output_dir:
        print("No output folder selected. Exiting.")
        return

    # Select encrypted images
    for i in range(1, 5):
        print(f"\nSelect Encrypted Image {i}...")
        path = filedialog.askopenfilename(title=f"Select Encrypted Image {i}")
        if not path:
            break
        
        export_nist_bitstream(path, output_dir, i)

    print("\nExtraction complete. You can now feed these .txt files into the NIST software.")

if __name__ == "__main__":
    main()