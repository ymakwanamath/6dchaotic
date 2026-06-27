import numpy as np
import hashlib
import oqs
import os
import math
from collections import Counter
from PIL import Image

# ==========================================
# Cryptographic Core Algorithms
# ==========================================

def generate_parameters(k_master):
    params = {}
    params['alpha'] = [4.1 + (k_master[i] / 255.0) * 0.8 for i in range(0, 4)]
    params['beta'] = [0.8 + (k_master[i] / 255.0) * 0.4 for i in range(4, 8)]
    params['mu'] = [0.005 + (k_master[i] / 255.0) * 0.010 for i in range(8, 10)]
    params['delta'] = [0.1 + (k_master[i] / 255.0) * 0.2 for i in range(10, 12)]
    params['gamma'] = 0.01 + (k_master[12] / 255.0) * 0.04
    return params

def generate_initial_states(k_master):
    shake = hashlib.shake_256()
    shake.update(k_master)
    psi = shake.digest(32)
    
    x = [0.0] * 6
    for i in range(3):
        chunk_sum = sum(psi[j] * (2 ** (8 * (3 - (j % 4)))) for j in range(i*4, i*4 + 4))
        x[i] = chunk_sum / (2**32)
        
    for i in range(3, 6):
        chunk_sum = sum(psi[j] * (2 ** (8 * (3 - (j % 4)))) for j in range(i*4, i*4 + 4))
        x[i] = -1 + 2 * (chunk_sum / (2**32))
        
    return x

def chaotic_sequence_generation(k_master, length):
    params = generate_parameters(k_master)
    state = generate_initial_states(k_master)
    
    for _ in range(100):
        x_next = params['alpha'][0] * state[0] * (1 - state[0]) + params['beta'][0] * (state[1]**2)
        y_next = params['alpha'][1] * state[1] * (1 - state[1]) + params['beta'][1] * (state[2]**2) + params['mu'][0] * state[3]
        z_next = params['alpha'][2] * state[2] * (1 - state[2]) + params['beta'][2] * (state[0]**2) + params['mu'][1] * state[4]
        u_next = params['delta'][0] * np.sin(np.pi * state[3]) + params['gamma'] * state[5]
        v_next = params['delta'][1] * np.cos(np.pi * state[4]) + params['gamma'] * state[3]
        w_next = np.tanh(state[3] + state[4]) * np.sin(np.pi * state[5])
        state = [x_next, y_next, z_next, u_next, v_next, w_next]

    sequences = [np.zeros(length) for _ in range(6)]
    for i in range(length):
        x_next = params['alpha'][0] * state[0] * (1 - state[0]) + params['beta'][0] * (state[1]**2)
        y_next = params['alpha'][1] * state[1] * (1 - state[1]) + params['beta'][1] * (state[2]**2) + params['mu'][0] * state[3]
        z_next = params['alpha'][2] * state[2] * (1 - state[2]) + params['beta'][2] * (state[0]**2) + params['mu'][1] * state[4]
        u_next = params['delta'][0] * np.sin(np.pi * state[3]) + params['gamma'] * state[5]
        v_next = params['delta'][1] * np.cos(np.pi * state[4]) + params['gamma'] * state[3]
        w_next = np.tanh(state[3] + state[4]) * np.sin(np.pi * state[5])
        
        state = [x_next, y_next, z_next, u_next, v_next, w_next]
        for j in range(6):
            sequences[j][i] = state[j]
            
    return sequences

def fisher_yates_shuffle(arr, seed_bytes):
    shuffled = list(arr)
    n = len(shuffled)
    for i in range(n - 1, 0, -1):
        j = seed_bytes[n - 1 - i] % (i + 1)
        shuffled[i], shuffled[j] = shuffled[j], shuffled[i]
    return shuffled

def construct_dynamic_sbox(k_round):
    shake = hashlib.shake_256()
    shake.update(k_round)
    digest = shake.digest(32)
    
    s_base = list(range(16))
    s_4h = fisher_yates_shuffle(s_base, digest[0:8])
    s_4l = fisher_yates_shuffle(s_base, digest[8:16])
    rk = [digest[16], digest[17], digest[18], digest[19]]
    p_box = [0, 4, 1, 5, 2, 6, 3, 7]
    
    S = np.zeros(256, dtype=np.uint8)
    
    for input_val in range(256):
        state = input_val
        for round_idx in range(4):
            state ^= rk[round_idx]
            high_nibble = (state >> 4) & 0x0F
            low_nibble = state & 0x0F
            state = (s_4h[high_nibble] << 4) | s_4l[low_nibble]
            
            new_state = 0
            for target_bit in range(8):
                source_bit = p_box[target_bit]
                bit_val = (state >> source_bit) & 1
                new_state |= (bit_val << target_bit)
            state = new_state
        S[input_val] = state
        
    return S

def quantize_sequence(seq):
    return np.array([int((abs(x) * 1e14)) % 256 for x in seq], dtype=np.uint8)

def bidirectional_diffusion_forward(V, Kd):
    L = len(V)
    Vf = np.zeros(L, dtype=np.uint8)
    Vb = np.zeros(L, dtype=np.uint8)
    
    for i in range(L):
        prev_fwd = int(Vf[i-1]) if i > 0 else 0
        Vf[i] = (int(V[i]) + prev_fwd + int(Kd[i])) % 256
        
    for i in range(L - 1, -1, -1):
        prev_back = int(Vb[i+1]) if i < L - 1 else 0
        Vb[i] = (int(Vf[i]) + prev_back + int(Kd[L - 1 - i])) % 256
        
    return Vb

def encrypt_image(image_channels, k_master):
    L = len(image_channels[0].flatten())
    sequences = chaotic_sequence_generation(k_master, L)
    
    encrypted_channels = []
    S = construct_dynamic_sbox(k_master)
    Kd = quantize_sequence(sequences[4] + sequences[5]) 
    
    for c_idx in range(len(image_channels)):
        channel_flat = image_channels[c_idx].flatten()
        seq = quantize_sequence(sequences[c_idx])
        
        xored = np.bitwise_xor(channel_flat, seq)
        substituted = np.array([S[val] for val in xored], dtype=np.uint8)
        diffused = bidirectional_diffusion_forward(substituted, Kd)
        
        encrypted_channels.append(diffused.reshape(image_channels[c_idx].shape))
        
    return encrypted_channels

# ==========================================
# Entropy Calculation
# ==========================================

def calculate_entropy(image_array):
    """Calculates Shannon entropy for an image matrix."""
    flat_array = image_array.flatten()
    counts = Counter(flat_array)
    total_pixels = len(flat_array)
    
    entropy = 0.0
    for count in counts.values():
        probability = count / total_pixels
        entropy -= probability * math.log2(probability)
        
    return entropy

if __name__ == "__main__":
    # Generate Key using ML-KEM
    kem = oqs.KeyEncapsulation("ML-KEM-768")
    public_key = kem.generate_keypair()
    ciphertext, key_A = kem.encap_secret(public_key)
    kem.free()

    # Image Dimensions
    width, height = 256, 256
    
    # Generate Plain Black and White Images (3 channels: RGB)
    black_channels = [np.zeros((height, width), dtype=np.uint8) for _ in range(3)]
    white_channels = [np.full((height, width), 255, dtype=np.uint8) for _ in range(3)]
    
    print("--- Black Image Encryption ---")
    encrypted_black = encrypt_image(black_channels, key_A)
    enc_black_arr = np.stack(encrypted_black, axis=-1)
    black_entropy = calculate_entropy(enc_black_arr)
    print(f"Encrypted Black Image Entropy: {black_entropy:.4f}")
    
    # Save encrypted black image
    Image.fromarray(enc_black_arr, 'RGB').save("encrypted_black.png", "PNG")
    print("Saved as 'encrypted_black.png'\n")

    print("--- White Image Encryption ---")
    encrypted_white = encrypt_image(white_channels, key_A)
    enc_white_arr = np.stack(encrypted_white, axis=-1)
    white_entropy = calculate_entropy(enc_white_arr)
    print(f"Encrypted White Image Entropy: {white_entropy:.4f}")
    
    # Save encrypted white image
    Image.fromarray(enc_white_arr, 'RGB').save("encrypted_white.png", "PNG")
    print("Saved as 'encrypted_white.png'\n")