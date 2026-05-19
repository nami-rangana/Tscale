import numpy as np
import pymbar

def run_mbar_from_npz(npz_file="meld_thermodynamics.npz", T_ref=300.0):
    print(f"Loading binary thermodynamics data from {npz_file}...")
    data = np.load(npz_file)
    
    U_matrix = data['U']  
    V_matrix = data['V']  
    T_array = data['T']   
    w_alphas = data['W']  
    
    N_frames_per_rep, K = U_matrix.shape
    N_total = N_frames_per_rep * K
    N_k = np.full(K, N_frames_per_rep)
    
    print(f"Detected {N_frames_per_rep} frames across {K} replicas. Total samples: {N_total}")

    U_flat = U_matrix.flatten(order='F') 
    V_flat = V_matrix.flatten(order='F')
    origin_k = np.repeat(np.arange(K), N_frames_per_rep)

    kb = 0.008314462618 
    beta_k = 1.0 / (kb * T_array)

    print("Analytically un-scaling restraint matrices using detected MELD weights...")
    V_raw = np.zeros_like(V_flat)
    for i in range(N_total):
        w_origin = w_alphas[origin_k[i]]
        if w_origin > 1e-6:
            V_raw[i] = V_flat[i] / w_origin
        else:
            V_raw[i] = 0.0 

    # --- THE UNSAMPLED STATE TRICK ---
    # Instead of K states, we initialize a matrix for K + 1 states!
    print(f"Constructing the reduced potential u_kn matrix for {K + 1} states...")
    u_kn = np.zeros((K + 1, N_total), dtype=np.float64)
    
    for k in range(K):
        # The 30 sampled states
        u_kn[k, :] = beta_k[k] * (U_flat + (w_alphas[k] * V_raw))

    # Row K (The 31st row) is our Target Unbiased State
    print(f"Injecting Target Unbiased State (T = {T_ref}K, Restraints = 0.0)...")
    beta_target = 1.0 / (kb * T_ref)
    u_kn[K, :] = beta_target * U_flat  # Physics only!
    
    # We append a '0' to N_k to tell PyMBAR that no frames were actually sampled here
    N_k_with_target = np.append(N_k, 0)

    # --- SOLVE ---
    print("\nSolving MBAR equations (JAX 64-bit backend active)...")
    # (It is perfectly normal to see a solver fallback warning here for MELD data)
    mbar = pymbar.MBAR(u_kn, N_k_with_target)
    
    print("\nExtracting unbiased statistical weights...")
    # Because we made the Target State the last row, its weights are 
    # sitting perfectly in the last row of the W_nk matrix!
    W_unbiased = np.array(mbar.W_nk[K, :])
    
    return W_unbiased, U_flat

if __name__ == "__main__":
    target_temperature = 300.0
    
    final_weights, combined_physics = run_mbar_from_npz("meld_thermodynamics.npz", target_temperature)
    
    print("Saving results to disk...")
    np.savetxt("mbar_weights.dat", final_weights, fmt="%.8e", header=f"Normalized MBAR weights (Unbiased at {target_temperature} K)")
    np.savetxt("combined_energies.dat", combined_physics, fmt="%.3f", header="Physical Potential Energies (kJ/mol)")
    
    print("\nSuccess! Weights generated.")