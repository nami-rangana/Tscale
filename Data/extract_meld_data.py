import numpy as np
import os
import argparse
from concurrent.futures import ProcessPoolExecutor
import multiprocessing
import math

def _worker_extract_chunk(args):
    """
    Reads a chunk of frames from the MELD vault.
    Returns: start_step, U_chunk (physics), V_chunk (restraints)
    """
    target_dir, start_step, end_step, n_replicas = args
    from meld import vault
    
    os.chdir(target_dir)
    try:
        store = vault.DataStore.load_data_store()
        store.initialize('r')
    except Exception:
        return start_step, None, None
        
    chunk_frames = end_step - start_step
    
    # Pre-allocate the arrays for this specific chunk
    U_chunk = np.zeros((chunk_frames, n_replicas), dtype=np.float64)
    V_chunk = np.zeros((chunk_frames, n_replicas), dtype=np.float64)
    
    local_idx = 0
    for step in range(start_step, end_step):
        try:
            states = store.load_states(step)
        except Exception:
            break
            
        for k_rep, state in enumerate(states):
            groups = state.group_energies
            
            if groups is not None and len(groups) >= 7:
                # Based on MELD CHANGELOG mapping:
                # Group 0 is the pure physical forcefield (Amber + Grappa)
                U_chunk[local_idx, k_rep] = float(groups[0])
                
                # Groups 1-6 contain all artificial MELD restraints
                V_chunk[local_idx, k_rep] = float(np.sum(groups[1:]))
            else:
                U_chunk[local_idx, k_rep] = np.nan
                V_chunk[local_idx, k_rep] = np.nan
            
        local_idx += 1
        
    return start_step, U_chunk[:local_idx, :], V_chunk[:local_idx, :]

def orchestrate_extraction(data_dir, num_workers):
    execution_dir = os.getcwd()
    abs_path = os.path.abspath(data_dir)
    target_dir = os.path.dirname(abs_path) if os.path.basename(abs_path) == "Data" else abs_path
        
    print(f"Mounting MELD vault at {target_dir}...")
    os.chdir(target_dir)
    
    from meld import vault 
    store = vault.DataStore.load_data_store()
    store.initialize('r')
    
    n_replicas = store.n_replicas
    
    # Get max frames safely
    if hasattr(store, 'max_safe_frame') and store.max_safe_frame > 0:
        n_frames = store.max_safe_frame
    elif hasattr(store, 'max_saved_step') and store.max_saved_step > 0:
        n_frames = store.max_saved_step
    else:
        n_frames = 20000 
        
    store = None # Release the file handle
    
    print(f"Topology Detected: {n_replicas} Replicas across {n_frames} Frames.")
    print(f"Extracting Physics (Group 0) and Restraints (Groups 1-6) using {num_workers} CPU cores...")
    
    # Allocate master matrices
    U_unbiased_matrix = np.zeros((n_frames, n_replicas), dtype=np.float64)
    V_bias_matrix = np.zeros((n_frames, n_replicas), dtype=np.float64)
    
    chunk_size = max(1, math.ceil(n_frames / num_workers))
    tasks = []
    for i in range(num_workers):
        start = (i * chunk_size) + 1
        end = min((i + 1) * chunk_size + 1, n_frames + 1)
        if start < end:
            tasks.append((target_dir, start, end, n_replicas))

    # Run blazing-fast CPU extraction
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        results = executor.map(_worker_extract_chunk, tasks)
        for start_step, u_res, v_res in results:
            if u_res is not None:
                idx_start = start_step - 1
                idx_end = idx_start + u_res.shape[0]
                
                U_unbiased_matrix[idx_start:idx_end, :] = u_res
                V_bias_matrix[idx_start:idx_end, :] = v_res

    os.chdir(execution_dir)
    print("\nSerializing identically dimensioned matrices to disk...")
    
    np.savetxt("restraint_matrix.dat", V_bias_matrix, fmt="%.6f", 
               header=f"Shape: {n_frames} frames x {n_replicas} replicas | MELD Restraints V(x) (kJ/mol)")
               
    np.savetxt("unbiased_energy_meld.dat", U_unbiased_matrix, fmt="%.6f", 
               header=f"Shape: {n_frames} frames x {n_replicas} replicas | Unbiased Physics U(x) (kJ/mol)")
    
    print("Extraction complete! You can immediately pipe these into gen_data.py.")

if __name__ == "__main__":
    multiprocessing.set_start_method('spawn')
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--data', help="MELD Data directory", required=True)
    parser.add_argument('-w', '--workers', help="CPU Cores for I/O", type=int, default=multiprocessing.cpu_count())
    args = parser.parse_args()
    
    orchestrate_extraction(args.data, args.workers)