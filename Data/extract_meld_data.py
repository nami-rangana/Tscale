import numpy as np
import os
import argparse
from concurrent.futures import ProcessPoolExecutor
import multiprocessing
import math

# =============================================================================
# WORKER: I/O Bound HDF5 Extraction (CPU)
# =============================================================================
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
    
    # Pre-allocate arrays
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
            
            # 1. Physics & Restraints
            if groups is not None and len(groups) >= 7:
                U_chunk[local_idx, k_rep] = float(groups[0])
                V_chunk[local_idx, k_rep] = float(np.sum(groups[1:]))
            else:
                U_chunk[local_idx, k_rep] = np.nan
                V_chunk[local_idx, k_rep] = np.nan
                
        local_idx += 1
        
    return start_step, U_chunk[:local_idx, :], V_chunk[:local_idx, :]

# =============================================================================
# ORCHESTRATOR
# =============================================================================
def orchestrate_extraction(data_dir, num_workers):
    import sys
    import types
    
    execution_dir = os.getcwd()
    abs_path = os.path.abspath(data_dir)
    target_dir = os.path.dirname(abs_path) if os.path.basename(abs_path) == "Data" else abs_path
        
    print(f"Mounting MELD vault at {target_dir}...\n")
    os.chdir(target_dir)
    
    # --- PICKLE CROSS-ENVIRONMENT SPOOF HACK ---
    class SpoofModule(types.ModuleType):
        def __getattr__(self, name):
            class SpoofClass:
                def __init__(self, *args, **kwargs): pass
                def __setstate__(self, state): 
                    if isinstance(state, dict): self.__dict__.update(state)
            SpoofClass.__name__ = name
            SpoofClass.__module__ = self.__name__
            return SpoofClass
    sys.modules['meld.system.density'] = SpoofModule('meld.system.density')
    # -------------------------------------------
    
    from meld import vault 
    store = vault.DataStore.load_data_store()
    store.initialize('r')
    
    n_replicas = store.n_replicas
    
    if hasattr(store, 'max_safe_frame') and store.max_safe_frame > 0:
        n_frames = store.max_safe_frame
    elif hasattr(store, 'max_saved_step') and store.max_saved_step > 0:
        n_frames = store.max_saved_step
    else:
        n_frames = 20000 
        
    system = store.load_system()
    temp_scaler = system.temperature_scaler
    
    states_frame_1 = store.load_states(1)
    
    # 1. THE MASTER ALPHAS
    alphas = [state.alpha for state in states_frame_1]
    
    # 2. THE TEMPERATURE SCALING
    try:
        from openmm import unit
    except ImportError:
        import simtk.unit as unit
        
    temperatures = np.zeros(n_replicas, dtype=np.float64)
    for i, alpha in enumerate(alphas):
        t_val = temp_scaler(alpha)
        if hasattr(t_val, 'value_in_unit'):
            temperatures[i] = float(t_val.value_in_unit(unit.kelvin))
        else:
            temperatures[i] = float(t_val)
            
    store = None # Release the HDF5 file handle

    # 3. THE RESTRAINT SCALING (Extracting your 0.4 and 1.0)
    master_scaler = None
    extracted_min = None
    extracted_max = None
    
    try:
        restraints_to_check = []
        if hasattr(system.restraints, '_selective_collections'):
            for coll in system.restraints._selective_collections:
                for group in coll.groups:
                    restraints_to_check.extend(group.restraints)
                    
        for rest in restraints_to_check:
            scaler_obj = getattr(rest, 'scaler', None)
            if scaler_obj and hasattr(scaler_obj, 'alpha_min') and hasattr(scaler_obj, 'alpha_max'):
                extracted_min = float(scaler_obj.alpha_min)
                extracted_max = float(scaler_obj.alpha_max)
                fac = float(getattr(scaler_obj, 'factor', 1.0))
                
                def make_scaler(min_val, max_val, f_val):
                    def calc(a):
                        if a <= min_val: return 0.0
                        if a >= max_val: return 1.0
                        return ((a - min_val) / (max_val - min_val)) ** f_val
                    return calc
                    
                master_scaler = make_scaler(extracted_min, extracted_max, fac)
                break
                
    except Exception:
        pass
        
    if master_scaler is None:
        extracted_min, extracted_max, fac = 0.4, 1.0, 4.0
        def make_scaler(min_val=0.4, max_val=1.0, f_val=4.0):
            def calc(a):
                if a <= min_val: return 0.0
                if a >= max_val: return 1.0
                return ((a - min_val) / (max_val - min_val)) ** f_val
            return calc
        master_scaler = make_scaler()
        
    restraint_weights = np.zeros(n_replicas, dtype=np.float64)
    for i, alpha in enumerate(alphas):
        restraint_weights[i] = float(master_scaler(alpha))
        
    # --- CLEAN, FORMATTED OUTPUT ---
    print("================ THERMODYNAMIC MAPPING ================")
    print(f"1. Master Alphas:   [{', '.join([f'{a:.4f}' for a in alphas])}]")
    print(f"2. Temperatures:    [{', '.join([f'{t:.1f}' for t in temperatures])}]")
    print(f"\nRestraint Scaler Auto-Detected: Active Window = {extracted_min} to {extracted_max}")
    print(f"3. Scaled Restraints:[{', '.join([f'{w:.4f}' for w in restraint_weights])}]")
    print("=======================================================\n")
    
    print(f"Detected: {n_replicas} Replicas across {n_frames} Frames.")
    print(f"Extracting U(x) and V(x) using {num_workers} CPU cores...")
    
    U_unbiased_matrix = np.zeros((n_frames, n_replicas), dtype=np.float64)
    V_bias_matrix = np.zeros((n_frames, n_replicas), dtype=np.float64)
    
    chunk_size = max(1, math.ceil(n_frames / num_workers))
    tasks = []
    for i in range(num_workers):
        start = (i * chunk_size) + 1
        end = min((i + 1) * chunk_size + 1, n_frames + 1)
        if start < end:
            tasks.append((target_dir, start, end, n_replicas))

    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        results = executor.map(_worker_extract_chunk, tasks)
        for start_step, u_res, v_res in results:
            if u_res is not None:
                idx_start = start_step - 1
                idx_end = idx_start + u_res.shape[0]
                
                U_unbiased_matrix[idx_start:idx_end, :] = u_res
                V_bias_matrix[idx_start:idx_end, :] = v_res

    os.chdir(execution_dir)
    print("\nSerializing to compressed binary NumPy format (.npz)...")
    
    np.savez_compressed('meld_thermodynamics.npz', 
                        U=U_unbiased_matrix, 
                        V=V_bias_matrix, 
                        T=temperatures,
                        W=restraint_weights)
                        
    print("Extraction complete! Saved to meld_thermodynamics.npz")

if __name__ == "__main__":
    multiprocessing.set_start_method('spawn')
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--data', help="MELD Data directory", required=True)
    parser.add_argument('-w', '--workers', help="CPU Cores for I/O", type=int, default=multiprocessing.cpu_count())
    args = parser.parse_args()
    
    orchestrate_extraction(args.data, args.workers)