import numpy as np
import mdtraj as md
import openmm as omm
import openmm.app as app
import openmm.unit as unit
import pymbar

def parse_temperatures(temp_file):
    """Parses temperatures.dat and returns a list of temperatures."""
    temps = []
    with open(temp_file, 'r') as f:
        for line in f:
            parts = line.split()
            if len(parts) >= 2:
                temps.append(float(parts[1]))
    return np.array(temps)

def compute_physical_energies(topology_file, traj_files):
    """
    Evaluates the physical potential energy for every frame in all trajectories.
    Returns the combined energy array and the number of frames per trajectory.
    """
    print("Setting up OpenMM system for energy evaluation...")
    pdb = app.PDBFile(topology_file)
    forcefield = app.ForceField('amber14-all.xml', 'amber14/tip3pfb.xml') # Adjust FF as needed
    
    system = forcefield.createSystem(pdb.topology, nonbondedMethod=app.NoCutoff, constraints=app.HBonds)
    integrator = omm.VerletIntegrator(1.0 * unit.femtoseconds)
    
    # Robust Platform Fallback Chain
    try:
        platform = omm.Platform.getPlatformByName('CUDA')
        context = omm.Context(system, integrator, platform)
        print("Successfully initialized CUDA platform.")
    except Exception as e1:
        print(f"CUDA failed ({e1}). Attempting OpenCL fallback...")
        try:
            platform = omm.Platform.getPlatformByName('OpenCL')
            context = omm.Context(system, integrator, platform)
            print("Successfully initialized OpenCL platform.")
        except Exception as e2:
            print(f"OpenCL failed ({e2}). Falling back to CPU platform. This will be slower.")
            platform = omm.Platform.getPlatformByName('CPU')
            context = omm.Context(system, integrator, platform)

    all_energies = []
    N_k = [] # Number of frames in each replica
    
    for i, traj_file in enumerate(traj_files):
        print(f"Processing {traj_file} ({i+1}/{len(traj_files)})...")
        traj = md.load(traj_file, top=topology_file)
        N_k.append(traj.n_frames)
        
        for j in range(traj.n_frames):
            context.setPositions(traj.xyz[j])
            state = context.getState(getEnergy=True)
            # Get energy in kJ/mol
            energy = state.getPotentialEnergy().value_in_unit(unit.kilojoules_per_mole)
            all_energies.append(energy)
            
    return np.array(all_energies), np.array(N_k)

def run_mbar(energies, N_k, temperatures, T_ref):
    """
    Uses PyMBAR to combine all data and calculate weights at the reference temperature.
    """
    print("\nConstructing MBAR reduced potential matrix (u_kn)...")
    K = len(temperatures)          # Number of states (replicas)
    N_total = np.sum(N_k)          # Total number of frames across all replicas
    
    kb = 0.00831446261815324       # Boltzmann constant in kJ/(mol*K)
    beta_k = 1.0 / (kb * temperatures)
    
    # u_kn is a K x N_total matrix. 
    u_kn = np.zeros([K, N_total])
    for k in range(K):
        u_kn[k, :] = energies * beta_k[k]

    print("Running MBAR (this may take a few minutes)...")
    # Initialize MBAR (PyMBAR 4.x will use its JAX backend here)
    mbar = pymbar.MBAR(u_kn, N_k)
    
    print(f"Extracting weights for target temperature: {T_ref} K...")
    
    # Find the index of the state that matches our target temperature
    # We use np.isclose to safely compare floats
    target_indices = np.where(np.isclose(temperatures, T_ref))[0]
    
    if len(target_indices) == 0:
        raise ValueError(f"Target temperature {T_ref} K not found in temperatures.dat!")
        
    # We just need the index of the first replica at 300K (State 0)
    k_target = target_indices[0] 
    
    # mbar.W_nk is an N_total x K matrix of weights.
    # We extract the column for our target state.
    # Wrapped in np.array() to safely convert from a JAX array to standard NumPy
    weights = np.array(mbar.W_nk[:, k_target])
    
    return weights

# ==========================================
# Main Execution Block
# ==========================================
if __name__ == "__main__":
    topology_path = "topology.pdb"
    temps_path = "temperatures.dat"
    T_reference = 300.0 
    
    # 1. Parse temperatures
    temperatures = parse_temperatures(temps_path)
    
    # Generate list of trajectory files in the correct order (1 to 30)
    traj_files = [f"trajectory_{i}.xtc" for i in range(1, len(temperatures) + 1)]
    
    # 2. Evaluate energies
    combined_energies, frame_counts = compute_physical_energies(topology_path, traj_files)
    
    # 3. Get MBAR weights
    final_weights = run_mbar(combined_energies, frame_counts, temperatures, T_reference)
    
    # 4. Save results to map structures to weights
    np.savetxt("mbar_weights.dat", final_weights, header=f"Normalized MBAR weights at {T_reference} K")
    np.savetxt("combined_energies.dat", combined_energies, header="Physical Potential Energies (kJ/mol)")
    
    print("\nDone! To map a specific structure to its weight:")
    print("Frame index 'i' in the concatenated trajectory corresponds to weight 'final_weights[i]'.")