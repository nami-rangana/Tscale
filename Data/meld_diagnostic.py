from meld import vault
import os

def run_diagnostic():
    target_dir = "/orange/alberto.perezant/imesh.ranaweera/test_softwares/test_grappa/systems/1POU"
    print(f"Shifting working directory to {target_dir}...")
    os.chdir(target_dir)
    
    store = vault.DataStore.load_data_store()
    store.initialize(mode='r')
    
    print("\n--- Scanning DataStore Methods ---")
    # Looking for methods like load_energies, load_energy_matrix, etc.
    store_methods = [attr for attr in dir(store) if not attr.startswith('_')]
    print([m for m in store_methods if 'energy' in m or 'restraint' in m or 'load' in m])
    
    print("\n--- Scanning SystemState Attributes ---")
    states = store.load_states(1)
    state = states[0]
    print([attr for attr in dir(state) if not attr.startswith('_')])

if __name__ == "__main__":
    run_diagnostic()