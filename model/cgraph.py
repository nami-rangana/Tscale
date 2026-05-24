import torch
import numpy as np
import mdtraj as md
from torch_geometric.data import Data, Dataset
from torch_geometric.nn import radius_graph

class ProteinPhaseSpaceDataset(Dataset):
    def __init__(self, traj_files, top_file, weights_file, embed_file, graph_type='cutoff', cutoff_angstroms=8.0):
        super().__init__()
        self.graph_type = graph_type.lower()
        self.cutoff_nm = cutoff_angstroms / 10.0 
        
        # 1. Load Topology and identify C-alpha indices
        print(f"Loading topology from {top_file}...")
        top = md.load_topology(top_file)
        self.ca_indices = top.select("name CA")
        self.n_nodes = len(self.ca_indices)
        
        # 2. Optimized Iterative Loading
        # Loading all files at once will cause OOM. We load, slice to CA, and append.
        print(f"Loading {len(traj_files)} trajectory files...")
        sliced_trajs = []
        for t_file in traj_files:
            chunk = md.load(t_file, top=top_file)
            sliced_chunk = chunk.atom_slice(self.ca_indices)
            sliced_trajs.append(sliced_chunk)
            
        self.traj = md.join(sliced_trajs)
        print(f"Total frames loaded: {self.traj.n_frames}")
        
        # 3. Handle Sequence Embeddings
        print(f"Loading sequence embeddings from {embed_file}...")
        if embed_file.endswith('.npz'):
            print("\n[WARNING] You passed an .npz file to --embed! The model expects Sequence Embeddings (e.g. ESM-2).")
            print("[WARNING] Generating dummy random embeddings so you can test the pipeline...\n")
            self.embeddings = torch.randn((self.n_nodes, 1280), dtype=torch.float32)
        else:
            self.embeddings = torch.tensor(np.load(embed_file), dtype=torch.float32)
            assert self.embeddings.shape[0] == self.n_nodes, "Embedding length must match C-alpha count!"
        
        # 4. Load MBAR Weights and convert to Free Energy
        print(f"Loading MBAR weights from {weights_file}...")
        raw_weights = np.loadtxt(weights_file)
        
        if len(raw_weights) != self.traj.n_frames:
            raise ValueError(f"Weight count ({len(raw_weights)}) does not match frame count ({self.traj.n_frames})! Check file sorting.")
            
        self.target_g = -np.log(raw_weights + 1e-12)
        self.target_g = torch.tensor(self.target_g, dtype=torch.float32)

        # Pre-build Fully Connected edges
        if self.graph_type == 'fully_connected':
            print("Pre-computing fully connected edge indices...")
            nodes = torch.arange(self.n_nodes)
            row = nodes.repeat_interleave(self.n_nodes)
            col = nodes.repeat(self.n_nodes)
            mask = row != col
            self.fc_edge_index = torch.stack([row[mask], col[mask]], dim=0)

    def len(self):
        return self.traj.n_frames

    def get(self, idx):
        pos = torch.tensor(self.traj.xyz[idx] * 10.0, dtype=torch.float32)
        
        if self.graph_type == 'cutoff':
            edge_index = radius_graph(pos, r=self.cutoff_nm * 10.0, loop=False)
        elif self.graph_type == 'fully_connected':
            edge_index = self.fc_edge_index

        data = Data(
            x=self.embeddings,        
            pos=pos,                  
            edge_index=edge_index,    
            y=self.target_g[idx]      
        )
        return data