import os
import argparse
import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch_geometric.loader import DataLoader
import torch.multiprocessing as mp

from cgraph import ProteinPhaseSpaceDataset
from model import FreeEnergyEstimator

def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    dist.init_process_group("nccl", rank=rank, world_size=world_size)

def cleanup():
    dist.destroy_process_group()

def train_worker(rank, world_size, args):
    setup(rank, world_size)
    torch.cuda.set_device(rank)
    
    # 1. Initialize Dataset
    dataset = ProteinPhaseSpaceDataset(
        traj_files=args.traj,
        top_file=args.top,
        weights_file=args.weights,
        embed_file=args.embed,
        graph_type=args.graph_type,
        cutoff_angstroms=args.cutoff
    )
    
    # 2. Distributed Sampler
    sampler = torch.utils.data.distributed.DistributedSampler(
        dataset, num_replicas=world_size, rank=rank, shuffle=True
    )
    loader = DataLoader(dataset, batch_size=args.batch_size, sampler=sampler, num_workers=4, pin_memory=True)

    # 3. Initialize Model (Assuming standard 1280-dim sequence embeddings)
    model = FreeEnergyEstimator(embed_dim=1280, hidden_dim=256, n_layers=4).to(rank)
    model = DDP(model, device_ids=[rank])
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-5)
    criterion = nn.MSELoss() 
    
    # 4. Training Loop
    if rank == 0:
        print(f"Starting training on {world_size} GPUs | Graph: {args.graph_type}")
        
    for epoch in range(args.epochs):
        sampler.set_epoch(epoch)
        model.train()
        total_loss = 0
        
        for batch in loader:
            batch = batch.to(rank)
            optimizer.zero_grad()
            
            preds = model(batch)
            loss = criterion(preds, batch.y)
            loss.backward()
            
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            total_loss += loss.item()
            
        if rank == 0:
            avg_loss = total_loss / len(loader)
            print(f"Epoch {epoch+1:>3}/{args.epochs} | MSE Loss: {avg_loss:.4f}")

    if rank == 0:
        torch.save(model.module.state_dict(), f"se3_free_energy_{args.graph_type}.pt")
        print("Training complete. Model saved.")

    cleanup()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # nargs='+' allows bash wildcards (*.dcd) to be passed as a list of files
    parser.add_argument('--traj', nargs='+', required=True, help="List of trajectory files (.dcd, .xtc, .nc)")
    parser.add_argument('--top', required=True, help="Topology file (.prmtop, .pdb)")
    parser.add_argument('--weights', required=True, help="mbar_weights.dat")
    parser.add_argument('--embed', required=True, help="Precomputed sequence embeddings (.npy)")
    parser.add_argument('--graph_type', choices=['cutoff', 'fully_connected'], default='cutoff')
    parser.add_argument('--cutoff', type=float, default=8.0, help="Distance cutoff in Angstroms")
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--epochs', type=int, default=50)
    args = parser.parse_args()

    # Crucial: Sort files alphabetically to ensure they match the MBAR weight frame order exactly!
    args.traj.sort()

    world_size = torch.cuda.device_count()
    if world_size < 1:
        raise RuntimeError("No CUDA GPUs detected.")
        
    mp.spawn(train_worker, args=(world_size, args), nprocs=world_size, join=True)