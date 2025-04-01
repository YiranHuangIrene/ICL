import os
import sys
import time
import wandb
import torch
import numpy as np
from tqdm import tqdm
from model import ModelArgs, Transformer
from dataset import ICLDataset
from torch.utils.data import DataLoader
import torch.nn.functional as F
from tqdm import tqdm

WANDB = True

def accuracy(outputs, labels, flip_labels=False):
    predictions = F.softmax(outputs, dim=-1)
    predictions_inds = torch.argmax(predictions, dim=-1)
    labels_inds = torch.argmax(labels, dim=-1)
    if flip_labels:
        labels_inds = (torch.argmax(labels, dim=-1) + 1) % labels.shape[-1]
    return (predictions_inds == labels_inds).float().mean()


def evaluate(model, dataloader, flip_labels=False, device=None):
    model.eval()
    loss_criterion = torch.nn.CrossEntropyLoss()
    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            outputs = model(inputs)
            loss = loss_criterion(outputs, labels)
            acc = accuracy(outputs, labels, flip_labels=flip_labels)
        return loss, acc

def train(model, train_loader, test_loader, test_ic_loader, test_ic2_loader, test_iw_loader, optimizer, device, print_every, ckpt_store_freq, prefix, niters):
    model.to(device)
    loss_criterion = torch.nn.CrossEntropyLoss()
    for n, batch in tqdm(enumerate(train_loader), total=niters):
        model.train()
        inputs, labels = batch
        inputs = inputs.to(device)
        labels = labels.to(device)
        
        optimizer.zero_grad()
        
        # Forward pass
        outputs = model(inputs)
        loss = loss_criterion(outputs, labels)
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        # Save checkpoint
        if n%ckpt_store_freq==0 and n!=0:
            if not os.path.exists(prefix):
                os.makedirs(prefix)
            if not os.path.exists(f"{prefix}/seed_{SEED}"):
                os.makedirs(f"{prefix}/seed_{SEED}")
            torch.save(model.state_dict(), f"{prefix}/seed_{SEED}/ckpt_{n}.pt")
            
        # Evaluate  
        if n%print_every==0:
            loss_test, acc_test = evaluate(model, test_loader, flip_labels=False, device=device)
            loss_ic, acc_ic = evaluate(model, test_ic_loader, flip_labels=False, device=device)
            loss_ic2, acc_ic2 = evaluate(model, test_ic2_loader, flip_labels=True, device=device)
            loss_iw, acc_iw = evaluate(model, test_iw_loader, flip_labels=False, device=device)
            print(f"Iteration {n}: Train loss: {loss:.4f}, Test loss: {loss_test:.4f}, Test acc: {acc_test:.4f}, IC loss: {loss_ic:.4f}, IC acc: {acc_ic:.4f}, IC2 loss: {loss_ic2:.4f}, IC2 acc: {acc_ic2:.4f}, IW loss: {loss_iw:.4f}, IW acc: {acc_iw:.4f}")
            if WANDB:
                wandb.log({"Iteration": n, "Test_Loss": loss_test, "Test_Accuracy": acc_test, "IC_Loss": loss_ic, "IC_Accuracy": acc_ic, "IC2_Loss": loss_ic2, "IC2_Accuracy": acc_ic2, "IW_Loss": loss_iw, "IW_Accuracy": acc_iw})

        
if __name__ == "__main__":
    # Set up CUDA and random seeds
    device = torch.device(f"cuda:{int(sys.argv[16])}" if torch.cuda.is_available() else "cpu")
    SEED = int(sys.argv[14])
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    torch.backends.cudnn.deterministic = True

    # Data Parameters
    K = int(sys.argv[1])  # Number of classes
    N = int(sys.argv[2])  # Number of item-label pairs in the context
    D = int(sys.argv[3])  # Feature dimension
    L = int(sys.argv[4])  # Number of labels
    alpha = float(sys.argv[5])  # Zipf's law exponent
    B = int(sys.argv[6])  # Burstiness
    p_B = float(sys.argv[7])  # Fraction of bursty sequences
    p_C = float(sys.argv[8])  # Fraction of OOD sequences
    eps = float(sys.argv[9])  # Within-class variance
    no_repeats = bool(int(sys.argv[10]))  # Whether repeated items are allowed in the context
    
    S = 10000  # Number of sequences in the test set
    Nmax = 32  # Maximum number of item-label pairs in the context
    P = 1.0/(np.arange(1,K+1)**alpha)
    P /= np.sum(P)  # Normalized power law distribution

    # Model Parameters
    n_heads = int(sys.argv[11])  # Number of attention heads
    n_layers = int(sys.argv[12])  # Number of transformer layers
    rope_theta = int(sys.argv[13])  # Rope base
    rms_norm = bool(int(sys.argv[14])) # Whether to use RMS normalization

    # Training parameters
    niters = 150000  # Number of iterations
    batch_size = 128  # Batch size
    lr = 1e-3  # Learning rate
    weight_decay = 1e-6  # Weight decay
    optimizer = sys.argv[15]
    print_every = 1000  # Print every n iterations
    ckpt_store_freq = 10000 # Store every n iterations

    # Initialize wandb
    prefix = f"./outs/K{K}_N{N}_D{D}_alpha{alpha}_B{B}_pB{p_B}_pC{p_C}_eps{eps}_no_repeats{no_repeats}_rope_theta{rope_theta}_n_heads{n_heads}_n_layers{n_layers}_rms_norm{rms_norm}_optimizer{optimizer}"
    if WANDB:
        wandb.init(project="ICL_torch",
                name=f"run_{SEED}_{prefix.split('/')[-1]}",
                config={
                    "K": K,
                    "N": N,
                    "D": D,
                    "L": L,
                    "S": S,
                    "Nmax": Nmax,
                    "alpha": alpha,
                    "B": B,
                    "pB": p_B,
                    "pC": p_C,
                    "eps": eps,
                    "no_repeats": no_repeats,
                    "rope_theta": rope_theta,
                    "n_heads": n_heads,
                    "n_layers": n_layers,
                    "rms_norm": rms_norm,
                    "niters": niters,
                    "batch_size": batch_size,
                    "lr": lr,
                    "weight_decay": weight_decay,
                    "optimizer": optimizer,
                    "seed": SEED
                })

    # Initialize model
    model_args = ModelArgs(
        dim=D,
        n_layers=n_layers,
        n_heads=n_heads,
        n_labels=L,
        max_position_embeddings=2*N+1,
        rope_theta=rope_theta,
        mlp_bias=True,
        rms_norm=rms_norm,
        norm_eps=1e-5
    )
    model = Transformer(model_args)
    print("Model structure:")
    print(model)

    # Initialize optimizer
    if optimizer == "SGD":
        optimizer = torch.optim.SGD(model.parameters(), lr=lr, weight_decay=weight_decay)
    elif optimizer == "Adam":
        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    elif optimizer == "AdamW":
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    # Initialize datasets
    train_dataset = ICLDataset(
        K=K, L=L, N=N, D=D, S=batch_size, Nmax=Nmax,
        eps=eps, B=B, p_B=p_B, p_C=p_C, P=P, datasize=niters,
        no_repeats=no_repeats, rope=True
    )

    test_dataset = ICLDataset(
        K=K, L=L, N=N, D=D, S=S, Nmax=Nmax,
        eps=eps, B=B, p_B=p_B, p_C=p_C, P=P, datasize=1,
        no_repeats=no_repeats, rope=True
    )
    test_dataset = [sample for sample in test_dataset]

    test_ic_dataset = ICLDataset(
        K=K, L=L, N=N, D=D, S=S, Nmax=Nmax,
        eps=eps, B=B, p_B=1, p_C=1, P=P, datasize=1,        
        no_repeats=no_repeats, rope=True
    )
    test_ic_dataset = [sample for sample in test_ic_dataset]
    test_ic2_dataset = ICLDataset(
        K=K, L=L, N=N, D=D, S=S, Nmax=Nmax,
        eps=eps, B=B, p_B=1, p_C=0, P=P, datasize=1,        
        flip_labels=True, no_repeats=no_repeats, rope=True
    )
    test_ic2_dataset = [sample for sample in test_ic2_dataset]
    test_iw_dataset = ICLDataset(
        K=K, L=L, N=N, D=D, S=S, Nmax=Nmax,
        eps=eps, B=0, p_B=0, p_C=0, P=P, datasize=1,        
        no_repeats=no_repeats, rope=True
    )
    test_iw_dataset = [sample for sample in test_iw_dataset]
    
    # Load the datasets with dataloader
    train_loader = DataLoader(train_dataset, batch_size=None)
    test_loader = DataLoader(test_dataset, batch_size=None)
    test_ic_loader = DataLoader(test_ic_dataset, batch_size=None)
    test_ic2_loader = DataLoader(test_ic2_dataset, batch_size=None)
    test_iw_loader = DataLoader(test_iw_dataset, batch_size=None)

    
    train(model=model, train_loader=train_loader, test_loader=test_loader, test_ic_loader=test_ic_loader, test_ic2_loader=test_ic2_loader, test_iw_loader=test_iw_loader, optimizer=optimizer, device=device, print_every=print_every, ckpt_store_freq=ckpt_store_freq, prefix=prefix, niters=niters)