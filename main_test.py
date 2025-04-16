import os
import sys
import time
import wandb
import torch
import numpy as np
from model import ModelArgs, Transformer
from dataset import ICLDataset, get_mus_label_class, generate_input_seqs
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


def evaluate(model, data, flip_labels=False, device=None):
    model.eval()
    loss_criterion = torch.nn.CrossEntropyLoss()
    with torch.no_grad():
        inputs, labels = data
        inputs = inputs.to(device)
        labels = labels.to(device)
        outputs = model(inputs)
        loss = loss_criterion(outputs, labels)
        acc = accuracy(outputs, labels, flip_labels=flip_labels)
        return loss, acc

def train(model,batch_size,train_data,test_data, test_ic_data, test_ic2_data, test_iw_data, optimizer, device, print_every, ckpt_store_freq, prefix, niters, n_epochs):
    model.to(device)
    loss_criterion = torch.nn.CrossEntropyLoss()
    # for n, batch in tqdm(enumerate(train_loader), total=niters):
    global_iter = 0
    for epoch in range(n_epochs):
        print(f"Epoch {epoch}")
        for n in tqdm(range(niters)):
            inputs, labels = train_data[0][batch_size*n:batch_size*(n+1)], train_data[1][batch_size*n:batch_size*(n+1)]
            model.train()
            inputs = inputs.to(device)
            labels = labels.to(device)
            
            optimizer.zero_grad()
            
            # Forward pass
            outputs = model(inputs)
            loss = loss_criterion(outputs, labels)
            acc = accuracy(outputs, labels)
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            # Save checkpoint
            if global_iter%ckpt_store_freq==0 and global_iter!=0:
                if not os.path.exists(prefix):
                    os.makedirs(prefix)
                if not os.path.exists(f"{prefix}/seed_{SEED}"):
                    os.makedirs(f"{prefix}/seed_{SEED}")
                torch.save(model.state_dict(), f"{prefix}/seed_{SEED}/ckpt_{global_iter}.pt")
                
            # Evaluate  
            if global_iter%print_every==0:
                loss_test, acc_test = evaluate(model, test_data, flip_labels=False, device=device)
                loss_ic, acc_ic = evaluate(model, test_ic_data, flip_labels=False, device=device)
                loss_ic2, acc_ic2 = evaluate(model, test_ic2_data, flip_labels=True, device=device)
                loss_iw, acc_iw = evaluate(model, test_iw_data, flip_labels=False, device=device)
                print(f"Epoch {epoch}, Iteration {n}: Train loss: {loss:.4f}, Train acc: {acc:.4f}, Test loss: {loss_test:.4f}, Test acc: {acc_test:.4f}, IC loss: {loss_ic:.4f}, IC acc: {acc_ic:.4f}, IC2 loss: {loss_ic2:.4f}, IC2 acc: {acc_ic2:.4f}, IW loss: {loss_iw:.4f}, IW acc: {acc_iw:.4f}")
                if WANDB:
                    wandb.log({"Epoch": epoch, "Iteration": n, "global_iter": global_iter, "Train_Loss": loss, "Train_Accuracy": acc, "Test_Loss": loss_test, "Test_Accuracy": acc_test, "IC_Loss": loss_ic, "IC_Accuracy": acc_ic, "IC2_Loss": loss_ic2, "IC2_Accuracy": acc_ic2, "IW_Loss": loss_iw, "IW_Accuracy": acc_iw})
            global_iter += 1

        
if __name__ == "__main__":
    # Set up CUDA and random seeds
    device = torch.device(f"cuda:{int(sys.argv[18])}" if torch.cuda.is_available() else "cpu")
    SEED = int(sys.argv[18])
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
    rope = bool(int(sys.argv[13]))  # Whether to use RoPE
    rope_theta = int(sys.argv[14])  # Rope base
    rms_norm = bool(int(sys.argv[15])) # Whether to use RMS normalization

    # Training parameters
    niters = 30000  # Number of iterations
    n_epochs = 10  # Number of epochs
    batch_size = int(sys.argv[16])
    lr = 1e-3  # Learning rate
    weight_decay = 1e-6  # Weight decay
    optimizer = sys.argv[17]
    print_every = 1000  # Print every n iterations
    ckpt_store_freq = 10000 # Store every n iterations

    if rope:
        input_dim = D
    else:
        input_dim = 2*Nmax + 1 + D
        
   # Initialize wandb
    prefix = f"./outs_torch/K{K}_N{N}_D{D}_alpha{alpha}_B{B}_pB{p_B}_pC{p_C}_eps{eps}_no_repeats{no_repeats}_rope_theta{rope_theta}_n_heads{n_heads}_n_layers{n_layers}_rms_norm{rms_norm}_optimizer{optimizer}_niters{niters}_n_epochs{n_epochs}"
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
                    "rope": rope,
                    "rope_theta": rope_theta,
                    "n_heads": n_heads,
                    "n_layers": n_layers,
                    "rms_norm": rms_norm,
                    "niters": niters,
                    "batch_size": batch_size,
                    "lr": lr,
                    "weight_decay": weight_decay,
                    "optimizer": optimizer,
                    "seed": SEED,
                    "n_epochs": n_epochs
                })

    # Initialize model
    model_args = ModelArgs(
        dim=input_dim,
        n_layers=n_layers,
        n_heads=n_heads,
        n_labels=L,
        max_position_embeddings=2*N+1,
        rope_theta=rope_theta,
        mlp_bias=True,
        rms_norm=rms_norm,
        rope=rope,
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
    mus_label, mus_class, labels_class = get_mus_label_class(K, L, D)
    print("Generating training data...")
    train_data = generate_input_seqs(mus_label,mus_class,labels_class,N,batch_size*niters, Nmax,eps = eps, B = B, p_B = p_B, P = P, p_C = p_C, no_repeats = no_repeats)
    print("Generating test data...")
    test_data  = generate_input_seqs(mus_label,mus_class,labels_class,N,S, Nmax,eps = eps, P = P, B = B, p_B = p_B, p_C = p_C, no_repeats = no_repeats)
    test_ic_data = generate_input_seqs(mus_label,mus_class,labels_class,N,S, Nmax,eps = eps, P = P, B = B, p_B = 1, p_C = 1, no_repeats = no_repeats)
    test_ic2_data = generate_input_seqs(mus_label,mus_class,labels_class,N,S, Nmax,eps = eps, P = P, B = B, p_B = 1, p_C = 0, flip_labels = True, no_repeats = no_repeats)
    test_iw_data = generate_input_seqs(mus_label,mus_class,labels_class,N,S, Nmax,eps = eps, P = P, B = 0, p_B = 0, p_C = 0, no_repeats = no_repeats)
    print("Training model...")
    train(model=model, batch_size=batch_size,train_data=train_data,test_data=test_data, test_ic_data=test_ic_data, test_ic2_data=test_ic2_data, test_iw_data=test_iw_data, optimizer=optimizer, device=device, print_every=print_every, ckpt_store_freq=ckpt_store_freq, prefix=prefix, niters=niters, n_epochs=n_epochs)
    