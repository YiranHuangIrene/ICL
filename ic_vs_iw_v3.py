import numpy as np
import jax
import jax.numpy as jnp
from jax import grad, jit, vmap
from jax import random
from jax.nn import relu
from transformer_v2 import *
from datasets_v2 import *
import sys
import os
import time
from tqdm import tqdm
import wandb
WANDB = True

# os.environ['CUDA_VISIBLE_DEVICES'] = "3"
os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] ='false'
os.environ['XLA_PYTHON_CLIENT_ALLOCATOR']='platform'
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'

ndevices = jax.local_device_count()
print(jax.devices())
device = int(sys.argv[16])

jax.config.update("jax_default_device", jax.devices()[device])
seed = device
# seed = 0
key = random.PRNGKey(seed)
np.random.seed(seed)

np.set_printoptions(precision = 3, suppress = True)

# Data Parameters
K = int(sys.argv[1]) # Number of classes
N = int(sys.argv[2]) # Number of item-label pairs in the context
D = int(sys.argv[3]) # Feature dimension
alpha = float(sys.argv[4]) # Zipf's law exponent
B = int(sys.argv[5]) # Burstiness
p_B = float(sys.argv[6]) # Fraction of bursty sequences
p_C = float(sys.argv[7]) # Fraction of OOD sequences
eps = float(sys.argv[8]) # Within-class variance
no_repeats = bool(int(sys.argv[9])) # Whether repeated items are allowed in the context
L = 32 # Number of labels
S = 10000 # Number of sequencesin the test set
Nmax = 32 # Maximum number of item-label pairs in the context
P = 1.0/(np.arange(1,K+1)**alpha)
P /= np.sum(P) # Normalized power law distribution

# Model Parameters
rope = bool(int(sys.argv[10])) # Whether to use rope
rope_base = int(sys.argv[11]) # Rope base
att_layers = int(sys.argv[12]) # Number of attention layers
num_heads = int(sys.argv[13]) # Number of attention heads
mlp_layers = int(sys.argv[14]) # Number of MLP layers
block = bool(int(sys.argv[15])) # Whether to use block-wise scaling up

# Training parameters
niters = 150000 # Number of iterations
batchsize = 128 # Batch size
lr = 0.01 # Learning rate
w_decay = 1e-6 # Weight decay

nruns = 1 # Number of runs
store= True # Whether to store the model


keys= random.split(key,nruns)
prefix = "./outs/K%d_N%d_L%d_D%d_a_%.2f_B%d_pB_%.2f_pC_%.2f_e%.3f_lr%.3f_nr%d_rope%d_base%d_att_layers%d_num_heads%d_mlp_layers%d_block%d" %(K,N,L,D,alpha,B,p_B,p_C,eps,lr,int(no_repeats),int(rope),rope_base,att_layers,num_heads,mlp_layers,block)
print(prefix)

# initialize wandb and log the parameters
if WANDB:
    wandb.init(project="ICL", name=f"run_{seed}_{prefix.split('/')[-1]}")
    config = wandb.config
    config.K = K
    config.L = L
    config.S = S
    config.N = N
    config.D = D
    config.alpha = alpha
    config.B = B
    config.p_B = p_B
    config.p_C = p_C
    config.batchsize = batchsize
    config.lr = lr
    config.w_decay = w_decay
    config.eps = eps
    config.no_repeats = no_repeats
    config.rope = rope
    config.rope_base = rope_base if rope else 0
    config.att_layers = att_layers
    config.num_heads = num_heads
    config.mlp_layers = mlp_layers
    config.block = block
    
    
for ii in range(nruns):
    run = 2*ii + device

    #Loading datasets

    mus_label, mus_class, labels_class = get_mus_label_class(K,L,D)

    test_inputs, test_labels  = generate_input_seqs(mus_label,mus_class,labels_class,S,N, Nmax,eps = eps, P = P, B = B, p_B = p_B, p_C = p_C, no_repeats = no_repeats, rope = rope)

    test_inputs_ic, test_labels_ic =  generate_input_seqs(mus_label,mus_class,labels_class,S,N, Nmax,eps = eps, P = P, B = B, p_B = 1, p_C = 1, no_repeats = no_repeats, rope = rope)

    test_inputs_ic2, test_labels_ic2 =  generate_input_seqs(mus_label,mus_class,labels_class,S,N, Nmax,eps = eps, P = P, B = B, p_B = 1, p_C = 0, flip_labels = True, no_repeats = no_repeats, rope = rope)

    test_inputs_iw, test_labels_iw =  generate_input_seqs(mus_label,mus_class,labels_class,S,N, Nmax,eps = eps, P = P, B = 0, p_B = 0, p_C = 0, no_repeats = no_repeats, rope = rope)


    k_dim = int(test_inputs.shape[-1]) #1. Also works for 1. Not really. 

    if rope:
        input_dim = D
    else:
        input_dim = 2*Nmax + 1 + D
    params = init_network_params(att_layers, mlp_layers, k_dim ,input_dim, L,keys[ii], scale = 1/np.sqrt(D))

    params_history = []
    targets_num = []

    print_acc = True
    print_loss = False
    print_freq= 1000 #5000
    param_store_freq = 500

    targets_num_iter = np.zeros((niters,K))

    for n in tqdm(range(niters)):
        start = time.time()
        if n%param_store_freq==0:
            params_store = []
            for p in range(len(params)):
                params_store += [[np.array(params[p][q]) for q in range(len(params[p]))]]

            params_history += [params_store]
        
        if n%print_freq == 0:
            if print_loss:
                loss_test = loss(params,test_inputs,test_labels,rope = rope, base = rope_base)
                loss_ic = loss(params,test_inputs_ic,test_labels_ic,rope = rope, base = rope_base )
                loss_ic2 = loss(params,test_inputs_ic2,test_labels_ic2,rope = rope, base = rope_base)
                loss_iw = loss(params,test_inputs_iw,test_labels_iw,rope = rope, base = rope_base)
                print("Run: %d  Iter (x%d): %04d  Test loss: %.3f IC loss: %.3f IC2 loss: %.3f IW loss: %.3f"\
                      %(run+1,print_freq, n/print_freq,loss_test, loss_ic, loss_ic2, loss_iw))
                if WANDB:
                    wandb.log({"Iteration": n, "Test_Loss": loss_test, "Test_Loss/IC": loss_ic, "Test_Loss/IC2": loss_ic2, "Test_Loss/IW": loss_iw})
            if print_acc:
                loss_test = loss(params,test_inputs,test_labels,rope = rope, base = rope_base)
                acc_test = accuracy(params,test_inputs,test_labels,rope = rope, base = rope_base)
                acc_ic = accuracy(params,test_inputs_ic,test_labels_ic,rope = rope, base = rope_base)
                acc_ic2 = accuracy(params,test_inputs_ic2,test_labels_ic2, flip_labels = True, rope = rope, base = rope_base)
                acc_iw = accuracy(params,test_inputs_iw,test_labels_iw,rope = rope, base = rope_base)
                print("Run: %d  Iter (x%d): %04d Test loss: %.3f Test acc: %.3f IC acc: %.3f IC2 acc: %.3f IW acc: %.3f"\
                      %(run+1,print_freq,n/print_freq,loss_test, acc_test, acc_ic, acc_ic2, acc_iw))
                if WANDB:
                    wandb.log({"Iteration": n, "Test_Loss": loss_test, "Test_Accuracy": acc_test, "Test_Accuracy/IC": acc_ic, "Test_Accuracy/IC2": acc_ic2, "Test_Accuracy/IW": acc_iw})

        
        end1 = time.time()
        inputs_batch, labels_batch, target_classes = generate_input_seqs(mus_label,mus_class,labels_class,batchsize,N, Nmax,eps = eps, P = P, B = B, p_B = p_B, p_C = p_C, output_target_labels = True, no_repeats = no_repeats, test=False, rope = rope)
        end2 = time.time()
        params = update(params,inputs_batch,labels_batch,  lr = lr, rope = rope, base = rope_base)
        end3 = time.time()

        #for k in range(K):
        #    targets_num_iter[n, k] = np.sum(target_classes == k)

        end4 = time.time()

        #print(end1  - start, end2 - end1, end3 - end2, end4 - end3)


    if store:        
        if not os.path.exists(prefix):
            os.makedirs(prefix)

        if not os.path.exists(prefix + "/iter%d"%run):
            os.makedirs(prefix + "/iter%d"%run)

        np.savez(prefix + "/iter%d"%run + "/labels_classes",mus_label, mus_class, labels_class)

        np.save(prefix + "/iter%d"%run + "/targets_num_iter", targets_num_iter)
           
        Q = {}
        K = {}
        V = {}    
        for h in range(len(params_history)):
            for l in range(att_layers):
                Q[l] = params_history[h][l][0]
                K[l] = params_history[h][l][1]
                V[l] = params_history[h][l][2]

            
            w1 = params_history[h][att_layers][0]
            b1 = params_history[h][att_layers][1]
            
            w2 = params_history[h][att_layers+1][0]
            b2 = params_history[h][att_layers+1][1]
            
            w3 = params_history[h][att_layers+2][0]
            b3 = params_history[h][att_layers+2][1]
            
            s = np.array(params_history[h][-1][0])
            if att_layers == 4:
                q1,k1,v1,q2,k2,v2,q3,k3,v3,q4,k4,v4 = Q[0],K[0],V[0],Q[1],K[1],V[1],Q[2],K[2],V[2],Q[3],K[3],V[3]
                np.savez(prefix + "/iter%d"%run + "/params%08d"%(h*param_store_freq ),q1,k1,v1,q2,k2,v2,q3,k3,v3,q4,k4,v4,w1,b1,w2,b2,w3,b3,s)
            elif att_layers == 3:
                q1,k1,v1,q2,k2,v2,q3,k3,v3 = Q[0],K[0],V[0],Q[1],K[1],V[1],Q[2],K[2],V[2]
                np.savez(prefix + "/iter%d"%run + "/params%08d"%(h*param_store_freq ),q1,k1,v1,q2,k2,v2,q3,k3,v3,w1,b1,w2,b2,w3,b3,s)
            elif att_layers == 2:
                q1,k1,v1,q2,k2,v2 = Q[0],K[0],V[0],Q[1],K[1],V[1]
                np.savez(prefix + "/iter%d"%run + "/params%08d"%(h*param_store_freq ),q1,k1,v1,q2,k2,v2,w1,b1,w2,b2,w3,b3,s)
            elif att_layers == 6:
                q1,k1,v1,q2,k2,v2,q3,k3,v3,q4,k4,v4,q5,k5,v5,q6,k6,v6 = Q[0],K[0],V[0],Q[1],K[1],V[1],Q[2],K[2],V[2],Q[3],K[3],V[3],Q[4],K[4],V[4],Q[5],K[5],V[5]
                np.savez(prefix + "/iter%d"%run + "/params%08d"%(h*param_store_freq ),q1,k1,v1,q2,k2,v2,q3,k3,v3,q4,k4,v4,q5,k5,v5,q6,k6,v6,w1,b1,w2,b2,w3,b3,s)
            elif att_layers == 5:
                q1,k1,v1,q2,k2,v2,q3,k3,v3,q4,k4,v4,q5,k5,v5 = Q[0],K[0],V[0],Q[1],K[1],V[1],Q[2],K[2],V[2],Q[3],K[3],V[3],Q[4],K[4],V[4]
                np.savez(prefix + "/iter%d"%run + "/params%08d"%(h*param_store_freq ),q1,k1,v1,q2,k2,v2,q3,k3,v3,q4,k4,v4,q5,k5,v5,w1,b1,w2,b2,w3,b3,s)

        


