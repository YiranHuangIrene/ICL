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
device = int(sys.argv[18])

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
block = int(sys.argv[15]) # Whether to use block-wise scaling up
act = sys.argv[16] # Activation function
rms_norm = bool(int(sys.argv[17]))

# Training parameters
niters = 150000 # Number of iterations
batchsize = 128 # Batch size
lr = 0.01 # Learning rate
w_decay = 1e-6 # Weight decay

nruns = 1 # Number of runs
store= True # Whether to store the model


keys= random.split(key,nruns)
prefix = "./outs/K%d_N%d_L%d_D%d_a_%.2f_B%d_pB_%.2f_pC_%.2f_e%.3f_lr%.3f_nr%d_rope%d_base%d_att_layers%d_num_heads%d_mlp_layers%d_block%d_act%s_rms_norm%d_new_residual" %(K,N,L,D,alpha,B,p_B,p_C,eps,lr,int(no_repeats),int(rope),rope_base,att_layers,num_heads,mlp_layers,block,act,rms_norm)
print(prefix)

# initialize wandb and log the parameters
if WANDB:
    wandb.init(project="ICL", name=f"run_{seed}_{prefix.split('/')[-1]}")
    config = wandb.config
    config.seed = seed
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
    config.act = act
    config.rms_norm = rms_norm
    
    
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
    params = init_network_params(att_layers, mlp_layers, k_dim ,input_dim, L,rms_norm, keys[ii], scale = 1/np.sqrt(D))

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
                loss_test = loss(params,test_inputs,test_labels,rope = rope, base = rope_base, act = act, rms_norm = rms_norm)
                loss_ic = loss(params,test_inputs_ic,test_labels_ic,rope = rope, base = rope_base, act = act, rms_norm = rms_norm)
                loss_ic2 = loss(params,test_inputs_ic2,test_labels_ic2,rope = rope, base = rope_base, act = act, rms_norm = rms_norm)
                loss_iw = loss(params,test_inputs_iw,test_labels_iw,rope = rope, base = rope_base, act = act, rms_norm = rms_norm)
                print("Run: %d  Iter (x%d): %04d  Test loss: %.3f IC loss: %.3f IC2 loss: %.3f IW loss: %.3f"\
                      %(run+1,print_freq, n/print_freq,loss_test, loss_ic, loss_ic2, loss_iw))
                if WANDB:
                    wandb.log({"Iteration": n, "Test_Loss": loss_test, "Test_Loss/IC": loss_ic, "Test_Loss/IC2": loss_ic2, "Test_Loss/IW": loss_iw})
            if print_acc:
                loss_test = loss(params,test_inputs,test_labels,rope = rope, base = rope_base, act = act, rms_norm = rms_norm)
                acc_test = accuracy(params,test_inputs,test_labels,rope = rope, base = rope_base, act = act, rms_norm = rms_norm)
                acc_ic = accuracy(params,test_inputs_ic,test_labels_ic,rope = rope, base = rope_base, act = act, rms_norm = rms_norm)
                acc_ic2 = accuracy(params,test_inputs_ic2,test_labels_ic2, flip_labels = True, rope = rope, base = rope_base, act = act, rms_norm = rms_norm)
                acc_iw = accuracy(params,test_inputs_iw,test_labels_iw,rope = rope, base = rope_base, act = act, rms_norm = rms_norm)
                print("Run: %d  Iter (x%d): %04d Test loss: %.3f Test acc: %.3f IC acc: %.3f IC2 acc: %.3f IW acc: %.3f"\
                      %(run+1,print_freq,n/print_freq,loss_test, acc_test, acc_ic, acc_ic2, acc_iw))
                if WANDB:
                    wandb.log({"Iteration": n, "Test_Loss": loss_test, "Test_Accuracy": acc_test, "Test_Accuracy/IC": acc_ic, "Test_Accuracy/IC2": acc_ic2, "Test_Accuracy/IW": acc_iw})

        
        end1 = time.time()
        inputs_batch, labels_batch, target_classes = generate_input_seqs(mus_label,mus_class,labels_class,batchsize,N, Nmax,eps = eps, P = P, B = B, p_B = p_B, p_C = p_C, output_target_labels = True, no_repeats = no_repeats, rope = rope)
        end2 = time.time()
        loss_train, params = update(params,inputs_batch,labels_batch,  lr = lr, rope = rope, base = rope_base, act = act, rms_norm = rms_norm)
        end3 = time.time()
        if n%print_freq == 0:
            if WANDB:
                wandb.log({"Iteration": n, "Train_Loss": loss_train})

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
        
        def save_model_parameters(params_history, att_layers, rms_norm, prefix, run, param_store_freq):
            for h in range(len(params_history)):
                # Initialize dictionaries to store parameters
                Q, K, V = {}, {}, {}
                rms_before, rms_after = {}, {}
                
                # Extract attention layer parameters
                for l in range(att_layers):
                    if rms_norm:
                        #before
                        # rms_before[l] = params_history[h][l][0]
                        # Q[l] = params_history[h][l][1][0]
                        # K[l] = params_history[h][l][1][1]
                        # V[l] = params_history[h][l][1][2]
                        # after
                        # Q[l] = params_history[h][l][0][0]
                        # K[l] = params_history[h][l][0][1]
                        # V[l] = params_history[h][l][0][2]
                        # rms_after[l] = params_history[h][l][1]
                        # all
                        rms_before[l] = params_history[h][l][0]
                        Q[l] = params_history[h][l][1][0]
                        K[l] = params_history[h][l][1][1]
                        V[l] = params_history[h][l][1][2]
                        rms_after[l] = params_history[h][l][2]
                    else:
                        Q[l] = params_history[h][l][0]
                        K[l] = params_history[h][l][1]
                        V[l] = params_history[h][l][2]
                
                # Extract feedforward network parameters
                w1 = params_history[h][att_layers][0]
                b1 = params_history[h][att_layers][1]
                
                w2 = params_history[h][att_layers+1][0]
                b2 = params_history[h][att_layers+1][1]
                
                w3 = params_history[h][att_layers+2][0]
                b3 = params_history[h][att_layers+2][1]
                
                # Extract scaling parameter
                s = np.array(params_history[h][-1][0])
                
                # Prepare parameters for saving
                save_path = f"{prefix}/iter{run}/params{h*param_store_freq:08d}"
                params_to_save = []
                
                # Add Q, K, V matrices for each attention layer
                for l in range(att_layers):
                    params_to_save.extend([Q[l], K[l], V[l]])
                
                # Add RMS normalization parameters if enabled
                if rms_norm:
                    for l in range(att_layers):
                        params_to_save.extend([rms_before[l], rms_after[l]])
                        # params_to_save.extend(rms_before[l])
                        # params_to_save.extend([rms_after[l]])
                
                # Add feedforward network parameters and scaling parameter
                params_to_save.extend([w1, b1, w2, b2, w3, b3, s])
                
                # Save parameters to npz file
                np.savez(save_path, *params_to_save)
        save_model_parameters(params_history, att_layers, rms_norm, prefix, run, param_store_freq)
        
        
        
        
           
        # Q = {}
        # K = {}
        # V = {}    
        # rms_before = {}
        # rms_after = {}
        # for h in range(len(params_history)):
        #     for l in range(att_layers):
        #         if rms_norm:
        #             rms_before[l] = params_history[h][l][0]
        #             Q[l] = params_history[h][l][1]
        #             K[l] = params_history[h][l][2]
        #             V[l] = params_history[h][l][3]
        #             rms_after[l] = params_history[h][l][4]
        #         else:
        #             Q[l] = params_history[h][l][0]
        #             K[l] = params_history[h][l][1]
        #             V[l] = params_history[h][l][2]

            
        #     w1 = params_history[h][att_layers][0]
        #     b1 = params_history[h][att_layers][1]
            
        #     w2 = params_history[h][att_layers+1][0]
        #     b2 = params_history[h][att_layers+1][1]
            
        #     w3 = params_history[h][att_layers+2][0]
        #     b3 = params_history[h][att_layers+2][1]
            
        #     s = np.array(params_history[h][-1][0])
        #     if att_layers == 4:
        #         q1,k1,v1,q2,k2,v2,q3,k3,v3,q4,k4,v4 = Q[0],K[0],V[0],Q[1],K[1],V[1],Q[2],K[2],V[2],Q[3],K[3],V[3]
        #         if rms_norm:
        #             rms_before_1,rms_after_1,rms_before_2,rms_after_2,rms_before_3,rms_after_3,rms_before_4,rms_after_4 = rms_before[0],rms_after[0],rms_before[1],rms_after[1],rms_before[2],rms_after[2],rms_before[3],rms_after[3]
        #             np.savez(prefix + "/iter%d"%run + "/params%08d"%(h*param_store_freq ),q1,k1,v1,q2,k2,v2,q3,k3,v3,q4,k4,v4,rms_before_1,rms_after_1,rms_before_2,rms_after_2,rms_before_3,rms_after_3,rms_before_4,rms_after_4,w1,b1,w2,b2,w3,b3,s)
        #         else:
        #             np.savez(prefix + "/iter%d"%run + "/params%08d"%(h*param_store_freq ),q1,k1,v1,q2,k2,v2,q3,k3,v3,q4,k4,v4,w1,b1,w2,b2,w3,b3,s)
        #     elif att_layers == 3:
        #         q1,k1,v1,q2,k2,v2,q3,k3,v3 = Q[0],K[0],V[0],Q[1],K[1],V[1],Q[2],K[2],V[2]
        #         if rms_norm:
        #             rms_before_1, rms_after_1, rms_before_2, rms_after_2, rms_before_3, rms_after_3 = rms_before[0],rms_after[0],rms_before[1],rms_after[1],rms_before[2],rms_after[2]
        #             np.savez(prefix + "/iter%d"%run + "/params%08d"%(h*param_store_freq ),q1,k1,v1,q2,k2,v2,q3,k3,v3,rms_before_1,rms_after_1,rms_before_2,rms_after_2,rms_before_3,rms_after_3,w1,b1,w2,b2,w3,b3,s)
        #         else:
        #             np.savez(prefix + "/iter%d"%run + "/params%08d"%(h*param_store_freq ),q1,k1,v1,q2,k2,v2,q3,k3,v3,w1,b1,w2,b2,w3,b3,s)
        #     elif att_layers == 2:
        #         q1,k1,v1,q2,k2,v2 = Q[0],K[0],V[0],Q[1],K[1],V[1]
        #         if rms_norm:
        #             rms_before_1, rms_after_1, rms_before_2, rms_after_2 = rms_before[0],rms_after[0],rms_before[1],rms_after[1]
        #             np.savez(prefix + "/iter%d"%run + "/params%08d"%(h*param_store_freq ),q1,k1,v1,q2,k2,v2,rms_before_1,rms_after_1,rms_before_2,rms_after_2,w1,b1,w2,b2,w3,b3,s)
        #         else:
        #             np.savez(prefix + "/iter%d"%run + "/params%08d"%(h*param_store_freq ),q1,k1,v1,q2,k2,v2,w1,b1,w2,b2,w3,b3,s)
        #     elif att_layers == 6:
        #         q1,k1,v1,q2,k2,v2,q3,k3,v3,q4,k4,v4,q5,k5,v5,q6,k6,v6 = Q[0],K[0],V[0],Q[1],K[1],V[1],Q[2],K[2],V[2],Q[3],K[3],V[3],Q[4],K[4],V[4],Q[5],K[5],V[5]
        #         if rms_norm:
        #             rms_before_1, rms_after_1, rms_before_2, rms_after_2, rms_before_3, rms_after_3, rms_before_4, rms_after_4, rms_before_5, rms_after_5, rms_before_6, rms_after_6 = rms_before[0],rms_after[0],rms_before[1],rms_after[1],rms_before[2],rms_after[2],rms_before[3],rms_after[3],rms_before[4],rms_after[4],rms_before[5],rms_after[5]
        #             np.savez(prefix + "/iter%d"%run + "/params%08d"%(h*param_store_freq ),q1,k1,v1,q2,k2,v2,q3,k3,v3,q4,k4,v4,q5,k5,v5,q6,k6,v6,rms_before_1,rms_after_1,rms_before_2,rms_after_2,rms_before_3,rms_after_3,rms_before_4,rms_after_4,rms_before_5,rms_after_5,rms_before_6,rms_after_6,w1,b1,w2,b2,w3,b3,s)
        #         else:
        #             np.savez(prefix + "/iter%d"%run + "/params%08d"%(h*param_store_freq ),q1,k1,v1,q2,k2,v2,q3,k3,v3,q4,k4,v4,q5,k5,v5,q6,k6,v6,w1,b1,w2,b2,w3,b3,s)
        #     elif att_layers == 5:
        #         q1,k1,v1,q2,k2,v2,q3,k3,v3,q4,k4,v4,q5,k5,v5 = Q[0],K[0],V[0],Q[1],K[1],V[1],Q[2],K[2],V[2],Q[3],K[3],V[3],Q[4],K[4],V[4]
        #         if rms_norm:
        #             rms_before_1, rms_after_1, rms_before_2, rms_after_2, rms_before_3, rms_after_3, rms_before_4, rms_after_4, rms_before_5, rms_after_5 = rms_before[0],rms_after[0],rms_before[1],rms_after[1],rms_before[2],rms_after[2],rms_before[3],rms_after[3],rms_before[4],rms_after[4]
        #             np.savez(prefix + "/iter%d"%run + "/params%08d"%(h*param_store_freq ),q1,k1,v1,q2,k2,v2,q3,k3,v3,q4,k4,v4,q5,k5,v5,rms_before_1,rms_after_1,rms_before_2,rms_after_2,rms_before_3,rms_after_3,rms_before_4,rms_after_4,rms_before_5,rms_after_5,w1,b1,w2,b2,w3,b3,s)
        #         else:
        #             np.savez(prefix + "/iter%d"%run + "/params%08d"%(h*param_store_freq ),q1,k1,v1,q2,k2,v2,q3,k3,v3,q4,k4,v4,q5,k5,v5,w1,b1,w2,b2,w3,b3,s)

        


