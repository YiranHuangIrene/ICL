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
os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] ='false'
os.environ['XLA_PYTHON_CLIENT_ALLOCATOR']='platform'
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'

ndevices = jax.local_device_count()
print(jax.devices())
device = int(sys.argv[9])

jax.config.update("jax_default_device", jax.devices()[device])
key = random.PRNGKey(device)
np.random.seed(device)

np.set_printoptions(precision = 3, suppress = True)

#Parameters
K = int(sys.argv[1])
L = 32
S = 10000
N = int(sys.argv[2])
Nmax = 32
D = int(sys.argv[3]) - (2*Nmax + 1)

alpha = float(sys.argv[4])
P = 1.0/(np.arange(1,K+1)**alpha)
P /= np.sum(P)

B = int(sys.argv[5])
p_B = float(sys.argv[6])
p_C = float(sys.argv[7])

niters = 150000
batchsize = 128
lr = 0.01
w_decay = 1e-6
eps = float(sys.argv[8])

att_layers = 2
mlp_layers = 3

no_repeats = True
nruns = 1
store= True


keys= random.split(key,nruns)

prefix = "./outs/I%08d_K%d_N%d_L%d_D%d_a_%.2f_B%d_pB_%.2f_pC_%.2f_e%.3f_lr%.3f_nr%d" %(niters,K,N,L,D,alpha,B,p_B,p_C,eps,lr,int(no_repeats))

print(prefix)

for ii in range(nruns):
    run = 2*ii + device

    #Loading datasets

    mus_label, mus_class, labels_class = get_mus_label_class(K,L,D)

    test_inputs, test_labels  = generate_input_seqs(mus_label,mus_class,labels_class,S,N, Nmax,eps = eps, P = P, B = B, p_B = p_B, p_C = p_C, no_repeats = no_repeats)

    test_inputs_ic, test_labels_ic =  generate_input_seqs(mus_label,mus_class,labels_class,S,N, Nmax,eps = eps, P = P, B = B, p_B = 1, p_C = 1, no_repeats = no_repeats)

    test_inputs_ic2, test_labels_ic2 =  generate_input_seqs(mus_label,mus_class,labels_class,S,N, Nmax,eps = eps, P = P, B = B, p_B = 1, p_C = 0, flip_labels = True, no_repeats = no_repeats)

    test_inputs_iw, test_labels_iw =  generate_input_seqs(mus_label,mus_class,labels_class,S,N, Nmax,eps = eps, P = P, B = 0, p_B = 0, p_C = 0, no_repeats = no_repeats)


    k_dim = int(test_inputs.shape[-1]) #1. Also works for 1. Not really. 

    params = init_network_params(att_layers, mlp_layers, k_dim ,2*Nmax + 1 + D, L,keys[ii], scale = 1/np.sqrt(D))

    params_history = []
    targets_num = []

    print_acc = True
    print_loss = False
    print_freq= 1000 #5000
    param_store_freq = 500

    targets_num_iter = np.zeros((niters,K))

    for n in range(niters):
        start = time.time()
        if n%param_store_freq==0:
            params_store = []
            for p in range(len(params)):
                params_store += [[np.array(params[p][q]) for q in range(len(params[p]))]]

            params_history += [params_store]
        
        if n%print_freq == 0:
            if print_loss:
                loss_test = loss(params,test_inputs,test_labels)
                loss_ic = loss(params,test_inputs_ic,test_labels_ic)
                loss_ic2 = loss(params,test_inputs_ic2,test_labels_ic2)
                loss_iw = loss(params,test_inputs_iw,test_labels_iw)
                print("Run: %d  Iter (x%d): %04d  Test loss: %.3f IC loss: %.3f IC2 loss: %.3f IW loss: %.3f"\
                      %(run+1,print_freq, n/print_freq,loss_test, loss_ic, loss_ic2, loss_iw))
            if print_acc:
                loss_test = loss(params,test_inputs,test_labels)
                acc_test = accuracy(params,test_inputs,test_labels)
                acc_ic = accuracy(params,test_inputs_ic,test_labels_ic)
                acc_ic2 = accuracy(params,test_inputs_ic2,test_labels_ic2, flip_labels = True)
                acc_iw = accuracy(params,test_inputs_iw,test_labels_iw)
                print("Run: %d  Iter (x%d): %04d Test loss: %.3f Test acc: %.3f IC acc: %.3f IC2 acc: %.3f IW acc: %.3f"\
                      %(run+1,print_freq,n/print_freq,loss_test, acc_test, acc_ic, acc_ic2, acc_iw))

        
        end1 = time.time()
        inputs_batch, labels_batch, target_classes = generate_input_seqs(mus_label,mus_class,labels_class,batchsize,N, Nmax,eps = eps, P = P, B = B, p_B = p_B, p_C = p_C, output_target_labels = True, no_repeats = no_repeats)
        end2 = time.time()
        params = update(params,inputs_batch,labels_batch,  lr = lr)
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
            
        for h in range(len(params_history)):
            q1 = params_history[h][0][0]
            k1 = params_history[h][0][1]
            v1 = params_history[h][0][2]
            
            q2 = params_history[h][1][0]
            k2 = params_history[h][1][1]
            v2 = params_history[h][1][2]
            
            w1 = params_history[h][2][0]
            b1 = params_history[h][2][1]
            
            w2 = params_history[h][3][0]
            b2 = params_history[h][3][1]
            
            w3 = params_history[h][4][0]
            b3 = params_history[h][4][1]
            
            s = np.array(params_history[h][-1][0])
            
            np.savez(prefix + "/iter%d"%run + "/params%08d"%(h*param_store_freq ),q1,k1,v1,q2,k2,v2,w1,b1,w2,b2,w3,b3,s)


        


