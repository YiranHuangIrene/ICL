import numpy as np
import jax
import jax.numpy as jnp
from jax import grad, jit, vmap,pmap,value_and_grad
from jax import random
from jax.nn import relu, silu
from functools import partial
from typing import Optional

def apply_rms_norm(
    x: jnp.ndarray,
    weight: jnp.ndarray,
    eps: float = 1e-6,
) -> jnp.ndarray:
    # Calculate the RMS value along the specified axis
    variance = jnp.mean(jnp.square(x), axis=-1, keepdims=True)
    # Add epsilon for numerical stability
    rms = jnp.sqrt(variance + eps)
    # Normalize x
    x_normalized = x / rms
    x_normalized = weight * x_normalized 
    
    return x_normalized

def rotary_embedding(seq_len, dim, base = 10000):
    theta = base ** (-2 * (jnp.arange(0, dim//2 ) / dim))
    pos = jnp.arange(seq_len)[:, None]
    sinusoid = pos * theta[None, :]
    cos = jnp.cos(sinusoid)
    cos = jnp.repeat(cos,2,axis=-1)
    sin = jnp.sin(sinusoid)
    sin = jnp.repeat(sin,2,axis=-1)
    return cos, sin

def apply_rotary_embedding(x, cos, sin):
    x1, x2 = x[..., ::2], x[..., 1::2]
    x_rot = jnp.stack([-x2, x1], axis=-1).reshape(x.shape)
    return x * cos + x_rot * sin

def entropy(p):
    return -jnp.sum(p*np.log(p+1e-12),axis=-1)

def get_attention_probs(QKV,seq,mask):
    querys,keys,values = [jnp.einsum('ijk,lk->ijl',seq,x) for x in QKV] #(batchsize x seq_length x input_dim) 
    probs = jax.nn.softmax(mask[None,:,:] + jnp.einsum('...ij,...kj->...ik',keys,querys),axis=-2)
    return probs[:,:,:]

def attention_head_params(k_dim, input_dim, key, scale=1e-2):
    q_key, k_key, v_key = random.split(key,3)
    return scale*random.normal(q_key, (k_dim, input_dim)), scale*random.normal(k_key, (k_dim, input_dim)),scale*random.normal(v_key, (input_dim, input_dim))

def rms_norm_params(input_dim):
    return jnp.ones((input_dim,))

def init_network_params(att_layers, mlp_layers, k_dim, input_dim, numclasses, rms_norm, key, scale = 1e-2):
    keys = random.split(key, att_layers + 1)
    params = []
    for k in range(att_layers):
        if rms_norm:
            # RMS norm before and after the attention head
            params += [[rms_norm_params(input_dim),attention_head_params(k_dim, input_dim ,keys[k], scale = scale),rms_norm_params(input_dim)]]
            ## RMS norm before the attention head
            # params += [[rms_norm_params(input_dim),attention_head_params(k_dim, input_dim ,keys[k], scale = scale)]]
            # RMS norm after the attention head
            # params += [[attention_head_params(k_dim, input_dim ,keys[k], scale = scale),rms_norm_params(input_dim)]]
        else:
            params += [attention_head_params(k_dim, input_dim ,keys[k], scale = scale)]


    keys = random.split(key, mlp_layers + 1) #TODO if numbes of att layers equals mlp layers, then the generated keys will be the same. 
    for k in range(mlp_layers):
        params += [[random.normal(keys[2*k],(input_dim ,input_dim))*np.sqrt(2/input_dim), \
                    random.normal(keys[2*k+1],(1,))*np.sqrt(1/input_dim)]]           
        
    params += [[random.normal(keys[-1],shape = (numclasses,input_dim))/jnp.sqrt(input_dim)]]

    return params

#input sequence is (batchsize x seq_length x input_dim)
#Q is (k_dim x input_dim)
#K is (k_dim x input_dim)
#V is (input_dim x input_dim)
def attention_head(QKV, seq,mask = None, rope = False, base = 10000):
    querys,keys,values = [jnp.einsum('ijk,lk->ijl',seq,x) for x in QKV] #(batchsize x seq_length x input_dim) 
    if rope:
        cos, sin= rotary_embedding(seq.shape[1], seq.shape[2], base)
        querys, keys = apply_rotary_embedding(querys, cos= cos, sin = sin), apply_rotary_embedding(keys, cos= cos, sin = sin)

    if mask is None:
        mask = jnp.zeros((seq.shape[1],seq.shape[1]))
        
    probs = jax.nn.softmax(mask[None,:,:] + jnp.einsum('...ij,...kj->...ik',keys,querys),axis=-2) # Key * Query_transpose, different from the original transformer, so the mask is also transposed 

    out = jnp.einsum('...ij,...ik->...jk',probs,values) #(batchsize x seq_length x input_dim)

    return out

def attention_head2(QKV, seq,mask = None):
    querys,keys,values = [jnp.einsum('ijk,lk->ijl',seq,x) for x in QKV] #(batchsize x seq_length x input_dim) 
    
    if mask is None:
        mask = jnp.zeros((seq.shape[1],seq.shape[1]))
        
    probs = jax.nn.softmax(mask[None,:,:] + jnp.einsum('...ij,...kj->...ik',keys,querys),axis=-2)

    out = jnp.einsum('...ij,...ik->...jk',probs,values) #(batchsize x seq_length x input_dim)

    return out

def forward(params, inputs,rope = False, base = 10000, act = "silu", rms_norm = True): #inputs are (batchsize x seq_length x input_dim)
    x = inputs
    
    mask = jnp.ones((x.shape[1],x.shape[1]))*(-jnp.inf)
    mask = jnp.tril(mask, k = -1)
    if act == "silu":
        act_fn = silu
    elif act == "relu":
        act_fn = relu

    if rms_norm:
         for l in range(len(params)-4):
            # before
            # residual = x
            # x = apply_rms_norm(x,params[l][0])
            # x = residual + attention_head(params[l][1],x,mask=mask,rope = rope, base = base)
            # after
            # residual = x
            # x = residual + attention_head(params[l][0],x,mask=mask,rope = rope, base = base)
            # x = apply_rms_norm(x,params[l][1])
            # all
            residual = x
            x = apply_rms_norm(x,params[l][0])
            x = residual + attention_head(params[l][1],x,mask=mask,rope = rope, base = base)
            x = apply_rms_norm(x,params[l][2])
            
    else:    
        for l in range(len(params)-4):
            x = x + attention_head(params[l],x,mask=mask,rope = rope, base = base)

    for l in range(len(params)-4, len(params)-1):
        x = act_fn(jnp.matmul(x,params[l][0].T)  + params[l][1])

    x = jnp.matmul(x[:,-1,:],params[-1][0].T)	
    
    return x

def loss(params, inputs, labels, rope = False, base = 10000, act = "silu", rms_norm = True):
    outputs = forward(params,inputs,rope = rope, base = base, act = "silu", rms_norm = rms_norm)
    label_preds = jax.nn.softmax(outputs)
    label_probs = jnp.sum(label_preds*labels,axis=-1)
    return -jnp.mean(jnp.log(label_probs))

def accuracy(params,inputs,labels, mask = None, flip_labels = False, rope = False, base = 10000, act = "silu", rms_norm = True):
    outputs = forward(params,inputs,rope = rope, base = base, act = "silu", rms_norm = rms_norm)
    label_preds = jax.nn.softmax(outputs)
    if mask is None:
        label_preds_inds = jnp.argmax(label_preds,axis=1)
    else:
        label_preds += mask
        label_preds_inds = jnp.argmax(label_preds,axis=1)

    label_inds= jnp.argmax(labels,axis=1)
    #print(label_preds_inds[:10], label_inds[:10])
    if flip_labels:
        label_inds = (jnp.argmax(labels,axis=1) + 1)%labels.shape[-1]

    return jnp.mean(label_preds_inds == label_inds)

@partial(jit, static_argnames=['rope', 'base','act','rms_norm'])
def update(params, x, y, lr = 1e-1, w_decay = 1e-6, rope = False, base = 10000, act = "silu", rms_norm = True):
    # grads = grad(loss,argnums=0)(params, x, y,rope = rope, base = base, act = "silu", rms_norm = rms_norm)
    losses, grads = value_and_grad(loss,argnums=0)(params, x, y,rope = rope, base = base, act = "silu", rms_norm = rms_norm)
    
    params2 = []
    # for p in range(len(params)):
    #     params2 += [[(1-w_decay)*params[p][q] - lr*grads[p][q] for q in range(len(params[p]))]]
        
    for p in range(len(params)):
        # before 
        # if p < len(params)-4 and rms_norm:
        #     rms_before = (1-w_decay)*params[p][0] - lr*grads[p][0]
        #     attn = [(1-w_decay)*params[p][1][r] - lr*grads[p][1][r] for r in range(len(params[p][1]))]
        #     params2 += [[rms_before,attn]]
        # after 
        # if p < len(params)-4 and rms_norm:
        #     attn = [(1-w_decay)*params[p][0][r] - lr*grads[p][0][r] for r in range(len(params[p][0]))]
        #     rms_after = (1-w_decay)*params[p][1] - lr*grads[p][1]
        #     params2 += [[attn,rms_after]]
        # all
        if len(params[p]) == 3 and rms_norm:
            rms_before = (1-w_decay)*params[p][0] - lr*grads[p][0]
            rms_after = (1-w_decay)*params[p][2] - lr*grads[p][2]
            attn = [(1-w_decay)*params[p][1][r] - lr*grads[p][1][r] for r in range(len(params[p][1]))]
            params2 += [[rms_before,attn,rms_after]]
        else:
            params2 += [[(1-w_decay)*params[p][q] - lr*grads[p][q] for q in range(len(params[p]))]]
        
        
    return losses, params2


def init_network_params_mlp(layers, input_dim, numclasses, key, scale = 1e0):
    keys = random.split(key, 2*(layers + 1))
    params = []
    for k in range(layers):
        params += [[random.normal(keys[2*k],(input_dim,input_dim))*np.sqrt(2/input_dim), \
                    random.normal(keys[2*k+1],(1,))*np.sqrt(1/input_dim)]]
        
    params += [[scale*random.normal(keys[-2],shape = (input_dim,numclasses))*np.sqrt(2/input_dim),\
                scale*random.normal(keys[2*k+1],(1,))*np.sqrt(1/input_dim)]]
    return params

def forward_mlp(params, inputs): #inputs are (batchsize x input_dim)
    x = inputs
    
    for i in range(len(params) - 1):
        x = relu(jnp.matmul(x,params[i][0]) + params[i][1])
        
    #x = inputs + x
    x = jnp.matmul(x,params[-1][0]) + params[-1][1]
    
    return x

def loss_mlp(params,inputs,labels):
    outputs = forward_mlp(params,inputs)
    label_preds = jax.nn.softmax(outputs)
    label_probs = jnp.sum(label_preds*labels,axis=-1)
    return -jnp.mean(jnp.log(label_probs))

def accuracy_mlp(params,inputs,labels, mask = None):
    outputs = forward_mlp(params,inputs)
    label_preds = jax.nn.softmax(outputs)
    if mask is None:
        label_preds_inds = jnp.argmax(label_preds,axis=1)
    else:
        label_preds += mask
        label_preds_inds = jnp.argmax(label_preds,axis=1)

    label_inds= jnp.argmax(labels,axis=1)
    #print(label_preds_inds[:10], label_inds[:10])
    return jnp.mean(label_preds_inds == label_inds)

@jit
def update_mlp(params, x, y, lr = 1e-1, w_decay = 1e-6):
    grads = grad(loss_mlp,argnums=0)(params, x, y)
    params2 = []
    for p in range(len(params)):
        params2 += [[(1-w_decay)*params[p][q] - lr*grads[p][q] for q in range(len(params[p]))]]
        
    return params2

