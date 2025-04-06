#/bin/bash
cd /shared-local/aoq609/MLLM_ICL/ICL
source /shared-local/aoq609/anaconda3/bin/activate ICL
export CUDA_VISIBLE_DEVICES=0

K=4096
N=8
D=128
a=0
B=2
pB=1
pC=0
eps=0.1
no_repeats=0
rope=1
rope_base=10000
num_heads=1
mlp_layers=3
block=1
act="silu"
rms_norm=1

# Example
# python3 main_jax.py ${K} ${N} ${D} ${a} ${B} ${pB} ${pC} ${eps} ${no_repeats} ${rope} ${rope_base} ${att_layers} ${num_heads} ${mlp_layers} ${block} ${act} ${rms_norm} ${device} &
for device in 0;
do
    for att_layers in 2
    do
        # python3 main_jax.py 2048 ${N} ${D} ${a} ${B} ${pB} ${pC} ${eps} ${no_repeats} ${rope} ${rope_base} ${att_layers} ${num_heads} ${mlp_layers} ${block} ${act} ${rms_norm} ${device} &
        # python3 main_jax.py ${K} 16 ${D} ${a} ${B} ${pB} ${pC} ${eps} ${no_repeats} ${rope} ${rope_base} ${att_layers} ${num_heads} ${mlp_layers} ${block} ${act} ${rms_norm} ${device} &
        python3 main_jax.py ${K} ${N} ${D} ${a} ${B} ${pB} ${pC} ${eps} ${no_repeats} ${rope} ${rope_base} ${att_layers} ${num_heads} ${mlp_layers} ${block} ${act} ${rms_norm} ${device} 
        # python3 main_jax.py ${K} ${N} ${D} 1 ${B} ${pB} ${pC} ${eps} ${no_repeats} ${rope} ${rope_base} ${att_layers} ${num_heads} ${mlp_layers} ${block} ${act} ${rms_norm} ${device} &
    done
done









