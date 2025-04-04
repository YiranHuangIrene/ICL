#/bin/bash
cd /shared-local/aoq609/MLLM_ICL/ICL
source /shared-local/aoq609/anaconda3/bin/activate ICL
export CUDA_VISIBLE_DEVICES=0

K=1024  # Number of classes
N=8  # Number of item-label pairs in the context
D=64  # Feature dimension
L=32  # Number of labels
alpha=0  # Zipf's law exponent
B=1 # Burstiness
p_B=1  # Fraction of bursty sequences
p_C=0  # Fraction of OOD sequences
eps=0.1  # Within-class variance
no_repeats=0 # Whether repeated items are allowed in the context
n_heads=1  # Number of attention heads
n_layers=2 # Number of transformer layers
rope_theta=10000  # Rope base
rms_norm=1 # Whether to use RMS normalization
batch_size=256
optimizer=SGD
device=0


# Example
# python3 main.py ${K} ${N} ${D} ${L} ${alpha} ${B} ${p_B} ${p_C} ${eps} ${no_repeats} ${n_heads} ${n_layers} ${rope_theta} ${rms_norm} ${optimizer} ${device}
python3 main.py ${K} ${N} ${D} ${L} ${alpha} ${B} ${p_B} ${p_C} ${eps} ${no_repeats} ${n_heads} ${n_layers} ${rope_theta} ${rms_norm} ${batch_size} ${optimizer} ${device} &