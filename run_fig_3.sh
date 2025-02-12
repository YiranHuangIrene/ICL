#/bin/bash
cd /shared-local/aoq609/MLLM_ICL/ICL
source /shared-local/aoq609/anaconda3/bin/activate ICL

K=256
N=8
D=63
a=0
B=1
pB=1
pC=0.8
eps=0.1


python3 ic_vs_iw_v3.py ${K} ${N} ${D} ${a} ${B} ${pB} ${pC} ${eps} 0 
