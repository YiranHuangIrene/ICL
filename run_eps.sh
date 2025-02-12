#/bin/bash
cd /shared-local/aoq609/MLLM_ICL/ICL
source /shared-local/aoq609/anaconda3/bin/activate ICL

K=1024
N=8
D=63
a=0
B=1
pB=1
pC=0

for eps in 0.3 0.5 0.6 0.7;
do
	python3 ic_vs_iw_v3.py ${K} ${N} ${D} ${a} ${B} ${pB} ${pC} ${eps} 0 &
	pid1=$!
	wait $pid1
done

