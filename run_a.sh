#/bin/bash
cd /shared-local/aoq609/MLLM_ICL/ICL
source /shared-local/aoq609/anaconda3/bin/activate ICL

K=1024
N=8
D=63
B=1
pC=0
pB=1
eps=0.1



for a in 0.25 0.5 0.75 1.0 1.25 1.5 1.75;
do
	python3 ic_vs_iw_v3.py ${K} ${N} ${D} ${a} ${B} ${pB} ${pC} ${eps} 0 &
	pid1=$!
	wait $pid1
done
