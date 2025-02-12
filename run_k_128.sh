#/bin/bash
cd /shared-local/aoq609/MLLM_ICL/ICL
source /shared-local/aoq609/anaconda3/bin/activate ICL

K=128
N=8
D=63
a=0
pB=1
pC=0
eps=0.1


for B in 0 1 2;
do
	python3 ic_vs_iw_v3.py ${K} ${N} ${D} ${a} ${B} ${pB} ${pC} ${eps} 0 &
	pid1=$!
	python3 ic_vs_iw_v3.py ${K} ${N} ${D} ${a} ${B} ${pB} ${pC} ${eps} 1 &
	pid2=$!
	wait $pid1
	wait $pid2
done


