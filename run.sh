#/bin/bash

K=128
N=8
D=63
a=0
B=1
pB=1
pC=0.8
eps=0.1

for K in 128 256 512 2048;
do
	for B in 0 1 2 4;
	do
		python3 ic_vs_iw_v3.py ${K} ${N} ${D} ${a} ${B} ${pB} ${pC} ${eps} 0 &
		pid1=$!
		python3 ic_vs_iw_v3.py ${K} ${N} ${D} ${a} ${B} ${pB} ${pC} ${eps} 1 &
		pid2=$!
		wait $pid1
		wait $pid2
	done
done

# for pB in 1.0;
# do
# 	for a in 0.0 0.25 0.5 0.75 1.0 1.25 1.5 1.75;
# 	do
# 		python3 ic_vs_iw_v3.py ${K} ${N} ${D} ${a} ${B} ${pB} ${pC} ${eps} 0 &
# 		pid1=$!
# 		python3 ic_vs_iw_v3.py ${K} ${N} ${D} ${a} ${B} ${pB} ${pC} ${eps} 1 &
# 		pid2=$!
# 		wait $pid1
# 		wait $pid2
# 	done
# done

#for eps in 0.0 0.1 0.2 0.4 0.8 1.6 3.2 6.4;
# for eps in 0.3 0.5 0.6 0.7;
# do
# 	python3 ic_vs_iw_v3.py ${K} ${N} ${D} ${a} ${B} ${pB} ${pC} ${eps} 0 &
# 	pid1=$!
# 	python3 ic_vs_iw_v3.py ${K} ${N} ${D} ${a} ${B} ${pB} ${pC} ${eps} 1 &
# 	pid2=$!
# 	wait $pid1
# 	wait $pid2
# done

# python3 ic_vs_iw_v3.py ${K} ${N} ${D} ${a} ${B} ${pB} ${pC} ${eps} 0 
# python3 ic_vs_iw_v3.py 128 8 63 0 1 1 0.8 0.3 0 