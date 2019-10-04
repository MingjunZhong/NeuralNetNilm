# Sequence-to-point learning for non-intrusive load monitoring (energy disaggregation)

We introduced two frameworks for NILM using neural networks in our AAAI-18 paper [1], which are

1. sequence-to-sequence (seq2seq) model, where both the input (mains) and output (appliance) of the networks are windows. (In the paper, we set the window length to have 599 timepoints which is equivalent to 599*7=4193 seconds.)

2. sequence-to-point (seq2point) model, where the input is the mains windows (599 timepoints in the paper) and output is the midpoint of the corresponding appliance windows (a single point).

A better code can be found at https://github.com/MingjunZhong/transferNILM

Any questions please drop me an email at mzhong@lincoln.ac.uk

Reference:
[1] Chaoyun Zhang, Mingjun Zhong, Zongzuo Wang, Nigel Goddard, and Charles Sutton. ``Sequence-to-point learning with neural networks for nonintrusive load monitoring.’’ Thirty-Second AAAI Conference on Artificial Intelligence (AAAI-18), Feb. 2-7, 2018.
