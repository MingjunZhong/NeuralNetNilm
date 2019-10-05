# Sequence-to-point learning for non-intrusive load monitoring (energy disaggregation)

We introduced two frameworks for NILM using neural networks in our AAAI-18 paper [1], which are

1. sequence-to-sequence (seq2seq) model, where both the input (mains) and output (appliance) of the networks are windows. (In the paper, we set the window length to have 599 timepoints which is equivalent to 599*7=4193 seconds.)

2. sequence-to-point (seq2point) model, where the input is the mains windows (599 timepoints in the paper) and output is the midpoint of the corresponding appliance windows (a single point).

A more refined code can be found at https://github.com/MingjunZhong/transferNILM, where you could also get preprocessed data.

Seq2seq & seq2point learning are also implemented in nilmtk-contrib [2], which is available at https://github.com/nilmtk/nilmtk-contrib

Any questions please drop me an email at mzhong@lincoln.ac.uk

Reference:

[1] Chaoyun Zhang, Mingjun Zhong, Zongzuo Wang, Nigel Goddard, and Charles Sutton. ``Sequence-to-point learning with neural networks for nonintrusive load monitoring.’’ Thirty-Second AAAI Conference on Artificial Intelligence (AAAI-18), Feb. 2-7, 2018.

[2] Nipun Batra, Rithwik Kukunuri, Ayush Pandey, Raktim Malakar, Rajat Kumar, Odysseas Krystalakos, Mingjun Zhong, Paulo Meira, and Oliver
Parson. 2019. ``Towards reproducible state-of-the-art energy disaggregation''. In The 6th ACM International Conference on Systems for Energy-Efficient Buildings, Cities, and Transportation (BuildSys ’19), November 13–14, 2019, New York, NY, USA. ACM, New York, NY, USA, 10 pages. https://doi.org/10.1145/3360322.3360844
