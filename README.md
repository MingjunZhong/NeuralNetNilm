# Neural networks for non-intrusive load monitoring (energy disaggregation)

1. Seq2seq model here is the model introduced in our AAAI-18 paper. Both the input and output of the networks are (mains and appliance) windows (599 timepoints: 599*7=4193 seconds).
2. Seq2point model: the input is the mains windows (599 timepoints); and output is the midpoint of the corresponding appliance windows.

Reference:
Chaoyun Zhang, Mingjun Zhong, Zongzuo Wang, Nigel Goddard, and Charles Sutton. ``Sequence-to-point learning with neural networks for nonintrusive load monitoring.’’ Thirty-Second AAAI Conference on Artificial Intelligence (AAAI-18), Feb. 2-7, 2018.
