Code for IEEE CI paper:

N Dethlefs (2017) Domain Transfer for Deep Natural Language Generation from Abstract Meaning Representations. 
IEEE Computational Intelligence Magazine 12 (3), 18-28. (Special Issue on Natural Language Generation with Computational Intelligence).

seq2seq-indomain.py trains an LSTM in the in-domain setting.

seq2seq-outofdomain.py inherits weights from a source domain and evaluates in a new target domain.

seq2seq-prior.py inherits weights from a source domain and then trains on top of that in a target domain.
