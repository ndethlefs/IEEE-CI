# Code for IEEE CIM: "Domain Transfer for Deep Natural Language Generation from Abstract Meaning Representations"

This repository contains code used in the following paper: 

N Dethlefs (2017) Domain Transfer for Deep Natural Language Generation from Abstract Meaning Representations. 
IEEE Computational Intelligence Magazine 12 (3), 18-28. (Special Issue on Natural Language Generation with Computational Intelligence).

See link here: https://ieeexplore.ieee.org/document/7983466

This paper explores the use of Abstract Meaning Representations (AMRs) to facilitate transfer learning in natural language 
generation. The basic idea is that using a common underlying input representation, a sequence-to-sequence deep learning model
will be able to learn a set general-purpose lexico-syntactic structures that are sufficiently generalisable to generate 
language in a new unseen target domain from training in a source domain even if the two are quite different.

The basic learning task is such that a model tries to generate a set of common paraphrases from an AMR input:

![alt text](/img/samples.pdf)

Experiments in three different domains and with six datasets demonstrate that the lexical-syntactic constructions learnt in one domain can be transferred to new domains and 
 achieve up to 75-100% of the performance of in-domain training. This is based on objective metrics such as BLEU and semantic
  error rate and a subjective human rating study. Training a policy from prior knowledge from a different domain is 
  consistently better than pure in-domain training by up to 10%.

![alt text](/img/res.png)


# Code

The data folder contains data from the available domains (not all are complete in the upload) alongside some sample AMR annotations.

There are 3 main scripts: 

seq2seq-indomain.py trains an LSTM in the in-domain setting.

seq2seq-outofdomain.py inherits weights from a source domain and evaluates in a new target domain.

seq2seq-prior.py inherits weights from a source domain and then trains on top of that in a target domain.

Domains can be varied by giving different input datasets for training and testing (see towards top of scripts), for example:

INPUT_DATA_FILE = "/data/give_amr.txt"
TEST_FILE = "/data/gre_amr.txt"
