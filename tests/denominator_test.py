#!python

import torch
import simplefst
import pychain

# num-pdfs (D) = 3237
# num-hmm-states (N) = 16602
# minibatch-size (B) = 128
# chunk-size (T) = 100

D = 3237
B = 1
T = 100

pychain.set_verbose_level(4)

use_cuda = False

cuda0 = torch.device('cuda:0')
dev = cuda0 if use_cuda else 'cpu'

with pychain.ostream_redirect():
    den_fst = simplefst.StdVectorFst.read("tests/den.fst")
    den_graph = pychain.DenominatorGraph(den_fst, D, use_cuda)

    N = den_graph.num_states()

    nnet_output = torch.zeros(B * T, D, device=dev)
    nnet_output_deriv = torch.zeros(B * T, D, device=dev)

    training_opts = pychain.ChainTrainingOptions()
    training_opts.leaky_hmm_coefficient = 0.0

    objf = pychain.compute_objf_and_deriv(training_opts, den_graph, 1, nnet_output, nnet_output_deriv)

    print ("objf = {0}, deriv = {1}".format(objf, nnet_output_deriv))
