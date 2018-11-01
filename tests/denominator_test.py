#!python

import torch
import simplefst
import pychain
import random
import numpy as np

# num-pdfs (D) = 3237
# num-hmm-states (N) = 16602

D = 3237

pychain.set_verbose_level(4)

def chain_denominator_test(den_graph, use_cuda):
    num_sequences = random.randint(1, 5)
    frames_per_sequence = random.randint(10, 20)

    if random.randint(0, 3) == 0:
        frames_per_sequence *= 30

    zero_output = (random.randint(0, 3) == 0)
    if not zero_output:
        nnet_output = torch.randn([num_sequences * frames_per_sequence, D], device=dev, dtype=torch.float)
    else:
        nnet_output = torch.zeros([num_sequences * frames_per_sequence, D], device=dev, dtype=torch.float)

    training_opts = pychain.ChainTrainingOptions()
    training_opts.leaky_hmm_coefficient = 0.00001

    nnet_output_deriv = torch.zeros_like(nnet_output)

    objf = pychain.compute_objf_and_deriv(
        training_opts, den_graph, num_sequences, nnet_output, nnet_output_deriv)
    per_frame = objf / (num_sequences * frames_per_sequence)

    print ("objf is {} = {} per frame".format(objf, per_frame))
    print ("deriv = {}".format(nnet_output_deriv))

    output_deriv_sum = nnet_output_deriv.sum()
    print ("Sum of nnet-output-deriv is {} vs expected {}".format(
        output_deriv_sum, num_sequences * frames_per_sequence))
    assert (float(output_deriv_sum) -
            float(num_sequences * frames_per_sequence) < 10.0)


    num_tries = 5
    epsilon = 1e-4

    predicted_objf_changes = np.zeros(num_tries)
    observed_objf_changes = np.zeros(num_tries)

    for p in range(num_tries):
        nnet_delta_output = torch.randn_like(nnet_output)
        nnet_delta_output.mul_(epsilon)

        predicted_objf_changes[p] = torch.trace(
            torch.mm(nnet_output_deriv, nnet_delta_output.transpose(0, 1)))

        nnet_output_perturbed = torch.tensor(nnet_delta_output)
        nnet_output_perturbed.add_(nnet_output)

        nnet_output_deriv_perturbed_temp = torch.zeros_like(nnet_output)
        objf_perturbed = pychain.compute_objf_and_deriv(
            training_opts, den_graph, num_sequences, nnet_output_perturbed,
            nnet_output_deriv_perturbed_temp)

        observed_objf_changes[p] = objf_perturbed - objf

    print ("Predicted objf changes are {}", predicted_objf_changes)
    print ("Observed objf changes are {}", observed_objf_changes)

    error = predicted_objf_changes - observed_objf_changes
    print ("num-sequences = {}, frames-per-sequence = {}, relative error is {}"
           "".format(num_sequences, frames_per_sequence,
                     np.linalg.norm(error) /
                     np.linalg.norm(predicted_objf_changes)))

    if frames_per_sequence < 50:
        assert(np.linalg.norm(error) <
               0.25 * np.linalg.norm(predicted_objf_changes))


cuda0 = torch.device('cuda:0')

for n in range(3):
    for use_cuda in [False, True]:
        dev = cuda0 if use_cuda else 'cpu'

        with pychain.ostream_redirect():
            den_fst = simplefst.StdVectorFst.read("tests/den.fst")
            den_graph = pychain.DenominatorGraph(den_fst, D, use_cuda)

            chain_denominator_test(den_graph, use_cuda)
