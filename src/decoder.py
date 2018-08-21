import sys
import os
import numpy as np
import torch
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor

from textutils import uxxxx_to_utf8


class LmDecoder:
    # max active 3k - 7k
    # beam: 11 - 17
    # lattice_beam 6-9
    def __init__(self, alphabet, lm_file, word_sym_file, lm_units, acoustic_weight=0.8, max_active=5000, beam=16.0, lattice_beam=10.0):
        self.alphabet = alphabet

        # Only pull in if needed
        script_path = os.path.dirname(os.path.realpath(__file__))
        #sys.path.append(script_path + "/../eesen")
        sys.path.insert(0,'/home/hltcoe/srawls/pyeesen')
        import eesen
        print("Loading eesen from: %s" % eesen.__file__)
        self.acoustic_weight = acoustic_weight
        self.lattice_decoder = eesen.LatticeDecoder(lm_file, word_sym_file, self.acoustic_weight, max_active, beam,
                                                    lattice_beam)

        #self.lattice_decoder.EnableLattices("/home/hltcoe/srawls/tmp.lat.gz")

        # Need to keep track of model-alphabet to LM-alphabet conversion
        units = ['<ctc-blank>']
        with open(lm_units, 'r') as fh:
            for line in fh:
                units.append(line.strip().split(' ')[0])

        self.lmidx_to_char = units
        self.lmchar_to_idx = dict(zip(units, range(len(units))))
        

        # Let's precompute some stuff to make lm faster
        print("Prep work...")
        self.lm_swap_idxs = []
        self.lm_swap_idxs_modelidx = []
        self.lm_swap_idxs_lmidx = []
        self.add_to_blank_char = []
        self.add_to_blank_idx = []
        for model_idx in range(len(self.alphabet.idx_to_char)):
            char = self.alphabet.idx_to_char[model_idx]
            if not char in self.lmchar_to_idx:
                self.add_to_blank_char.append(char)
                self.add_to_blank_idx.append(model_idx)
                continue
            lm_idx = self.lmchar_to_idx[char]
            self.lm_swap_idxs.append( (model_idx,lm_idx) )
            self.lm_swap_idxs_modelidx.append(model_idx)
            self.lm_swap_idxs_lmidx.append(lm_idx)
        print("Done prep work")
        if len(self.add_to_blank_char) > 0:
            print("\tFYI: these chars were in model but not in LM:  %s" % str(self.add_to_blank_char))

    def decode(self, tp_executor, model_output, batch_actual_timesteps, uttids, uxxxx=False):
        T = model_output.size()[0]
        B = model_output.size()[1]

        # Actual model output is not set to probability vector yet, need to run softmax
        probs = torch.nn.functional.log_softmax(model_output.view(-1, model_output.size(2)), dim=1).view(model_output.size(0),
                                                                                                  model_output.size(1),
                                                                                                  -1)

        # Need to take care of issue where prob goes to a char in model-alphabet but not in lm-alphabet
        # Just assign high prob to ctc-blank?
        #print("Sum of missing chars' prob = %s" % str(model_output[:,:,self.add_to_blank_idx].sum(dim=2)))
        #probs[:,:,0] += probs[:,:,self.add_to_blank_idx].sum(dim=2)
        #probs[:,:,self.add_to_blank_idx] = 0

        # Make sure we're on CPU
        probs = probs.data.cpu()

        # We process decoder parallely in worker threads; store those async futures here
        decoder_futures = [None]*B

        def decode_helper(probs, uttid, uxxxx):
            res = self.lattice_decoder.Decode(probs, uttid)
            res_utf8 = ''
            if uxxxx == False:
                for uxxxx_word in res.split(' '):
                    res_utf8 += ''.join([uxxxx_to_utf8(r) for r in uxxxx_word.split('_')])
                res = res_utf8
            else:
                res_flatten = ''
                for uxxxx_word in res.split(' '):
                    for uxxxx_char in uxxxx_word.split('_'):
                        res_flatten += uxxxx_char
                        res_flatten += ' '
                res = res_flatten.strip()

            return res

        for b in range(B):
            probs_remapped = np.full( (batch_actual_timesteps[b], len(self.lmidx_to_char)), np.log(1e-10))
            probs_remapped[:,self.lm_swap_idxs_lmidx] = probs[:batch_actual_timesteps[b], b, self.lm_swap_idxs_modelidx]

            # Just for right-to-left languages!
            #probs_remapped = probs_remapped[::-1]

            decoder_futures[b] = tp_executor.submit(decode_helper, probs_remapped, uttids[b], uxxxx)

        # At this point all decoder tasks are done (we are outside scope of with ThreadPoolExecutor, so it has finished)
        return decoder_futures


class ArgmaxDecoder:
    def __init__(self, alphabet):
        self.alphabet = alphabet

    def decode(self, model_output, batch_actual_timesteps, uxxxx=False, lang=None):
        start_decode = datetime.now()

        if lang is None:
            min_prob_thresh = 3 * 1 / len(self.alphabet)
        else:
            min_prob_thresh = 3 * 1 / len(self.alphabet[lang])

        T = model_output.size()[0]
        B = model_output.size()[1]

        prev_char = ['' for _ in range(B)]
        result = ['' for _ in range(B)]

        for t in range(T):

            # #gpu argmax (bug!!!!!)
            # gpu_argmax = True
            # argmaxs, argmax_idxs = model_output.data[t].max(dim=1)
            # argmaxs.squeeze_()
            # argmax_idxs.squeeze_()    

            # cpu argmax
            gpu_argmax = False
            model_output_at_t_cpu = model_output.data[t].cpu().numpy()
            argmaxs = model_output_at_t_cpu.max(1).flatten()
            argmax_idxs = model_output_at_t_cpu.argmax(1).flatten()

            for b in range(B):
                # Only look at valid model output for this batch entry
                if t >= batch_actual_timesteps[b]:
                    continue

                if argmax_idxs[b] == 0:  # CTC Blank
                    prev_char[b] = ''
                    continue

                # Heuristic
                # If model is predicting very low probability for all letters in alphabet, treat that the
                # samed as a CTC blank
                if argmaxs[b] < min_prob_thresh:
                    prev_char[b] = ''
                    continue


                if lang is None:
                    char = self.alphabet.idx_to_char[argmax_idxs[b]]
                else:
                    char = self.alphabet[lang].idx_to_char[argmax_idxs[b]]

                if prev_char[b] == char:
                    continue

                result[b] += char
                prev_char[b] = char

                # Add a space to all but last iteration
                if t != T - 1:
                    result[b] += ' '

        # Strip off final token-stream space if needed
        for b in range(B):
            if len(result[b]) > 0 and result[b][-1] == ' ':
                result[b] = result[b][:-1]

        # Check if we should return utf8 output
        if uxxxx == False:
            result = [uxxxx_to_utf8(r) for r in result]

        return result
