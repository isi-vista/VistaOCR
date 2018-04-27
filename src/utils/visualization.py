import torch
import matplotlib.pyplot as plt
import matplotlib
import cv2
import math
import random
import numpy as np

_tableau20 = [(31, 119, 180), (174, 199, 232), (255, 127, 14), (255, 187, 120),    
             (44, 160, 44), (152, 223, 138), (214, 39, 40), (255, 152, 150),    
             (148, 103, 189), (197, 176, 213), (140, 86, 75), (196, 156, 148),    
             (227, 119, 194), (247, 182, 210), (127, 127, 127), (199, 199, 199),    
             (188, 189, 34), (219, 219, 141), (23, 190, 207), (158, 218, 229)]    
  
# Scale the RGB values to the [0, 1] range, which is the format matplotlib accepts.    
for i in range(len(_tableau20)):    
    r, g, b = _tableau20[i]    
    _tableau20[i] = (r / 255., g / 255., b / 255.)   



def _setup_image(model_input, batch_id=None):
    # Handle torch Variable instances
    if batch_id is None:
        if isinstance(model_input, torch.autograd.Variable):
            img = model_input.data
        else:
            img = model_input
    else:
        if isinstance(model_input, torch.autograd.Variable):
            img = model_input.data[batch_id]
        else:
            img = model_input[batch_id]

    # Copy to CPU if needed
    if isinstance(img, torch.cuda.FloatTensor):
        img = img.cpu()

    # NumPy-ify and change from CHW to HWC
    img = img.numpy().transpose( (1,2,0) )

    # Undo image normalization
    img = img*(-255)+255

    if img.shape[2] == 1:
        # matplotlib plots grayscale images correctly only if you get rid of channel dimension
        img = img[:,:,0]
        cmap = plt.cm.gray
    else:
        # OpenCV images are BGR whereas matplotlib assumes RGB
        img = cv2.convertColor(img, cv.COLOR_BGR2RGB)
        cmap = None # fallback to default

    return img, cmap 


def _form_display_char(idx, alphabet):
    # Special case for CTC Blank
    if idx == 0:
        return '_'

    # Special case for space so it shows up
    if alphabet[idx] == 'u0020':
        return '[SP]'
        
    # Otherwise, just convert to utf-8
    return chr(int(alphabet[idx][1:], 16))

def _find_low_confidence_spans(model_output, alphabet, conf_thresh, batch_id=None):
    # Actual model output is not set to probability vector yet, need to run softmax
    probs = torch.nn.functional.softmax(model_output.view(-1, model_output.size(2))).view(model_output.size(0), model_output.size(1), -1)

    if batch_id is None:
        batch_id = 0

    # Handle torch Variable instances
    if isinstance(probs, torch.autograd.Variable):
        probs = probs.data[:,batch_id,:]
    else:
        probs = probs[:,batch_id,:]

    # Copy to CPU if needed
    if isinstance(probs, torch.cuda.FloatTensor):
        probs = probs.cpu()

    # Squeeze away unused dimension
    probs.squeeze_()

    # Now let's cycle through frames and check for low confidence regions
    low_confidence_spans = []
    topk = 5
    for t in range(probs.size(0)):
        topk_vals, topk_idxs = torch.topk(probs[t], topk)
        if topk_vals[0] < conf_thresh:
            options = []
            for i in range(topk):
                char = _form_display_char(topk_idxs[i], alphabet)
                options.append( (char, topk_vals[i], topk_idxs[i] ) )
                tot_conf = 0
                for _, prob, _ in options:
                    tot_conf += prob

                if tot_conf >= conf_thresh:
                    break

            low_confidence_spans.append( (t, t, options) )


    return low_confidence_spans

def _decode_with_alignment_spans(model_output, alphabet, batch_id=None):
    min_prob_thresh = 3* 1/len(alphabet)

    if batch_id is None:
        batch_id = 0
    # Handle torch Variable instances
    if isinstance(model_output, torch.autograd.Variable):
        probs = model_output.data[:,batch_id,:]
    else:
        probs = model_output[:,batch_id,:]

    # Copy to CPU if needed
    if isinstance(probs, torch.cuda.FloatTensor):
        probs = probs.cpu()

    # Now time to decode
    argmaxs, argmax_idxs = probs.max(dim=1)
    argmax_idxs.squeeze_()
    argmaxs.squeeze_()
    prev_max = None
    span_start = 0

    alignment_tuples = []
    for t in range(probs.size(0)):
        cur_max_prob = argmaxs[t]
        cur_max = argmax_idxs[t]

        # Heuristic
        # If model is predicting very low probability for all letters in alphabet, treat that the
        # samed as a CTC blank
        if cur_max_prob < min_prob_thresh:
            cur_max = 0

        if prev_max is None:
            prev_max = cur_max
            continue
        if prev_max != cur_max:
            char = _form_display_char(prev_max, alphabet)
            alignment_tuples.append( (span_start, t, char, prev_max) )
            span_start = t+1
            prev_max = cur_max

    # Handle last leftover if nescesary
    if span_start != probs.size(0):
        char = _form_display_char(prev_max, alphabet)
        alignment_tuples.append( (span_start, probs.size(0)-1, char, prev_max) )

    return alignment_tuples



def display_target(target, alphabet):
    string_utf8 = ""
    string_uxxxx = ""
    for char_idx in target:
        string_uxxxx += alphabet[char_idx] + ' '
        string_utf8 += chr(int(alphabet[char_idx][1:], 16))

    print("Target utf8 string is [%s]" % string_utf8)

    # For Arabic, it is sometimes helpful to dipslay the uxxxx output
#    if not alphabet.left_to_right:
#        print("Target uxxxx string is: \n\t%s" % string_uxxxx)

def display_image(model_input, batch_id=None):
    img, cmap = _setup_image(model_input, batch_id)

    # Need to determine appropriate figure size
    # For now, hardcoded to 12 inches wide seems to work okay
    w = 12
    h = math.ceil(img.shape[0] * w / img.shape[1])
    fig = plt.figure(figsize=(w,h), dpi=300)

    # Setup axis with a bit of margin for viewability
    margin=0.05
    ax = fig.add_axes([margin, margin, 1 - 2*margin, 1 - 2*margin])

    # Finally, show image
    ax.set_yticks([])
    ax.set_xticks([])
    ax.imshow(img, cmap=plt.cm.gray)
    plt.show()


def overlay_hidden_activations(model_input, hidden, scale_factor=(1.0/0.49), batch_id=None):
    # Setup input image
    img, cmap = _setup_image(model_input, batch_id)

    # Need to determine appropriate figure size
    # For now, hardcoded to 12 inches wide seems to work okay
    w = 12
    h = math.ceil(img.shape[0] * w / img.shape[1])

    # (1) Setup raw plot of hidden activations overlayed on image
    fig = plt.figure(figsize=(w,h), dpi=300)

    # Setup axis with a bit of margin for viewability
    margin=0.05
    ax = fig.add_axes([margin, margin, 1 - 2*margin, 1 - 2*margin])

    # Now simply plot hidden activations in image
    if isinstance(hidden, torch.autograd.Variable) or isinstance(hidden, torch.FloatTensor):
        hidden = hidden.cpu().numpy()

    ax2 = ax.twinx()
    hidden_xs = range(hidden.shape[0])
    hidden_xs = [x*scale_factor for x in hidden_xs]
    ax2.plot(hidden_xs, hidden)
    ax2.set_ylim(-1,1)
    ax2.set_yticks([])
    ax2.set_xticks([])

    # Finally, show image
    ax.set_yticks([])
    ax.set_xticks([])

    ax.imshow(img, cmap=plt.cm.gray)
    plt.show()

    # (2) Setup color-coded background overlay
    fig = plt.figure(figsize=(w,h), dpi=300)

    # Setup axis with a bit of margin for viewability
    margin=0.05
    ax = fig.add_axes([margin, margin, 1 - 2*margin, 1 - 2*margin])

    # Correct for interpolation due to scaling
    # Basic idea is to split the difference: half of 'gap' goes to left-side, half of 'gap' goes to right side
    left_correction = math.floor(scale_factor/2)
    right_correction = math.floor(scale_factor/2)
    for t in range(hidden.shape[0]):
        left_x = scale_factor*t - left_correction
        right_x = scale_factor*(t+1) + right_correction
    
        #seismic or bwr
        ax.axvspan(left_x, right_x, color=plt.cm.seismic( (hidden[t]+1)/2 ), alpha=0.5)


    ax.set_yticks([])
    ax.set_xticks([])
    ax.imshow(img, cmap=plt.cm.gray)
    plt.show()



def overlay_alignment(model_input, model_output, alphabet, scale_factor=(1.0/0.49), batch_id=None):
    # Setup input image
    img, cmap = _setup_image(model_input, batch_id)

    # Need to determine appropriate figure size
    # For now, hardcoded to 12 inches wide seems to work okay
    w = 12
    h = math.ceil(img.shape[0] * w / img.shape[1])
    fig = plt.figure(figsize=(w,h), dpi=300)

    # Setup axis with a bit of margin for viewability
    margin=0.05
    ax = fig.add_axes([margin, margin, 1 - 2*margin, 1 - 2*margin])

    # Now handle argmax decoding
    alignment_tuples = _decode_with_alignment_spans(model_output, alphabet, batch_id)

    # Now color-code spans
    for span_start, span_end, span_char, span_char_id in alignment_tuples: 
        letter_color = _tableau20[span_char_id % len(_tableau20)]

        # Correct for interpolation due to scaling
        # Basic idea is to split the difference: half of 'gap' goes to left-side, half of 'gap' goes to right side
        left_correction = math.floor(scale_factor/2)
        right_correction = math.floor(scale_factor/2)

        left_x = scale_factor*span_start - left_correction
        right_x = scale_factor*span_end + right_correction

        ax.axvspan(left_x, right_x, color=letter_color, alpha=0.5)

        # Place label for span in center of span
        # Also prepare line segment to point to span
        label_x = (left_x + right_x)/2
        label_y = -10
        rotation = 0
        label_x_correction = 0
        if span_char == "[SP]":
            rotation = 90
            label_x_correction = -2
            label_y = -20

        ax.annotate(span_char, (label_x,0), (label_x + label_x_correction, label_y), arrowprops={'arrowstyle': '->'}, xycoords='data', textcoords='data', rotation=rotation)

    # Finally, show image
    ax.set_yticks([])
    ax.set_xticks([])
    ax.imshow(img, cmap=plt.cm.gray)
    plt.show()


def display_low_confidence_regions(model_input, model_output, alphabet, scale_factor=(1.0/0.49), conf_thresh=0.99, batch_id=None):
    # Setup input image
    img, cmap = _setup_image(model_input, batch_id)

    # Need to determine appropriate figure size
    # For now, hardcoded to 12 inches wide seems to work okay
    w = 12
    h = math.ceil(img.shape[0] * w / img.shape[1])
    fig = plt.figure(figsize=(w,h), dpi=300)

    # Setup axis with a bit of margin for viewability
    margin=0.05
    ax = fig.add_axes([margin, margin, 1 - 2*margin, 1 - 2*margin])
    ax.set_yticks([])
    ax.set_xticks([])

    # Now handle argmax decoding
    spans = _find_low_confidence_spans(model_output, alphabet, conf_thresh, batch_id)

    low_conf_area = 0
    low_conf_area_v2 = 0

    # Use these to remember the number of characters we showed previuosly, to properly position current labels
    # Even-numbered spans are above the image and odd-numbered spans are below, so keep track of both seperately
    prev_len_even = 2
    prev_len_odd = 2

    # Now color-code spans
    for span_idx, (span_start, span_end, char_array) in enumerate(spans): 
        low_conf_area += (span_end - span_start + 1)

        # Want to count perent of area where confusion is between more than one character in the model,
        # Not just between CTC-Blank and a character in the model
        if len(char_array) > 2 or (len(char_array) == 2 and (char_array[0][2] != 0 and char_array[1][2] != 0)):
            low_conf_area_v2 += (span_end - span_start + 1)

        span_color = _tableau20[random.randint(0, len(_tableau20)-1)]

        # Correct for interpolation due to scaling
        # Basic idea is to split the difference: half of 'gap' goes to left-side, half of 'gap' goes to right side
        left_correction = math.floor(scale_factor/2)
        right_correction = math.floor(scale_factor/2)

        left_x = scale_factor*span_start - left_correction
        right_x = scale_factor*span_end + right_correction

        ax.axvspan(left_x, right_x, color=span_color, alpha=0.5)

        # Place label for span in center of span
        # Also prepare line segment to point to span
        label_x = (left_x + right_x)/2
        label_x_correction = -8

        delta = 10
        if span_idx % 2 == 0:
            delta_y = -delta
            arrow_y = 5

            if span_idx % 4 == 0:
                label_y = -delta
            else:
                label_y = -delta + prev_len_even * delta_y 


            prev_len_even = len(char_array)
        else:
            delta_y = delta
            arrow_y = img.shape[0] - delta

            if span_idx % 4 == 1:
                label_y = img.shape[0] + delta
            else:
                label_y = delta + prev_len_odd * delta_y + img.shape[0]

            prev_len_odd = len(char_array)
            char_array = list(reversed(char_array))

        for i, (char, prob, char_idx) in enumerate(char_array):
            if i == len(char_array)-1:
                ax.annotate("%s (%d)" % (char,int(100*prob)), (label_x,arrow_y), (label_x + label_x_correction, (label_y + delta_y*(len(char_array)-i-1))), arrowprops={'arrowstyle': '->', 'alpha': 0.2}, xycoords='data', textcoords='data', color=span_color)
            else:
                ax.text(label_x + label_x_correction, (label_y + delta_y*(len(char_array)-i-1)), "%s (%d)" % (char,int(100*prob)), color=span_color)



    # Finally, show image
    print("Percentage of frames having confidence < %.2f is %.2f%%. Shown below:" % (conf_thresh, 100*low_conf_area/model_output.size(0)))
    print("Percentage of frames having confidence < %.2f with confusion b/w more than CTC blank is %.2f%%. Shown below:" % (conf_thresh, 100*low_conf_area_v2/model_output.size(0)))
    ax.set_yticks([])
    ax.set_xticks([])
    ax.imshow(img, cmap=plt.cm.gray)
    plt.show()
