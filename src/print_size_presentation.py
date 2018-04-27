
h=240

# Wdith @h=30
base_w = 975

# Parameters
# Activations
# Gradients for Activations
# Gradients for Parameters
# Overhead

def fast_downsample_cnn_mem_cost(input_h, batch_size=32, in_gb=False):
    h = (input_h/30)*30
    w = (input_h/30)*base_w


    params_mem = 0
    activation_mem = 0

    while h > 30:
        # Param cost
        params_mem += (3*3*1*16 + 16)
        activation_mem += h*w*16 + h*0.5*w*0.5*16
        h *= 0.5
        w *= 0.5

    # multiply by 4 because there are 4 bytes per param (32bit float = 4 bytes)
    params_mem *= 4
    activation_mem *= (4*batch_size )

    if in_gb:
        params_mem /= 1000000000
        activation_mem /= 1000000000

    return activation_mem, params_mem


def cnn_mem_cost(input_h, batch_size=32, in_gb=False):
    h = input_h
    w = (input_h/30)*base_w

    # Param cost
    #              conv1+bias         conv2+bias          conv3+bias         conv4+bias             conv5+bias             conv6+bias            conv7+bias
    params_mem = (3*3*1*64 + 64) + (3*3*64*64 + 64) + (3*3*64*128 + 128) + (3*3*128*128 + 128) + (3*3*128*256 + 256) + (3*3*256*256 + 256) + (3*3*256*256 + 256)

    # Mem cost
    c1 = h*w*64
    c2 = h*w*64
    p1 = (h*.5)*(w*.7)*64
    c3 = (h*.5)*(w*.7)*128
    c4 = (h*.5)*(w*.7)*128
    p2 = (h*.5*.5)*(w*.7*.7)*128
    c5 = (h*.5*.5)*(w*.7*.7)*256
    c6 = (h*.5*.5)*(w*.7*.7)*256
    c7 = (h*.5*.5)*(w*.7*.7)*256

    activation_mem = c1 + c2 + p1 + c3 + c4 + p2 + c5 + c6 + c7


    # multiply by 4 because there are 4 bytes per param (32bit float = 4 bytes)
    params_mem *= 4
    activation_mem *= (4*batch_size )

    if in_gb:
        params_mem /= 1000000000
        activation_mem /= 1000000000

    return activation_mem, params_mem


def lstm_mem_cost(input_h, batch_size=32, in_gb=False):
    h = input_h * 0.5*0.5
    w = (input_h/30)*base_w * 0.7*0.7

    # FC: h->128;  LSTMx3(512x2);  FC: 512x2->167
    # Param cost:  4(nh+h^2) where n=input dim, h=output dim/hidden-dim
    #   For bidirectional LSTM must mulitply by 2
    #   For bias, add an extra h
    #                   LSTM-1 (x=128, h=512)         LSTM-2  (x=512, h=512)        LSTM-3  (x=512, h=512)
    params_mem = 4*2*(128*512 + 512*512 + 512) + 4*2*(512*512 + 512*512 + 512) + 4*2*(512*512 + 512*512 + 512)

    # Mem Cost for LSTM activations: output and hidden cell state (times 2 because biderectional), times number of timesteps
    #       Cell-state   Hidden-state/output
    lstm_1 = 512*2*w + 512*2*w
    lstm_2 = 512*2*w + 512*2*w
    lstm_3 = 512*2*w + 512*2*w

    activation_mem = lstm_1 + lstm_2 + lstm_3

    # multiply by 4 because there are 4 bytes per param (32bit float = 4 bytes)
    params_mem *= 4
    activation_mem *= (4*batch_size )

    if in_gb:
        params_mem /= 1000000000
        activation_mem /= 1000000000

    return activation_mem, params_mem

def other_layers_mem_cost(input_h, batch_size=32, in_gb=False):
    h = input_h * 0.5*0.5
    w = (input_h/30)*base_w * 0.7*0.7

    # dim-reduction 1:  h->128
    # dim-reduction 2: 512 ->167
    params_mem = (h*128*256 + 128)  + (512*2*167 + 167)

    bridge = 128*w
    alphabet = 167*w

    activation_mem = bridge + alphabet

    # multiply by 4 because there are 4 bytes per param (32bit float = 4 bytes)
    params_mem *= 4
    activation_mem *= (4*batch_size )

    if in_gb:
        params_mem /= 1000000000
        activation_mem /= 1000000000

    return activation_mem, params_mem



print("Line height = %d" % h)
activation_mem, params_mem = cnn_mem_cost(h, in_gb=True)
print("CNN params = %f" % params_mem)
print("CNN Activations = %f" % activation_mem)
print("CNN Activation Gradients = %f" % activation_mem)
print("CNN param grads = %f" % params_mem)
print("CNN ADAM overhaed = %f" % (2*params_mem))

cnn_mem = 4*params_mem + 2*activation_mem

print("")

activation_mem, params_mem = lstm_mem_cost(h, in_gb=True)
print("LSTM params = %f" % params_mem)
print("LSTM Activations = %f" % activation_mem)
print("LSTM Activation Gradients = %f" % activation_mem)
print("LSTM param grads = %f" % params_mem)
print("LSTM ADAM overhaed = %f" % (2*params_mem))

lstm_mem = 4*params_mem + 2*activation_mem
print("")

activation_mem, params_mem = other_layers_mem_cost(h, in_gb=True)
print("Other Layers params = %f" % params_mem)
print("Other Layers Activations = %f" % activation_mem)
print("Other Layers Activation Gradients = %f" % activation_mem)
print("Other Layers param grads = %f" % params_mem)
print("Other Layers ADAM overhaed = %f" % (2*params_mem))

other_mem = 4*params_mem + 2*activation_mem

activation_mem, params_mem = fast_downsample_cnn_mem_cost(h, in_gb=True)
print("RapidDownsample params = %f" % params_mem)
print("RapidDownsample Activations = %f" % activation_mem)
print("RapidDownsample Activation Gradients = %f" % activation_mem)
print("RapidDownsample param grads = %f" % params_mem)
print("RapidDownsample ADAM overhaed = %f" % (2*params_mem))

rapid_mem = 4*params_mem + 2*activation_mem



total_mem = cnn_mem + lstm_mem + other_mem

print("")
print("Cnn mem: %f (%f%%)" % (cnn_mem, 100*cnn_mem/total_mem))
print("LSTM mem: %f (%f%%)" % (lstm_mem, 100*lstm_mem/total_mem))
print("Other mem: %f (%f%%)" % (other_mem, 100*other_mem/total_mem))
print("Rapid mem: %f" % (rapid_mem))

print("Rapid in context")
rapid = rapid_mem+2.64
lstm_30 = 0.92
other_30 = 0.04
total = rapid + lstm_30 + other_30
print("Rapid mem + CNN@30mem: %f (%f%%)" % (rapid, 100*rapid/total))
print("LSTM: %f (%f%%)" % (lstm_30, 100*lstm_30/total))
print("Other: %f (%f%%)" % (other_30, 100*other_30/total))
