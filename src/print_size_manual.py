
#base_w = 1175
base_w = 600

def fast_downsample_cnn_mem_cost(input_h, batch_size=32):
    h = (input_h/30)*30
    w = (input_h/30)*base_w


    params = 0
    mem = 0

    while h > 30:
        # Param cost
        params += (3*3*1*16 + 16)
        mem += h*w*16 + h*0.5*w*0.5*16
        h *= 0.5
        w *= 0.5

    return 4*(2*batch_size*mem + params*3)


def cnn_mem_cost(input_h, batch_size=32):
    h = (input_h/30)*30
    w = (input_h/30)*base_w

    # Param cost
    params = (3*3*1*64 + 64) + (3*3*64*64 + 64) + (3*3*64*128 + 128) + (3*3*128*128 + 128) + (3*3*128*128 + 128)

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

    mem = batch_size * (c1 + c2 + p1 + c3 + c4 + p2 + c5 + c6 + c7)

    # *4 because 4 bytes per param
    return 4*(2*mem + params*3)


def lstm_fc_cost(input_h, batch_size=32):
    h = (input_h/30)*30 * 0.5*0.5
    w = (input_h/30)*base_w * 0.7*0.7

    # FC: h->128;  LSTMx3(512x2);  FC: 512x2->167
    # Param cost
    params = (h*128*256 + 128) + ((4*2*512*128 + 4*2*512*512) + (4*2*512*512 + 4*2*512*512) + (4*2*512*512 + 4*2*512*512)) + (512*2*167 + 167)

    # Mem Cost
    bridge = 128*w
    lstm_1 = 4*512*2*w + 512*2*w
    lstm_2 = 4*512*2*w + 512*2*w
    lstm_3 = 4*512*2*w + 512*2*w
    alphabet = 167*w

    mem = batch_size * (bridge + lstm_1 + lstm_2 + lstm_3 + alphabet)

    # *4 because 4 bytes per param
    return 4*(2*mem + params*3)


print("Total Mem Cost (lh=30) = %0.1f GB" % ((cnn_mem_cost(30)+lstm_fc_cost(30))/1000000000))
print("Total Mem Cost (lh=60) = %0.1f GB" % ((cnn_mem_cost(60)+lstm_fc_cost(60))/1000000000))
print("Total Mem Cost (lh=120) = %0.1f GB" % ((cnn_mem_cost(120)+lstm_fc_cost(120))/1000000000))
print("Total Mem Cost (lh=240) = %0.1f GB" % ((cnn_mem_cost(240)+lstm_fc_cost(240))/1000000000))
print("")
print("CNN Mem Cost (lh=30) = %0.1f GB" % (cnn_mem_cost(30)/1000000000))
print("CNN Mem Cost (lh=120) = %0.1f GB" % (cnn_mem_cost(120)/1000000000))
print("CNN Mem Cost (lh=240) = %0.1f GB" % (cnn_mem_cost(240)/1000000000))
print("CNN Mem Cost (lh=1000) = %0.1f GB" % (cnn_mem_cost(1000)/1000000000))
print("CNN Mem Cost (lh=2000) = %0.1f GB" % (cnn_mem_cost(2000)/1000000000))
print("")
print("LSTM Mem Cost (lh=30) = %0.1f GB" % (lstm_fc_cost(30)/1000000000))
print("LSTM Mem Cost (lh=120) = %0.1f GB" % (lstm_fc_cost(120)/1000000000))
print("LSTM Mem Cost (lh=240) = %0.1f GB" % (lstm_fc_cost(240)/1000000000))
print("LSTM Mem Cost (lh=1000) = %0.1f GB" % (lstm_fc_cost(1000)/1000000000))
print("LSTM Mem Cost (lh=2000) = %0.1f GB" % (lstm_fc_cost(2000)/1000000000))
print("")
print("Fast downsample (lh=60) = %0.1f GB" % (fast_downsample_cnn_mem_cost(60)/1000000000))
print("Fast downsample (lh=120) = %0.1f GB" % (fast_downsample_cnn_mem_cost(120)/1000000000))
print("Fast downsample (lh=240) = %0.1f GB" % (fast_downsample_cnn_mem_cost(240)/1000000000))
