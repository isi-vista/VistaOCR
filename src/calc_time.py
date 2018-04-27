import numpy as np
import json

#                      Benchmarking   NO Benchmarking
#  Width=100, Avg Time =   0.13          0.14
#  Width=150, Avg Time =   0.19          0.21
#  Width=200, Avg Time =   0.25          0.27
#  Width=250, Avg Time =   0.31          0.34
#  Width=300, Avg Time =   0.38          0.41
#  Width=350, Avg Time =   0.43          0.47
#  Width=400, Avg Time =   0.49          0.54
#  Width=450, Avg Time =   0.55          0.61
#  Width=500, Avg Time =   0.61          0.69
#  Width=550, Avg Time =   0.67          0.75
#  Width=600, Avg Time =   0.73          0.81

def get_speed(w, benchmarking=True):
    if True:
        return 0.0012 * w + 0.01

    speed = { 100: 0.13,
              150: 0.19,
              200: 0.25,
              250: 0.31,
              300: 0.38,
              350: 0.43,
              400: 0.49,
              450: 0.55,
              500: 0.61,
              550: 0.67,
              600: 0.73
    }

    if w in speed:
        return speed[w]

    if w > 600:
        return (w/600 * 0.73)

    raise Exception("Haven't done this bit yet")

# One let's load IAM
#data_file = '/nfs/isicvlnas01/users/srawls/ocr-dev/data/iam/desc.json'
data_file = '/nfs/isicvlnas01/users/srawls/ocr-dev/data/madcat/desc.json'
with open(data_file, 'r') as fh:
    data = json.load(fh)

# let's say max width is 600
max_width = 800

width_counts = np.zeros((max_width))
trash = 0
keep = 0
for entry in data['train']:
    w = entry['width'] 
    if w > max_width:
        trash+=1
        continue
    else:
        keep+=1
    width_counts[w-1] += 1

print("Thrown out %0.2f%% of images as too wide." % (100*trash/(keep+trash)))

width_cumsum = np.cumsum(width_counts)

# Now let's compute time of one epoch under some conditions

# 1) Let's say we only have one bin
etime = keep/64 * get_speed(max_width) / 60
print("1 bin, Time of one epoch = %0.2f min" % etime)

# 2) Let's say we have two bins; let's vary bin delimiter
print("")
times = []
for b in range(50,max_width, 20):
    etime = width_cumsum[b-1]/64 * get_speed(b) + (width_cumsum[max_width-1] - width_cumsum[b-1])/64*get_speed(max_width)
    etime /= 60
    times.append( (etime,b) )

etime, b = min(times, key=lambda x:x[0])
print("2 bins, best solution: [0-%d], [%d,%d], Time of one epoch = %0.2f min" % (b, b, max_width, etime))


# 2) Let's say we have thre bins; let's vary bin delimiter
print("")
times = []
for b1 in range(50,max_width, 20):
    for b2 in range(50,max_width, 20):
        if b2 <= b1: continue

        etime = width_cumsum[b1-1]/64 * get_speed(b1) + (width_cumsum[b2-1] - width_cumsum[b1-1])/64*get_speed(b2) + (width_cumsum[max_width-1] - width_cumsum[b2-1])/64*get_speed(max_width)
        etime /= 60
        times.append( (etime, b1, b2) )

etime, b1, b2 = min(times, key=lambda x:x[0])
print("3 bins, best solution: [0-%d], [%d,%d], [%d,%d], Time of one epoch = %0.2f min" % (b1, b1, b2, b2, max_width, etime))



# 3) Let's say we have four bins; let's vary bin delimiter
print("")
times = []
for b1 in range(50,max_width, 20):
    for b2 in range(50,max_width, 20):
        if b2 <= b1: continue

        for b3 in range(50,max_width, 20):
            if b3 <= b2: continue

            etime = width_cumsum[b1-1]/64 * get_speed(b1) + \
                    (width_cumsum[b2-1] - width_cumsum[b1-1])/64*get_speed(b2) + \
                    (width_cumsum[b3-1] - width_cumsum[b2-1])/64*get_speed(b3) + \
                    (width_cumsum[max_width-1] - width_cumsum[b3-1])/64*get_speed(max_width)
            etime /= 60
            times.append( (etime, b1, b2, b3) )

etime, b1, b2, b3 = min(times, key=lambda x:x[0])
print("4 bins, best solution: [0-%d], [%d,%d], [%d,%d], [%d, %d], Time of one epoch = %0.2f min" % (b1, b1, b2, b2, b3, b3, max_width, etime))



# 4) Let's say we have five bins; let's vary bin delimiter
print("")
times = []
for b1 in range(50,max_width, 20):
    for b2 in range(50,max_width, 20):
        if b2 <= b1: continue

        for b3 in range(50,max_width, 20):
            if b3 <= b2: continue

            for b4 in range(50,max_width, 20):
                if b4 <= b2: continue

#                etime = width_cumsum[b1-1]/64 * get_speed(b1) + \
#                        (width_cumsum[b2-1] - width_cumsum[b1-1])/64*get_speed(b2) + \
#                        (width_cumsum[b3-1] - width_cumsum[b2-1])/64*get_speed(b3) + \
#                        (width_cumsum[b4-1] - width_cumsum[b3-1])/64*get_speed(b4) + \
#                        (width_cumsum[max_width-1] - width_cumsum[b4-1])/64*get_speed(max_width)

                etime = width_counts[0:b1].sum()/64 * get_speed(b1) + \
                        width_counts[b1:b2].sum()/64*get_speed(b2) + \
                        width_counts[b2:b3].sum()/64*get_speed(b3) + \
                        width_counts[b3:b4].sum()/64*get_speed(b4) + \
                        width_counts[b4:].sum()/64*get_speed(max_width)

                etime /= 60
                times.append( (etime, b1, b2, b3, b4) )

etime, b1, b2, b3, b4 = min(times, key=lambda x:x[0])
print("5 bins, best solution: [0-%d], [%d,%d], [%d,%d], [%d, %d], [%d, %d], Time of one epoch = %0.2f min" % (b1, b1, b2, b2, b3, b3, b4, b4, max_width, etime))


print("")
# What about if I don't use bins at all!
time = 0

num_leftover = 0
for w in range(len(width_counts)):

    num_pure_batches = (num_leftover + width_counts[w]) // 64
    num_leftover = (num_leftover + width_counts[w]) % 64

    time += get_speed(w+1) * num_pure_batches

print("Leftover = %d" % num_leftover)

time /= 60
print("Time w/o bins is: %f" % time)
print("Time w/o bins accounting for no benchmarking is: %f" % (time*1.1))
