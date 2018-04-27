import numpy as np
import time

def edit_distance(A, B):
    # If both strings are empty, edit distance is 0                                                                               
    if len(A) == 0 and len(B) == 0:
        return 0
    # If one or the other is empty, then edit distance is length of the other one                                                 
    if len(A) == 0 or len(B) == 0:
        return len(A) + len(B)

    # Otherwise have to actually compute it the hard way :)                                                                       
    dist_matrix = np.zeros((len(A)+1, len(B)+1))
    trace = np.zeros((len(A)+1, len(B)+1), dtype=np.int)

    COPY = 1
    SUB = 2
    INS = 3
    DEL = 4
    

    for i in range(len(A)+1):
        dist_matrix[i, 0] = i
    for j in range(len(B)+1):
        dist_matrix[0, j] = j

    for i in range(1,len(A)+1):
        for j in range(1,len(B)+1):
            insertion_cost = dist_matrix[i - 1, j] + 1
            deletion_cost = dist_matrix[i, j - 1] + 1
            substitution_or_copy_cost = dist_matrix[i - 1, j - 1]
            if A[i-1] != B[j-1]:
                substitution_or_copy_cost += 1

            dist_matrix[i,j] = min(insertion_cost, deletion_cost, substitution_or_copy_cost)

            if substitution_or_copy_cost <= insertion_cost and substitution_or_copy_cost <= deletion_cost:
                if A[i-1] == B[j-1]:
                    # Copy
                    trace[i,j] = COPY
                else:
                    # Substitituion
                    trace[i,j] = SUB
            elif insertion_cost <= substitution_or_copy_cost and insertion_cost <= deletion_cost:
                # Insertion
                trace[i,j] = INS
            else:
                # Deletion
                trace[i,j] = DEL

    # Form Trace
    i,j = len(A),len(B)
    output_trace = []
    while i>=1 and j>=1:
        op = trace[i,j]
        op_str = ['_',"COPY","SUB","INS","DEL"][op]

        output_trace.append( (op_str, A[i-1], B[j-1]) )

        if op == COPY or op == SUB:
            i -= 1
            j -= 1
        elif op == INS:
            i -= 1
        elif op == DEL:
            j -= 1

    return dist_matrix[-1, -1], output_trace



ref = "\" McNamara's Band , \" \" Greensleeves \" and \" English Rose . \""
hyp = " ' He Namarod's Layd , \" \" breensleeres \" and \" English hose . '"

dist, trace = edit_distance(hyp,ref)
print("ref = %s" % ref)
print("hyp = %s" % hyp)
print("dist = %d" % dist)
print("trace = %s" % str(list(reversed(trace))))

#ref =  " *McNamara's Band , " " Greensleeves " and " English Rose . "
#hyp =  ' He Namarod's Layd , " " breensleeres " and " English hose . '
#dist = 13
#trace = [('SUB', "'", '"'), ('COPY', ' ', ' '), ('INS', 'H', ' '), ('SUB', 'e', 'M'), ('SUB', ' ', 'c'), ('COPY', 'N', 'N'), ('COPY', 'a', 'a'), ('COPY', 'm', 'm'), ('COPY', 'a', 'a'), ('COPY', 'r', 'r'), ('INS', 'o', 'r'), ('SUB', 'd', 'a'), ('COPY', "'", "'"), ('COPY', 's', 's'), ('COPY', ' ', ' '), ('SUB', 'L', 'B'), ('COPY', 'a', 'a'), ('SUB', 'y', 'n'), ('COPY', 'd', 'd'), ('COPY', ' ', ' '), ('COPY', ',', ','), ('COPY', ' ', ' '), ('COPY', '"', '"'), ('COPY', ' ', ' '), ('COPY', '"', '"'), ('COPY', ' ', ' '), ('SUB', 'b', 'G'), ('COPY', 'r', 'r'), ('COPY', 'e', 'e'), ('COPY', 'e', 'e'), ('COPY', 'n', 'n'), ('COPY', 's', 's'), ('COPY', 'l', 'l'), ('COPY', 'e', 'e'), ('COPY', 'e', 'e'), ('SUB', 'r', 'v'), ('COPY', 'e', 'e'), ('COPY', 's', 's'), ('COPY', ' ', ' '), ('COPY', '"', '"'), ('COPY', ' ', ' '), ('COPY', 'a', 'a'), ('COPY', 'n', 'n'), ('COPY', 'd', 'd'), ('COPY', ' ', ' '), ('COPY', '"', '"'), ('COPY', ' ', ' '), ('COPY', 'E', 'E'), ('COPY', 'n', 'n'), ('COPY', 'g', 'g'), ('COPY', 'l', 'l'), ('COPY', 'i', 'i'), ('COPY', 's', 's'), ('COPY', 'h', 'h'), ('COPY', ' ', ' '), ('SUB', 'h', 'R'), ('COPY', 'o', 'o'), ('COPY', 's', 's'), ('COPY', 'e', 'e'), ('COPY', ' ', ' '), ('COPY', '.', '.'), ('COPY', ' ', ' '), ('SUB', "'", '"')]
