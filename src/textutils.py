import math
import sys
import icu_bidi
import unicodedata

import numpy as np


chinese_decompose_map_file = '/home/hltcoe/srawls/VistaOCR/src/chinese-decompose-map-v1.txt'
chinese_decompose_map = dict()
chinese_compose_map = dict()
with open(chinese_decompose_map_file, 'r') as fh:
    for line in fh:
        char, dseq = line.strip().split(",")
        chinese_decompose_map[char] = dseq.split("_")
        chinese_compose_map[dseq.replace("_","")] = char

        #if dseq[-1] in ['₀', '₁', '₂']:
        if dseq[-1] == '₀':
            # Need to offer a 'default' option just in case decoder doesn't work properly
            chinese_compose_map[dseq[:-2].replace("_","")] = char

chinese_decompose_map_v3_file = '/home/hltcoe/srawls/VistaOCR/src/chinese-decompose-map-v3.txt'
chinese_decompose_map_v3 = dict()
chinese_compose_map_v3 = dict()
with open(chinese_decompose_map_v3_file, 'r') as fh:
    for line in fh:
        char, dseq = line.strip().split(",")
        chinese_decompose_map_v3[char] = dseq.split("_")
        chinese_compose_map_v3[dseq.replace("_","")] = char

        #if dseq[-1] in ['₀', '₁', '₂']:
        if dseq[-1] == '₀':
            # Need to offer a 'default' option just in case decoder doesn't work properly
            chinese_compose_map_v3[dseq[:-2].replace("_","")] = char


def decompose_chinese_char(char):
    if char not in chinese_decompose_map:
        return char
    dseq = chinese_decompose_map[char]
    return ''.join(dseq)

def decompose_chinese_string(string):
    out_str = ''
    for char in string:
        out_str += decompose_chinese_char(char)
    return out_str

def compose_chinese_char(decomposed_char):
    #print("Called for: [%s]" % decomposed_char)
    if decomposed_char not in chinese_compose_map:
        #raise Exception("Shouldn't get here...")
        # return something!
        if len(decomposed_char) > 1:
            return decomposed_char[1]
        else:
            return decomposed_char[0]

    return chinese_compose_map[decomposed_char]

# Need to handle ill-formed strings
# possible errors include, e.g.:
#   一X吕八厶司:  ->  XX公司:
#  the first '一' should be an X but is being interpreted as the start of a new chinese char (which it is not)
#   since it is followed by X this is an error
#  Possible solutions:  take '一' on its own followd by an illegitimate char to just be itslef; or remove it?
#
#  Othe rpossible error:  '吅山厈'   should be '吕山厈'  (with veritical composition), can't re-compose with wrong composition type predicted

def compose_chinese_string(decomposed_string):
    # First pass-- let's assume we have a well-formed string
    cur_idx = 0
    out_str = ''
    while cur_idx < len(decomposed_string):
        composition_type = decomposed_string[cur_idx]
        if composition_type == '一' or composition_type == '品' or composition_type == '叕':
            # only have a single-part composition
            # But may need to handle a v0/v1/v2 subscript
            nchars = 2
            if (cur_idx + 2 < len(decomposed_string)) and (decomposed_string[cur_idx + 2] == '₀' or decomposed_string[cur_idx + 2] == '₁' or decomposed_string[cur_idx + 2] == '₂'):
                nchars = 3

            decomposed_char = decomposed_string[cur_idx : cur_idx + nchars]
            out_str += compose_chinese_char(decomposed_char)
            cur_idx += nchars
        elif composition_type == '吅' or composition_type == '吕' or composition_type == '回' or composition_type == '咒' or composition_type == '弼' or composition_type == '冖' or composition_type == '˖':
            # here we have two-part composition
            # Need to handle: do we have a v0/v1/v2 subscript
            nchars = 3
            if (cur_idx + 3 < len(decomposed_string)) and (decomposed_string[cur_idx + 3] == '₀' or decomposed_string[cur_idx + 3] == '₁' or decomposed_string[cur_idx + 3] == '₂'):
                nchars = 4

            decomposed_char = decomposed_string[cur_idx : cur_idx + nchars]
            out_str += compose_chinese_char(decomposed_char)
            cur_idx += nchars
        else:
            # We don't have a match; possibly the char was not decomposable; possibly error in parsing
            # For now, just append char and move to the next spot
            out_str += decomposed_string[cur_idx]
            cur_idx += 1


    return out_str


#### new version
filler_space = '\ue000'
def decompose_chinese_char_v3(char):
    if char not in chinese_decompose_map_v3:
        return filler_space + char + filler_space
    dseq = chinese_decompose_map[char]
    return ''.join(dseq)

def decompose_chinese_string_v3(string):
    out_str = ''
    for char in string:
        out_str += decompose_chinese_char_v3(char)
    return out_str

def compose_chinese_char_v3(decomposed_char):
    #print("Called for: [%s]" % decomposed_char)
    if decomposed_char not in chinese_compose_map_v3:
        raise Exception("Shouldn't get here...")
        # return something!
        #if len(decomposed_char) > 1:
        #    return decomposed_char[1]
        #else:
        #    return decomposed_char[0]

    return chinese_compose_map_v3[decomposed_char]

# Need to handle ill-formed strings
# possible errors include, e.g.:
#   一X吕八厶司:  ->  XX公司:
#  the first '一' should be an X but is being interpreted as the start of a new chinese char (which it is not)
#   since it is followed by X this is an error
#  Possible solutions:  take '一' on its own followd by an illegitimate char to just be itslef; or remove it?
#
#  Othe rpossible error:  '吅山厈'   should be '吕山厈'  (with veritical composition), can't re-compose with wrong composition type predicted

def compose_chinese_string_v3(decomposed_string):
    # First pass-- let's assume we have a well-formed string
    cur_idx = 0
    out_str = ''
    while cur_idx < len(decomposed_string):
        composition_type = decomposed_string[cur_idx]

        if composition_type in ['一', '品', '叕', '吅', '吕', '回', '咒', '弼', '冖', '˖']:
            # Usually have 3 chars to consume (composition type, part1, part2)
            # But may need to handle a v0/v1/v2 subscript
            nchars = 3
            if (cur_idx + 3 < len(decomposed_string)) and (decomposed_string[cur_idx + 3] in ['\ue001','\ue002','\ue003']):
                nchars = 4

            decomposed_char = decomposed_string[cur_idx : cur_idx + nchars]
            out_str += compose_chinese_char_v3(decomposed_char)
            cur_idx += nchars
        else:
            # We don't have a match; possibly the char was not decomposable; possibly error in parsing
            # For now, just append char and move to the next spot
            out_str += decomposed_string[cur_idx]
            cur_idx += 1


    return out_str



#### old version


# R=strong right-to-left;  AL=strong arabic right-to-left
rtl_set =  set(chr(i) for i in range(sys.maxunicode)
               if unicodedata.bidirectional(chr(i)) in ['R','AL'])


def determine_text_direction(text):
    # Easy case first
    for char in text:
        if char in rtl_set:
            return icu_bidi.UBiDiLevel.UBIDI_RTL
    # If we made it here we did not encounter any strongly rtl char
    return icu_bidi.UBiDiLevel.UBIDI_LTR

def utf8_visual_to_logical(text):
    text_dir = determine_text_direction(text)

    bidi = icu_bidi.Bidi()
    bidi.inverse = True
    bidi.reordering_mode = icu_bidi.UBiDiReorderingMode.UBIDI_REORDER_INVERSE_LIKE_DIRECT
    bidi.reordering_options = icu_bidi.UBiDiReorderingOption.UBIDI_OPTION_DEFAULT # icu_bidi.UBiDiReorderingOption.UBIDI_OPTION_INSERT_MARKS

    bidi.set_para(text, text_dir, None)

    res = bidi.get_reordered(0 | icu_bidi.UBidiWriteReorderedOpt.UBIDI_DO_MIRRORING | icu_bidi.UBidiWriteReorderedOpt.UBIDI_KEEP_BASE_COMBINING)

    return res

def utf8_logical_to_visual(text):
    text_dir = determine_text_direction(text)

    bidi = icu_bidi.Bidi()

    bidi.reordering_mode = icu_bidi.UBiDiReorderingMode.UBIDI_REORDER_DEFAULT
    bidi.reordering_options = icu_bidi.UBiDiReorderingOption.UBIDI_OPTION_DEFAULT  #icu_bidi.UBiDiReorderingOption.UBIDI_OPTION_INSERT_MARKS

    bidi.set_para(text, text_dir, None)

    res = bidi.get_reordered(0 | icu_bidi.UBidiWriteReorderedOpt.UBIDI_DO_MIRRORING | icu_bidi.UBidiWriteReorderedOpt.UBIDI_KEEP_BASE_COMBINING)

    return res



def uxxxx_to_utf8(in_str):
    idx = 0
    result = ''
    if in_str.strip() == "":
        return ""

    for uxxxx in in_str.split():
        if uxxxx == '':
            continue

        if uxxxx == "<unk>" or uxxxx == "<s>" or uxxxx == "</s>":
            cur_utf8_char = uxxxx
        else:
            # First get the 'xxxx' part out of the current 'uxxxx' char

            cur_char = uxxxx[1:]

            # Now decode the hex code point into utf-8
            try:
                cur_utf8_char = chr(int(cur_char, 16))
            except:
                print("Exception converting cur_char = [%s]" % cur_char)
                sys.exit(1)

        # And add it to the result buffer
        result = result + cur_utf8_char

    return result

def utf8_to_uxxxx(in_str, output_array=False):

    char_array = []
    for char in in_str:
        raw_hex = hex(ord(char))[2:].zfill(4).lower()
        char_array.append( "u%s" % raw_hex )

    if output_array:
        return char_array
    else:
        return ' '.join(char_array)


# Rudimintary dynamic programming method of computing edit distance bewteen two sequences
# Let dist[i,j] = edit distance between A[0..i] and B[0..j]
#  Then dist[i,j] is the smallest of:
#    (1) dist[i,j-1] + 1   i.e. between A[0..i] and B[0..j-1] plus 1 to cover the insertion of B[j]
#    (2) dist[i-1,j] + 1   i.e. between A[0..i-1] and B[0..j] plus 1 to cover the insertion of A[i]
#    (3) dist[i-1,j-1] + (1 if A[i]===B[j], else 0)  i.e. between A[0..i-1] and B[0..j-1] and  the edit distance between A[i],B[j]
def edit_distance(A, B):
    # If both strings are empty, edit distance is 0                                                                               
    if len(A) == 0 and len(B) == 0:
        return 0
    # If one or the other is empty, then edit distance is length of the other one                                                 
    if len(A) == 0 or len(B) == 0:
        return len(A) + len(B)

    # Otherwise have to actually compute it the hard way :)                                                                       
    dist_matrix = np.zeros((len(A)+1, len(B)+1))
    for i in range(len(A)+1):
        dist_matrix[i, 0] = i
    for j in range(len(B)+1):
        dist_matrix[0, j] = j

    for i in range(1,len(A)+1):
        for j in range(1,len(B)+1):
            if A[i-1] == B[j-1]:
                dist_matrix[i,j] = dist_matrix[i-1,j-1]
            else:
                dist_matrix[i, j] = 1 + min(dist_matrix[i, j - 1], dist_matrix[i - 1, j],
                                            dist_matrix[i - 1, j - 1])

    return dist_matrix[-1, -1]


def form_tokenized_words(chars, with_spaces=False):
    punctuations = {"u002e", "u002c", "u003b", "u0027", "u0022", "u002f", "u0021", "u0028", "u0029", "u005b", "u005d", "u003c", "u003e",
                   "u002d", "u005f", "u007b", "u007d", "u0024", "u0025", "u0023", "u0026", "u060c", "u201d", "u060d", "u060f", "u061f",
                   "u066d", "ufd3e", "ufd3f", "u061e", "u066a", "u066b", "u066c", "u002a", "u002b", "u003a", "u003d", "u005e", "u0060", "u007c", "u007e"}
    digits = {
        "u0660", "u0661", "u0662", "u0663", "u0664", "u0665", "u0666", "u0667", "u0668", "u0669", "u0030", "u0031",
        "u0032", "u0033", "u0034", "u0035", "u0036", "u0037", "u0038", "u0039"
    }

    words = []
    start_idx = 0
    for i in range(len(chars)):
        if chars[i] == 'u0020':  # Space denotes new word
            if start_idx != i:
                words.append('_'.join(chars[start_idx:i]))

                if with_spaces:
                    words.append("u0020")
            start_idx = i + 1
            continue
        if chars[i] in punctuations or chars[i] in digits:
            if start_idx != i:
                words.append('_'.join(chars[start_idx:i]))
            words.append(chars[i])
            start_idx = i + 1
            continue
        if i == len(chars) - 1:
            # At end of line, so just toss remaining line into word array
            if start_idx == i:
                words.append(chars[start_idx])
            else:
                words.append('_'.join(chars[start_idx:]))

    return words


def compute_cer_wer(hyp_transcription, ref_transcription):
    # Assume input in uxxxx format, i.e. looks like this:  "u0062 u0020 u0064 ..."

    # To compute CER we need to split on uxxxx chars, which are seperated by space
    hyp_chars = hyp_transcription.split(' ')
    ref_chars = ref_transcription.split(' ')

    char_dist = edit_distance(hyp_chars, ref_chars)

    # To compute WER we need to split by words, and tokenize on punctuation
    # We rely on Alphabet objects to provide the chars to tokenize on
    hyp_words = form_tokenized_words(hyp_chars)
    ref_words = form_tokenized_words(ref_chars)
    # Remove whitespace at beg/end
    while len(hyp_words) > 0 and hyp_words[0] == 'u0020':
        hyp_words = hyp_words[1:]
    while len(hyp_words) > 0 and hyp_words[-1] == 'u0020':
        hyp_words = hyp_words[:-1]
    while len(ref_words) > 0 and ref_words[0] == 'u0020':
        ref_words = ref_words[1:]
    while len(ref_words) > 0 and ref_words[-1] == 'u0020':
        ref_words = ref_words[:-1]

    word_dist = edit_distance(hyp_words, ref_words)

    return float(char_dist) / len(ref_chars), float(word_dist) / len(ref_words)


def form_target_transcription(target, alphabet):
    return ' '.join([alphabet.idx_to_char[i] for i in target])


def pretty_print_timespan(ts):
    total_seconds = ts.total_seconds()
    if total_seconds < 60:
        return "%0.2fs" % total_seconds
    if total_seconds < 60 * 60:
        minutes = total_seconds // 60
        secs = math.ceil(total_seconds - 60 * minutes)
        return "%dm %ds" % (minutes, secs)
    if total_seconds < 60 * 60 * 24:
        hours = total_seconds // (60 * 60)
        minutes = (total_seconds - 60 * 60 * hours) // 60
        secs = math.ceil(total_seconds - 60 * 60 * hours - 60 * minutes)
        return "%dh %0dm %ds" % (hours, minutes, secs)
    else:
        days = total_seconds // (60 * 60 * 24)
        hours = (total_seconds - 60 * 60 * 24 * days) // (60 * 60)
        minutes = (total_seconds - 60 * 60 * 24 * days - 60 * 60 * hours) // 60
        secs = math.ceil(total_seconds - 60 * 60 * 24 * days - 60 * 60 * hours - 60 * minutes)
        return "%d days %dh %dm %ds" % (days, hours, minutes, secs)
