#!/bin/env python

import argparse
from collections import Counter
import xml.dom.minidom as md
import os, sys
import csv
import codecs


def read_vocab(ltf_dirs, other_dict = {}):
    threshold = 5
    short_word_length = 3
    short_word_threshold = 50
    word_counter = Counter()
    nfiles = 0
    
    for d in ltf_dirs:
        skipped_dict = Counter()
        print("*** directory: {}".format(d))
        for ltf_file in [x for x in os.listdir(d) if".ltf.xml" in x]:
            nfiles +=1
            ltf_xml =  md.parse(os.path.join(d, ltf_file))
            tokens = ltf_xml.documentElement.getElementsByTagName("TOKEN")
            words = [t.firstChild.data for t in tokens]
            if other_dict:
                oth_words = [w for w in words if w in other_dict]
                new_words = [w for w in words if w not in other_dict]
                skipped_dict.update(oth_words)
                word_counter.update(new_words)
            else:
                word_counter.update(words)
            if nfiles % 5000 == 0:
                print("finished reading {} files; unpruned vocab: {}".format(nfiles, (len(word_counter))))
                print(word_counter.most_common(30))
            if nfiles == 200000:
                break
        print("For {} skipped {} tokens".format(d, len(skipped_dict)))
        print(skipped_dict.most_common(20))
    print("finished reading {} files; unpruned vocab: {}".format(nfiles, (len(word_counter))))
                                                                                                                                                    
    #prune
    trusted_vocabulary = {}
    for (word) in word_counter.keys():
        count = word_counter[word]
        if count > short_word_threshold:
            trusted_vocabulary[word] = count
        elif len(word) > short_word_length and count > threshold:
            trusted_vocabulary[word] = count
    print("Pruned vocab: {}".format((len(trusted_vocabulary))))
    return trusted_vocabulary

def clean_ocr(ocr_file, out_dir, lang, trusted_vocab):
    rows_to_keep = []
    with codecs.open(ocr_file, "rb", "utf8") as f:
        reader = csv.reader(f)
        for row in reader:
            tokens = row[6].split()
            known = [t for t in tokens if t in trusted_vocab]
            long_tokens = [t for t in known if len(t) > 3]
            if len(long_tokens) < 4:
                if ("EU" in known or "US" in known) and len(known) < 3:
                    rows_to_keep.append(row)                    
            elif len(known) > 0.3 * len(tokens):
                rows_to_keep.append(row)

    with codecs.open(os.path.join(out_dir, lang+".cleaned.csv"), "w", "utf8") as outf:
        csv_writer = csv.writer(outf)
        for row in rows_to_keep:
            csv_writer.writerow(row)

def main(argv=None):
    if argv is None:
        argv = sys.argv[1:] 

    parser = argparse.ArgumentParser(description='Clean OCR outputs using dictionary')
    parser.add_argument('-i', '--input-text', dest='text',
                        help="The input ocr text")
    parser.add_argument('-m', '--dictionary-root', dest='dict_root',
                        help="Dictionary root dir")
    parser.add_argument('-o', '--out-dir', dest='out_dir',
                        help="The output text dir")
    args = parser.parse_args()

    lrlp_root = args.dict_root 
    en_ltf1 = os.path.join(lrlp_root,"data/translation/from_rus/eng/ltf")
    en_ltf2 = os.path.join(lrlp_root,"data/translation/from_eng/news/eng/ltf")
    en_ltf3 = os.path.join(lrlp_root,"data/translation/from_eng/phrasebook/eng/ltf")
    en_ltf4 = os.path.join(lrlp_root,"data/translation/from_eng/elicitation/eng/ltf")
    en_ltf5 = os.path.join(lrlp_root,"data/translation/found/eng/ltf")

    ru_ltf1 = os.path.join(lrlp_root,"data/translation/from_rus/rus/ltf")
    ru_ltf2 = os.path.join(lrlp_root,"data/translation/from_eng/news/rus/ltf")
    ru_ltf3 = os.path.join(lrlp_root,"data/translation/from_eng/phrasebook/rus/ltf")
    ru_ltf4 = os.path.join(lrlp_root,"data/translation/from_eng/elicitation/rus/ltf")
    ru_ltf5 = os.path.join(lrlp_root,"data/translation/found/rus/ltf")

    trusted_vocab_en = read_vocab([en_ltf1, en_ltf2, en_ltf3, en_ltf4, en_ltf5])
    trusted_vocab_ru = read_vocab([ru_ltf1, ru_ltf2, ru_ltf3, ru_ltf4, ru_ltf5], trusted_vocab_en)

    clean_ocr(args.text, args.out_dir, "en", trusted_vocab_en)
    clean_ocr(args.text, args.out_dir, "ru", trusted_vocab_ru)

if __name__ == '__main__':
    main()
