#!/bin/bash

source ~/.bashrc

if [ $# -ne 2 ] && [ $# -ne 3 ]; then
    echo "USAGE:  ./decode_testset.sh <model-path> <data-path>  [<lm-path>]"
    exit 1
fi


model_path=$1
datadir=$2

if [ $# -eq 3 ]; then
    lmpath=$3
else
    lmpath=
fi

OUTDIR=./hyp-output
mkdir -p ${OUTDIR}

SCLITE=sclite
script_dir="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

# Run Decoding
python3 ${script_dir}/decode_testset.py --datadir=${datadir} --model-path=${model_path} --lm-path=${lmpath} --outdir=${OUTDIR}

if [[ $? -ne 0 ]]; then
    echo "Error in decoding"
    exit 1
fi

# Generate ref output
python ${script_dir}/print_ref.py ${datadir} > $OUTDIR/ref-chars.txt


#python ${script_dir}/normalize_arabic.py $OUTDIR/ref-chars.txt $OUTDIR/ref-chars.txt-BACKUP
#mv $OUTDIR/ref-chars.txt-BACKUP $OUTDIR/ref-chars.txt
#python ${script_dir}/normalize_arabic.py $OUTDIR/hyp-chars.txt $OUTDIR/hyp-chars.txt-BACKUP
#mv $OUTDIR/hyp-chars.txt-BACKUP $OUTDIR/hyp-chars.txt

# Turn decoding output into tokenized words
python ${script_dir}/chars_to_tokenized_words.py $OUTDIR/hyp-chars.txt $OUTDIR/hyp-words.txt
python ${script_dir}/chars_to_tokenized_words.py $OUTDIR/ref-chars.txt $OUTDIR/ref-words.txt

if [[ -e $OUTDIR/hyp-lm-chars.txt ]]; then
    python ${script_dir}/chars_to_tokenized_words.py $OUTDIR/hyp-lm-chars.txt $OUTDIR/hyp-lm-words.txt
fi


# Do CER measurement
${SCLITE} -r $OUTDIR/ref-chars.txt -h $OUTDIR/hyp-chars.txt -i swb -o all >/dev/null

if [[ -e $OUTDIR/hyp-lm-chars.txt ]]; then
    ${SCLITE} -r $OUTDIR/ref-chars.txt -h $OUTDIR/hyp-lm-chars.txt -i swb -o all >/dev/null
fi

# Do WER measurement
${SCLITE} -r $OUTDIR/ref-words.txt -h $OUTDIR/hyp-words.txt -i swb -o all >/dev/null

if [[ -e $OUTDIR/hyp-lm-words.txt ]]; then
    ${SCLITE} -r $OUTDIR/ref-words.txt -h $OUTDIR/hyp-lm-words.txt -i swb -o all >/dev/null
fi


# Now display results
# TODO -- prettier printing / display
echo "No LM CER:"
grep 'Sum/Avg' $OUTDIR/hyp-chars.txt.sys

if [[ -e $OUTDIR/hyp-lm-chars.txt.sys ]]; then
    echo "LM CER:"
    grep 'Sum/Avg' $OUTDIR/hyp-lm-chars.txt.sys
fi

echo "No LM WER:"
grep 'Sum/Avg' $OUTDIR/hyp-words.txt.sys

if [[ -e $OUTDIR/hyp-lm-words.txt.sys ]]; then
    echo "LM WER:"
    grep 'Sum/Avg' $OUTDIR/hyp-lm-words.txt.sys
fi

