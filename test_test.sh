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
    lmpath=''
fi

#model_path=/nfs/isicvlnas01/users/srawls/ocr-dev/tmp-fake-model-best_model.pth
#datadir=/nfs/isicvlnas01/users/srawls/ocr-dev/data/iam
#lmpath=/nfs/isicvlnas01/users/jmathai/experiments/iam_lm_augment_more_data/IAM-LM/
#lmpath = /nfs/isicvlnas01/users/jschmidt/aida-lm/ocr-eesen-lm/AIDA-LM-SCRATCH

export PYTHONPATH=/nas/home/srawls/ocr/PyTorchOCR/eesen/:$PYTHONPATH


OUTDIR=./hyp-output
mkdir -p ${OUTDIR}


SCLITE=/nfs/isicvlnas01/share/sclite/sclite
script_dir="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

# Run Decoding
#python ${script_dir}/decode_testset.py --datadir=${datadir} --model-path=${model_path} --lm-path=${lmpath} --outdir=${OUTDIR}

if [[ $? -ne 0 ]]; then
    echo "Error in decoding"
    exit 1
fi


# Turn decoding output into tokenized words
#python ${script_dir}/chars_to_tokenized_words.py $OUTDIR/hyp-chars.txt $OUTDIR/hyp-words.txt
#python ${script_dir}/chars_to_tokenized_words.py $OUTDIR/hyp-lm-chars.txt $OUTDIR/hyp-lm-words.txt
#python ${script_dir}/chars_to_tokenized_words.py $OUTDIR/ref-chars.txt $OUTDIR/ref-words.txt
#python ${script_dir}/chars_to_tokenized_words.py $OUTDIR/font-hyp-chars.txt $OUTDIR/font-hyp-words.txt
#python ${script_dir}/chars_to_tokenized_words.py $OUTDIR/font-hyp-lm-chars.txt $OUTDIR/font-hyp-lm-words.txt
#python ${script_dir}/chars_to_tokenized_words.py $OUTDIR/font-ref-chars.txt $OUTDIR/font-ref-words.txt


echo "Doing CER"
# Do CER measurement
${SCLITE} -r $OUTDIR/ref-chars.txt -h $OUTDIR/hyp-chars.txt -i swb -o all >/dev/null
echo "LM"
${SCLITE} -r $OUTDIR/ref-chars.txt.uniq -h $OUTDIR/hyp-lm-chars.txt.uniq -i swb -o all >/dev/null
echo "after LM"
${SCLITE} -r $OUTDIR/font-ref-chars.txt -h $OUTDIR/font-hyp-chars.txt -i swb -o all >/dev/null
#${SCLITE} -r $OUTDIR/font-ref-chars.txt -h $OUTDIR/font-hyp-lm-chars.txt -i swb -o all >/dev/null

echo "Doing WER"
# Do WER measurement
${SCLITE} -r $OUTDIR/ref-words.txt -h $OUTDIR/hyp-words.txt -i swb -o all >/dev/null
echo "LM"
${SCLITE} -r $OUTDIR/ref-words.txt -h $OUTDIR/hyp-lm-words.txt.uniq -i swb -o all >/dev/null
echo "After LM"
${SCLITE} -r $OUTDIR/font-ref-words.txt -h $OUTDIR/font-hyp-words.txt -i swb -o all >/dev/null
#${SCLITE} -r $OUTDIR/font-ref-words.txt -h $OUTDIR/font-hyp-lm-words.txt -i swb -o all >/dev/null


# Now display results
# TODO -- prettier printing / display
echo "No LM CER:"
grep 'Sum/Avg' $OUTDIR/hyp-chars.txt.sys
echo "LM CER:"
grep 'Sum/Avg' $OUTDIR/hyp-lm-chars.txt.sys

echo "No LM WER:"
grep 'Sum/Avg' $OUTDIR/hyp-words.txt.sys
echo "LM WER:"
grep 'Sum/Avg' $OUTDIR/hyp-lm-words.txt.sys

