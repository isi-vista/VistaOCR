 #!/bin/bash

source ~/.bashrc


#model_path=/nfs/isicvlnas01/users/srawls/ocr-dev/tmp-fake-model-best_model.pth
#datadir=/nfs/isicvlnas01/users/srawls/ocr-dev/data/iam
#lmpath=/nfs/isicvlnas01/users/jmathai/experiments/iam_lm_augment_more_data/IAM-LM/
#lmpath = /nfs/isicvlnas01/users/jschmidt/aida-lm/ocr-eesen-lm/AIDA-LM-SCRATCH

export PYTHONPATH=/nas/home/srawls/ocr/PyTorchOCR/eesen/:$PYTHONPATH


OUTDIR=./hyp-output
mkdir -p ${OUTDIR}


SCLITE=/nfs/isicvlnas01/share/sclite/sclite
script_dir="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

python removepunctuation.py hyp-output/ref-chars.txt > hyp-output/ref-chars-np.txt
python removepunctuation.py hyp-output/hyp-chars.txt > hyp-output/hyp-chars-np.txt
python removepunctuation.py hyp-output/hyp-lm-chars.txt > hyp-output/hyp-lm-chars-np.txt

# Turn decoding output into tokenized words
python ${script_dir}/chars_to_tokenized_words.py $OUTDIR/hyp-chars-np.txt $OUTDIR/hyp-words.txt
python ${script_dir}/chars_to_tokenized_words.py $OUTDIR/hyp-lm-chars-np.txt $OUTDIR/hyp-lm-words.txt
python ${script_dir}/chars_to_tokenized_words.py $OUTDIR/ref-chars-np.txt $OUTDIR/ref-words.txt
#python ${script_dir}/chars_to_tokenized_words.py $OUTDIR/font-hyp-chars-np.txt $OUTDIR/font-hyp-words.txt
#python ${script_dir}/chars_to_tokenized_words.py $OUTDIR/font-hyp-lm-chars-np.txt $OUTDIR/font-hyp-lm-words.txt
#python ${script_dir}/chars_to_tokenized_words.py $OUTDIR/font-ref-chars-np.txt $OUTDIR/font-ref-words.txt



# Do CER measurement
${SCLITE} -r $OUTDIR/ref-chars-np.txt -h $OUTDIR/hyp-chars-np.txt -i swb -o all >/dev/null
${SCLITE} -r $OUTDIR/ref-chars-np.txt -h $OUTDIR/hyp-lm-chars-np.txt -i swb -o all >/dev/null
#${SCLITE} -r $OUTDIR/font-ref-chars.txt -h $OUTDIR/font-hyp-chars.txt -i swb -o all >/dev/null
#${SCLITE} -r $OUTDIR/font-ref-chars.txt -h $OUTDIR/font-hyp-lm-chars.txt -i swb -o all >/dev/null


# Do WER measurement
${SCLITE} -r $OUTDIR/ref-words.txt -h $OUTDIR/hyp-words.txt -i swb -o all >/dev/null
${SCLITE} -r $OUTDIR/ref-words.txt -h $OUTDIR/hyp-lm-words.txt -i swb -o all >/dev/null
#${SCLITE} -r $OUTDIR/font-ref-words.txt -h $OUTDIR/font-hyp-words.txt -i swb -o all >/dev/null
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

