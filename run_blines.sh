for model_name in "mn04_as" "mn05_as" "mn10_as" "mn20_as" "mn30_as" "mn40_as" "mn40_as_ext"
do
    python ex_audioset.py --cuda --model_name=$model_name
done