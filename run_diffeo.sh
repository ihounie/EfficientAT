for model_name in "mn04_as" "mn05_as" "mn10_as" "mn20_as" "mn30_as" "mn40_as" "mn40_as_ext"
do
    python ex_audioset_diffeo.py --cuda --model_name=$model_name --axis 0 --cutoff 20 --disp 2
    python ex_audioset_diffeo.py --cuda --model_name=$model_name --axis 1 --cutoff 5 --disp 1
done