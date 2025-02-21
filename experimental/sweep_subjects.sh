output_prefix="sweep_3a_subj_"

for x in {0..23}; do
    python fit_single_v1.py --subject_id "$x" --max_iter 50 --file_name_prefix "$output_prefix" > "${output_prefix}${x}.log" 2>&1 &
done