#!/bin/bash
# sweep_script.sh
#
# Usage:
#   sweep_script.sh --this_sweep_name <sweep_name> [--min <min_value>] [--max <max_value>] <command template>
#
# The command template should include:
#   - _sweep_var : A placeholder that will be replaced by the loop index.
#   - _sweep_name: A placeholder that will be replaced by the value provided via --this_sweep_name.
#
# For example, you might invoke the script like this:
#
#   ./sweep_script.sh --this_sweep_name sweep_conj_grad --min 0 --max 23 \
#       python --subject_id _sweep_var --max_iter 50 --file_name_prefix "_sweep_name" \
#       '>' "\"_sweep_name\${_sweep_var}.log\"" '2>&1'
#
# In this example:
# - The python argument --file_name_prefix gets the value "sweep_conj_grad" (via _sweep_name).
# - The log file name will also be prefixed with "sweep_conj_grad" and suffixed with the iteration number.
#

usage() {
    echo "Usage: $0 --this_sweep_name <sweep_name> [--min <min_value>] [--max <max_value>] <command template>"
    exit 1
}

# Ensure we have enough arguments.
if [ "$#" -lt 3 ]; then
    usage
fi

# Default values.
min_value=0
max_value=23
sweep_name=""

# Process the options.
while [[ "$1" == --* ]]; do
    case "$1" in
        --this_sweep_name)
            shift
            [ -z "$1" ] && usage
            sweep_name="$1"
            shift
            ;;
        --min)
            shift
            [ -z "$1" ] && usage
            min_value="$1"
            shift
            ;;
        --max)
            shift
            [ -z "$1" ] && usage
            max_value="$1"
            shift
            ;;
        *)
            break
            ;;
    esac
done

# The remaining arguments form the command template.
if [ "$#" -eq 0 ]; then
    usage
fi

cmd_template="$*"

# Define placeholder tokens.
ph_var="_sweep_var"
ph_name="_sweep_name"

# Loop over the defined range.
for (( i = min_value; i <= max_value; i++ )); do
    # Replace _sweep_var with the current iteration index.
    cmd="${cmd_template//$ph_var/$i}"
    # Replace _sweep_name with the provided sweep name.
    cmd="${cmd//${ph_name}/$sweep_name}"
    
    echo "Executing: $cmd"
    eval "$cmd"
done




# output_prefix="sweep_3a_subj_"

# for x in {0..23}; do
#     python fit_single_v1.py --subject_id "$x" --max_iter 50 --file_name_prefix "$output_prefix" > "${output_prefix}${x}.log" 2>&1 &
# done


#   ./sweep_subjects.sh --this_sweep_name test_steps_etc --min 3 --max 12 \
#       python --subject_id "_sweep_var" --max_iter 2 --file_name_prefix "_sweep_name" \
#       '>' "${_sweep_name}${_sweep_var}.log" '2>&1'

# MORE EXAMPLES:
#   280  bash   ./sweep_subjects.sh --min 1 --max 8 --this_sweep_name coin_evoked_run001_MAE_ python fit_single_v2.py  --param_config_id 4 --experimental_data coin --paradigm evoked   --subject_id "_sweep_var" --max_iter 10 --fitting_loss  MAE --shift_model_out --file_name_prefix "_sweep_name"  '>' '_sweep_name_sweep_var.log' '2>&1&'
#   281  bash   ./sweep_subjects.sh --min 1 --max 8 --this_sweep_name coin_evoked_run001_MAE_ python fit_single_v2.py  --param_config_id 4 --experimental_data coin --paradigm evoked   --subject_id "_sweep_var" --max_iter 10 --fitting_loss  MAE  --file_name_prefix "_sweep_name"  '>' '_sweep_name_sweep_var.log' '2>&1&'
#   282  bash   ./sweep_subjects.sh --min 1 --max 8 --this_sweep_name coin_evoked_run002_MAE_ python fit_single_v2.py  --param_config_id 4 --experimental_data coin --paradigm evoked   --subject_id "_sweep_var" --max_iter 150 --fitting_loss  MAE  --file_name_prefix "_sweep_name"  '>' '_sweep_name_sweep_var.log' '2>&1&'

# bash   ./sweep_subjects.sh --min 1 --max 2 --this_sweep_name coin_evoked_run021_MSE_ python fit_single_v2.py  --param_config_id 41 --experimental_data coin --paradigm evoked   --subject_id "_sweep_var" --max_iter 2 --fitting_loss  MSE  --file_name_prefix "_sweep_name"  '>' '_sweep_name_sweep_var.log' '2>&1&'