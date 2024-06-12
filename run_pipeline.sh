#cleaning any previous work
echo "Cleaning any previous files"
rm -f -v pipe_step_2_FS/pipe_step_2_output/*csv
rm -f -v pipe_step_3_FAggregation/pipe_step_3_Fselected_input/*csv
rm -f -v pipe_step_3_FAggregation/pipe_step_3_make_venn/output_vennDiagram/*pdf
rm -f -v pipe_step_3_FAggregation/pipe_step_3_make_venn/output_vennDiagram/*txt
rm -f -v pipe_step_3_FAggregation/pipe_step_3_make_ensemble/ensemble_output/*csv
rm -f -v pipe_step_3_FAggregation/pipe_step_3_make_aggregates/*csv
rm -f -v pipe_step_4_clf/pipe_step_4_clf_clfers/NB/*csv
rm -f -v pipe_step_4_clf/pipe_step_4_clf_clfers/NB/*pdf
rm -f -v pipe_step_4_clf/pipe_step_4_clf_clfers/NB/*txt
rm -f -v pipe_step_4_clf/pipe_step_4_clf_clfers/SVM/*csv
rm -f -v pipe_step_4_clf/pipe_step_4_clf_clfers/SVM/*pdf
rm -f -v pipe_step_4_clf/pipe_step_4_clf_clfers/SVM/*txt
rm -f -v pipe_step_4_clf/pipe_step_4_clf_clfers/RF/*csv
rm -f -v pipe_step_4_clf/pipe_step_4_clf_clfers/RF/*pdf
rm -f -v pipe_step_4_clf/pipe_step_4_clf_clfers/RF/*txt
rm -f -v pipe_step_4_clf/result_auc_for_each_position/*csv
rm -f -v pipe_step_4_clf/result_bar_plots/*pdf
rm -f -v pipe_step_4_clf/result_classifier_evaluations/*png
echo " "
echo "Cleaning any previous files completed."

# Directory containing the input files, passed as first argument
input_dir=$1

# Additional argument, might be used in one of the steps, passed as second argument
number_of_features=$2

# Loop over each CSV file in the input directory
for input_file in ${input_dir}*.csv; do
    file_name=$(basename "$input_file" .csv)  # Extracts the file name without extension

    # Create directories for outputs corresponding to this file
    mkdir -p "outputs/${file_name}/pipe_step_2_FS"
    mkdir -p "outputs/${file_name}/pipe_step_3_FAggregation"
    mkdir -p "outputs/${file_name}/pipe_step_4_clf"
    mkdir -p "outputs/${file_name}/pipe_step_4_clf/NB/"
    mkdir -p "outputs/${file_name}/pipe_step_4_clf/RF/"
    mkdir -p "outputs/${file_name}/pipe_step_4_clf/SVM/"
    mkdir -p "outputs/${file_name}/pipe_step_4_clf/results/"
    
    # Copy input file to the needed location
    cp "$input_file" "pipe_step_2_FS/pipe_step_2_input/input.csv"
    cp "$input_file" "pipe_step_4_clf/pipe_step_4_clf_clfers/NB/Randomized.csv"
    cp "$input_file" "pipe_step_4_clf/pipe_step_4_clf_clfers/RF/Randomized.csv"
    cp "$input_file" "pipe_step_4_clf/pipe_step_4_clf_clfers/SVM/Randomized.csv"

    # Execute scripts for each step, directing outputs to the specific directory
    echo ""
    echo "Feature Selection Started."
    echo " "
    cd pipe_step_2_FS/
    bash pipe_step_2_FS.sh
    echo " "
    echo "Feature Selection Complete"
    cd ..
    cp -r pipe_step_2_FS/pipe_step_2_output/ "outputs/${file_name}/pipe_step_2_FS/"
    

    cd pipe_step_3_FAggregation/
    bash pipe_step_3_FA.sh $number_of_features
    cd ..
    cp -r pipe_step_3_FAggregation/pipe_step_3_make_venn/output_vennDiagram/ "outputs/${file_name}/pipe_step_3_FAggregation/"
    cp -r pipe_step_3_FAggregation/pipe_step_3_make_aggregates/ "outputs/${file_name}/pipe_step_3_FAggregation/"
    cp -r pipe_step_3_FAggregation/pipe_step_3_make_ensemble/ensemble_output/ "outputs/${file_name}/pipe_step_3_FAggregation/"

    cd pipe_step_4_clf/
    bash pipe_step_4_clf.sh
    cd ..
    cp pipe_step_4_clf/pipe_step_4_clf_clfers/NB/*.csv "outputs/${file_name}/pipe_step_4_clf/NB/"
    cp pipe_step_4_clf/pipe_step_4_clf_clfers/NB/*.log "outputs/${file_name}/pipe_step_4_clf/NB/"
    cp pipe_step_4_clf/pipe_step_4_clf_clfers/RF/*.csv "outputs/${file_name}/pipe_step_4_clf/RF/"
    cp pipe_step_4_clf/pipe_step_4_clf_clfers/RF/*.log "outputs/${file_name}/pipe_step_4_clf/RF/"
    cp pipe_step_4_clf/pipe_step_4_clf_clfers/SVM/*.csv "outputs/${file_name}/pipe_step_4_clf/SVM/"
    cp pipe_step_4_clf/pipe_step_4_clf_clfers/SVM/*.log "outputs/${file_name}/pipe_step_4_clf/SVM/"
    cp -r pipe_step_4_clf/result_auc_for_each_position/ "outputs/${file_name}/pipe_step_4_clf/results/"
    cp -r pipe_step_4_clf/result_classifier_evaluations/ "outputs/${file_name}/pipe_step_4_clf/results/"

    echo "Processing completed for ${file_name}"
done


echo " "
echo "=========log output============"
echo ""
echo "Results saved, File processing Complete."
echo "---------------------------------------"
