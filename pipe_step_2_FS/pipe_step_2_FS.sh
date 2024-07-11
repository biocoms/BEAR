python3 pipe_step_2_scripts/Correlation_Fselection.py pipe_step_2_input/input.csv pipe_step_2_output/
python3 pipe_step_2_scripts/Inforgain_Fselection.py pipe_step_2_input/input.csv pipe_step_2_output/
python3 pipe_step_2_scripts/InformationGainRatio_Fselection.py pipe_step_2_input/input.csv pipe_step_2_output/
python3 pipe_step_2_scripts/Relief_Fselection.py pipe_step_2_input/input.csv pipe_step_2_output/
python3 pipe_step_2_scripts/SymmetricalUncert_Fselection.py pipe_step_2_input/input.csv pipe_step_2_output/
python3 pipe_step_2_scripts/MRMR_Fselection.py pipe_step_2_input/input.csv pipe_step_2_output/

scp pipe_step_2_output/*.csv ../pipe_step_4_clf/pipe_step_4_clf_clfers/NB
scp pipe_step_2_output/*.csv ../pipe_step_4_clf/pipe_step_4_clf_clfers/SVM
scp pipe_step_2_output/*.csv ../pipe_step_4_clf/pipe_step_4_clf_clfers/RF
scp pipe_step_2_output/*.csv ../pipe_step_3_FAggregation/pipe_step_3_Fselected_input/
