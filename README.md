# BEAR - Bootstrap and Attribute Ranking

## Directory Structure

|-- inputs
|   `-- test
|       |-- DILI_a375_set1.csv
|       `-- multi_data.csv
|-- outputs
|   |-- DILI_a375_set1
|   `-- multi_data
|-- pipe_step_2_FS
|   |-- pipe_step_2_FS.sh
|   |-- pipe_step_2_input
|   |   `-- input.csv
|   |-- pipe_step_2_output
|   |   |-- correlation_ranking.csv
|   |   |-- info_gain_ranking.csv
|   |   |-- info_gain_ratio_ranking.csv
|   |   |-- mrmr_ranking.csv
|   |   |-- relief_ranking.csv
|   |   `-- sym_uncertainty_ranking.csv
|   `-- pipe_step_2_scripts
|       |-- Correlation_Fselection.py
|       |-- Inforgain_Fselection.py
|       |-- InformationGainRatio_Fselection.py
|       |-- MRMR_Fselection.py
|       |-- Relief_Fselection.py
|       `-- SymmetricalUncert_Fselection.py
|-- pipe_step_3_FAggregation
|   |-- pipe_step_3_FA.sh
|   |-- pipe_step_3_Fselected_input
|   |   |-- correlation_ranking.csv
|   |   |-- info_gain_ranking.csv
|   |   |-- info_gain_ratio_ranking.csv
|   |   |-- mrmr_ranking.csv
|   |   |-- relief_ranking.csv
|   |   `-- sym_uncertainty_ranking.csv
|   |-- pipe_step_3_make_ensemble
|   |   |-- ensemble_output
|   |   |   |-- Ensemble.csv
|   |   |   `-- Feature_Weights.csv
|   |   `-- ensemble_weighting.py
|   `-- pipe_step_3_make_venn
|       |-- create_venn.py
|       `-- output_vennDiagram
|           |-- VennDiagram.png
|           `-- feature_set_arrangement.txt
|-- pipe_step_4_clf
|   |-- pipe_step_4_clf.sh
|   |-- pipe_step_4_clf_clfers
|   |   |-- NB
|   |   |   |-- Ensemble.csv
|   |   |   |-- NB.py
|   |   |   |-- NB1.py
|   |   |   |-- correlation_ranking.csv
|   |   |   |-- correlation_ranking_auc_vs_features.png
|   |   |   |-- info_gain_ranking.csv
|   |   |   |-- info_gain_ratio_ranking.csv
|   |   |   |-- mrmr_ranking.csv
|   |   |   |-- naive_bayes.log
|   |   |   |-- relief_ranking.csv
|   |   |   `-- sym_uncertainty_ranking.csv
|   |   |-- RF
|   |   |   |-- Ensemble.csv
|   |   |   |-- RF.py
|   |   |   |-- correlation_ranking.csv
|   |   |   |-- info_gain_ranking.csv
|   |   |   |-- info_gain_ratio_ranking.csv
|   |   |   |-- mrmr_ranking.csv
|   |   |   |-- random_forest.log
|   |   |   |-- relief_ranking.csv
|   |   |   `-- sym_uncertainty_ranking.csv
|   |   |-- SVM
|   |   |   |-- Ensemble.csv
|   |   |   |-- SVM.py
|   |   |   |-- correlation_ranking.csv
|   |   |   |-- info_gain_ranking.csv
|   |   |   |-- info_gain_ratio_ranking.csv
|   |   |   |-- mrmr_ranking.csv
|   |   |   |-- relief_ranking.csv
|   |   |   |-- svm.log
|   |   |   `-- sym_uncertainty_ranking.csv
|   |   `-- preprocess.py
|   |-- result_auc_for_each_position
|   |   `-- ComplementSVM_AUC.csv
|   `-- result_classifier_evaluations
|-- pipeline_trial.job
`-- run_pipeline.sh


## Description
run_pipeline.sh is the main command that runs the pipeline using bash scripts
