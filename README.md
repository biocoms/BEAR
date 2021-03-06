# BEAR
=======================================================


BEAR (Bootstrap and Ensemble Attribute Ranking)

=======================================================

The BEAR program takes an input data file with attributes (features) and class labels, and applies ensemble and bootstrap strategies to select and evaluate discriminative features with respect to the class labels. Features are ranked first using five base feature selection methods (Person's correlatioin, Information Gain, Information Gain Ratio, Relief, Symmetrical Uncertainity). Then, an ensemble method is used to aggregate five base feature sets to obtain an ensemble feature set, which is evaluated using three classifiers including Naive Bayes (NB), Support Vector Machine (SVM), and Random Forest (RF). If the BEAR is used for predicting, the outcome is an ensemble prediction of the results of three classifiers. The BEAR can handle large datasets using a bootsrapping strategy. User has option to perform bootsrapping for either attributes or samples.




Installation Instructions
__________________________
1. Following are the dependencies of BEAR tool, please installl them first.
	
	1. javabridge 1.0.18
	
		Installation instructions: https://fracpete.github.io/python-weka-wrapper/install.html
	
	2. matplotlob 3.1.3
	
		Installation instructions: https://matplotlib.org/users/installing.html
	
	
	3. matplotlib-venn 0.11.5
	
		Installation instructions: https://pypi.org/project/matplotlib-venn/
		
		or https://anaconda.org/conda-forge/matplotlib-venn
		
		
	4. numpy 1.16.3
	
		Installation instructions: https://scipy.org/install.html
		
		or https://anaconda.org/anaconda/numpy
		
		
	5. pandas 1.0.1
	
		Installation instructions: https://pandas.pydata.org/pandas-docs/stable/getting_started/install.html
	
	
	6. python-weka-wrapper 0.1.12
	
		Installation instructions: https://pypi.org/project/python-weka-wrapper/
		
		or https://fracpete.github.io/python-weka-wrapper/install.html
		
		
	7. scikit-learn 0.20.3
	
		Installation instructions: https://scikit-learn.org/stable/install.html
	
	
	8. scipy 1.2.1
		
		Installation instructions: https://www.scipy.org/install.html
		
		
	9. venn 0.1.3
		
		Installation instructions: https://pypi.org/project/venn/

2. Download the BEAR pipeline as a zipped file (BEAR-master.zip) and unzip it to your machine.
	
	```
	unzip BEAR-master.zip
	```
	
3. Access the BEAR-master folder and run Install.sh. 
	
	```
	cd ./BEAR-master 

	```
	Make the Install.sh file an executable file. You may have to execute these as sudo commands.
	```
	sudo chmod +x Install.sh
	
	bash ./Install.sh
	```
	or
	
	```
	sudo ./Install.sh
	```
	After runnung Install.sh script, you may need to edit the **configuration file**. Configuration file is located in the folder configuration in the BEAR-master working folder. It is located as,
	
		*./configuration/configuration.py*
	You may have to edit the file to include an alternative Weka home directory. the defulat configuration file contain the following.
	
		weka_location_in_anaconda="/anaconda3/lib/python3.7/site-packages/weka/"
		
		Depending on your weka installing, you may have to modify the path accordingly. to modify,
		
		nano ./configuration/configuration.py
		
		After saving the file, you can start running the program. Each program will place the configuration file in the relevent folder (./pipe_step_2_FS/pipe_step_2_scripts/) as the first thing. You can manually check and place this file in relevent location to make sure if you wish. 
		
 *Set your Python Interpreter.*
The default location for python interpreter for these scripts is 

		/usr/local/bin/python3.7

Depending on your python installation and dependencies, this location can vary. Because of that, you must use changer.sh script to set the python path correctly. changer.sh script should strictly be placed in BEAR-master working folder. Do not mess with this script.
	First, make the changer.sh executable file as follows.
	
			sudo chmod +x changer.sh
			
  Next, you should find the location of correct python interpretor. And, pass that location as an argument to the changer.sh in the following way. Let's assume that you had used anaconda package manager to install the dependencies of BEAR. Your working python interpreter could be located as /anaconda3/bin/python.
  e.g.,
  		bash ./changer.sh /anaconda3/bin/python

If you accidentally make a mistake when entering the python location, you will have to re-install the whole thing.

4. Place your own data to be processed in the following folder.

	Foldername:  input_file
	
	
---input data format--

The input data is essentially a comma delimited (csv) file containing features and class labels. The last column should be the class label. The header of last column should be strictly "class". Currently, only binary class labels are allowed. It is essential that class labels to be strings. All other columns are features aka attributes and their header are feature names. Identical feature names are not allowed. The values for features should be numeric. Since we are using Naive Bayes classifier, it is important that values are non-negative.

A sample input data file "Randomized.iris.data.2.class.csv" can be found in "input_file" folder.

Note that, user has option to run BEAR with or without bootstrapping. 

	We have created an interactive script to walk you through the first steps of each of the two options. 
	This will give the user a substantial idea about the arguments being passed into the scripts at the very begining. 
	To start the interactive mode, run "Start_interactive_mode.sh" bash script. 
	This script will interact with user to get the parameters and file selection.
	
	bash ./Start_interactive_mode.sh

**Option 1: run BEAR without bootstrapping**

5. Run the "run_step_1_file_processing_withoutB.sh" bash along with an input file as the first argument. This step will copy the input file into necessary processing folders. It takes the input csv file as its first command line argument.
Here is an example command to run preprocessing on sample input file "Randomized.iris.data.2.class.csv":

```
bash ./run_step_1_file_processing_withoutB.sh nput_file/Randomized.iris.data.2.class.csv
```
 
 6. Run the "run_step_2_FeatureSelection.sh" bash
   and then perform feature selection. Five base feature selection methods are used, including 1. Pearson's correlation, 2. Information Gain, 3. Information Gain Ratio, 4. Relief, and 5. Symmetrical Uncertainity. The output is new reordered datasets according to feature rankings. The output is available in folder "pipe_step_2_FS/pipe_step_2_output". This step prepares required files for further processing by copying the reordered output files into other necessary folders.
   
```
bash ./run_step_2_FeatureSelection.sh
```

Output files: 
   Location -> ./pipe_step_2_FS/pipe_step_2_output/
 
   Description: 
	
   The outcomes are five csv files each with features re-ordered according to feature ranksings (1. Pearson's correlation, 2. Information Gain, 3. Information Gain Ratio, 4. Relief, and 5. Symmetrical Uncertainity). The final column of the csv file contains the class labels.
	
	** Features in csv file are sorted with a decreasing order of rankings based on feature selection method.
	We will keep this format throughout the pipeline. **
	

7. Run the "run_step_3_vennDiFeAEns_without_bootstrapping.sh".
   This script allows user to pick top n features for each base feature selection method by specifying a numeric argument. 
   Then, five sets of top n features will be used for Venn diagrams, feature aggregation, and feature ensemble.
   
   For example, following script picks top 30 ranked features:
   
   	bash ./run_step_3_vennDiFeAEns_without_bootstrapping.sh 30
	
Output files: 

A. Location: --> ./pipe_step_3_FAggregation/pipe_step_3_make_venn/output_vennDiagram/
	
   Description: 
		
   There is a PDF file recording a 5-way Venn diagram and, 
   a text file containing features belonging to the different parts of the Venn diagram.
			
B. Location: --> ./pipe_step_3_FAggregation/pipe_step_3_make_aggregates/
	
Description: 
	This output location contains results of five different feature aggregates: 1). at_Least.1.csv, 2). at_Least.2.csv, 3).at_Least.3.csv, 4). at_Least.4.csv, 5).at_Least.5.csv.
			The file at_Least.1.csv is same as the union of features.
			The file at_Least.2.csv contains all features that are present in at least 2 feature selection methods.
   			The file at_Least.3.csv contains all features that are present in at least 3 feature selection methods.
			The file at_Least.4.csv contains all features that are present in at least 4 feature selection methods.
			The file at_Least.5.csv the same as the intersection of features.
			
			
 C. Location --> ./pipe_step_3_FAggregation/pipe_step_3_make_ensemble/ensemble_output/
		
   Description: 
		
   There are two files in this folder. One is a feature ensemble csv file. The other is a csv file with ensemble scores. Feature ensemble is created using a feature ensemble scoring function based on the rankings of five base feature selection methods. A feature ranked higher by multiple methods will receive a higher ensemble score. 
   
8. Run "run_step_4_clfEvaluation_without_bootstrapping.sh". This step evaluates all selected feature sets from step 6, which is obtained based on the classification performance measured by the area under curve (AUC) value of the classification ROC curve. Three classifiers (NB, SVM, and RF) with default parameter sets are used. More advanced users can modify the parameter sets of these classifiers by editing their corresponding python scripts. Depending on the dataset and the parameters in the classifiers, this step can take longer to complete.

Here is a sample code for stet 7:

	bash ./run_step_4_clfEvaluation_without_bootstrapping.sh
	
Output files: 
A. Location: --> ./pipe_step_4_clf/result_classifier_evalutions/
		
   Description: 
		
   This folder contains three PDF files. They are graphs of AUC values (Y axis) progressively calculated using features over the sorted ranks (X axis).
		
B. Location: --> ./pipe_step_4_clf/result_auc_for_each_position/
		
   Description: 
		
   This folder contains a text file with file names and AUC values. 
		
	

9. Run "run_step_5_barplots_without_bootstrapping.sh" to generate bar plots of selected AUC values. When this step is executed, the program searches for the folder where it saves the classifier evaulation results to retrieve the AUC values of ROC curves. 

		bash ./run_step_5_barplots_without_bootstrapping.sh

Output files: 
	Location: --> ./pipe_step_4_clf/result_bar_plots/
	Description: This folder contains PDF files. They are bar graphs showing performances of ranked features, feature aggregates, and the featuer ensemble. X axis is feature aggregation methods and Y axis is the AUC values.
	
*Output file locations*
*=====================*

Ranked Feature input files
./pipe_step_2_FS/pipe_step_2_output/

Venn Diagrams and Ranked Feature Combinations
./pipe_step_3_FAggregation/pipe_step_3_make_venn/output_vennDiagram/

Feature Ensemble Generated Using Scoring Function
./pipe_step_3_FAggregation/pipe_step_3_make_ensemble/ensemble_output/

Ranked Feature Combinations
./pipe_step_3_FAggregation/pipe_step_3_make_aggregates/



Once finising step 9, **Running bash ./run_step_0_clean.sh** to remove the results in the folder (Recomended when user has copied all the results to his/her own folders and is ready to perform a new analysis).


**Option 2: run BEAR with bootstrapping**

1. Run run_bootstrapping_substep_1.sh. This script takes 4 commandline arguments. 

   argument 1: input csv file.
   
   argument 2: fraction of the features to be used. This should strictly be a value between 0 and 1. It can be 1 as well. 
   
   argument 3: number of bootstrap samples (sample size) to draw from data file. This value should strictly be integer.
   
   argument 4: this argument should be specified as yes or no. Here, user has the chance to specify the positive class (yes). 
   
   e.g., 
   
   	```
	bash ./run_bootstrapping_substep_1.sh input_file/Randomized.iris.data.2.class.csv 0.3 10 yes
	```
	
	During execution of the script, somee parameters used will be shown.
	
	1. Number of features in input file.
	2. Number of samples in input file.
	3. User-specified sampling fraction.
	4. User-specified number of bootstrap samples.
	5. The class labels.
	6. positive class label (indicated as 1).
	7. Table of samples with feature aggregated from bootstrapped samples. This table contains bootstrapped samples with information about number of features and percentage when compared to total number of features.
		
	
2. Run ./run_bootstrapping_substep_2.sh. The script will prompt user an menu of enumerated files of bootstrapped samples. User has to select a file from the table. This enable script to copy the chosen file to approparite folder for processing further.

e.g., 
	```
	bash ./run_bootstrapping_substep_2.sh
	```
	
3. Run the feature selection on bootstrapped samples. Please follow the Option 1 (run BEAR without bootstrapping) pipeline from step 6 (6. Run the script "run_step_2_FeatureSelection.sh") to the end (step 9).
