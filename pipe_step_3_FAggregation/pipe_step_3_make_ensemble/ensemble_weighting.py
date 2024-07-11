from matplotlib import pyplot as plt
from  matplotlib_venn import venn2
import pandas as pd
from os import getcwd,chdir
from glob import glob
from itertools import combinations

def safe_indexing(feature, feature_list, scoring_list):
    try:
        # If the feature is in the list, return its score
        return scoring_list[feature_list.index(feature)]
    except ValueError:
        # If the feature is not in the list, return 0
        return 0



# thre cutoff needs to be the same value
print("Ensemble Creation Started.")
Information_Gain = pd.read_csv('../pipe_step_3_Fselected_input/info_gain_ranking.csv')
Correlation = pd.read_csv('../pipe_step_3_Fselected_input/correlation_ranking.csv')
Information_Gain_ratio = pd.read_csv('../pipe_step_3_Fselected_input/info_gain_ratio_ranking.csv')
Relief = pd.read_csv('../pipe_step_3_Fselected_input/relief_ranking.csv')
Symmetrical_Uncertainity = pd.read_csv('../pipe_step_3_Fselected_input/sym_uncertainty_ranking.csv')

default_val = Information_Gain.shape[1]-1
cutoff =  default_val #default_val
# get header names without the class label
# and they need to be lists
Information_Gain_List=list(Information_Gain.columns.values[0:cutoff])
Correlation_List=list(Correlation.columns.values[0:cutoff])
Information_Gain_ratio_List=list(Information_Gain_ratio.columns.values[0:cutoff])
Relief_List=list(Relief.columns.values[0:cutoff])
Information_Gain_ratio_List=list(Information_Gain_ratio.columns.values[0:cutoff])
Symmetrical_Uncertainity_List=list(Symmetrical_Uncertainity.columns.values[0:cutoff])

# creating scoreing function
# oth element needs to be zero for convenience
scoring_list = []
for f in range(len(Information_Gain_List),-1,-1):
    #print(f)
    scoring_list.append(f)

#print("give following score for items that tops the list")
#print(scoring_list[4])
# Get the score of following one
# consider the feature f1
#
#df_ = pd.DataFrame(columns=Information_Gain_List)
#print("here is the zeroth element's position")

#print("creating the ensemble list")
ensemble_list = Symmetrical_Uncertainity_List
ensemble_sum_list = []
for item in ensemble_list:
    #print("the item is")
    #print(item)
    #print(Information_Gain_List.index(item))
    sum = safe_indexing(item, Information_Gain_List, scoring_list) + safe_indexing(item, Information_Gain_ratio_List, scoring_list) + safe_indexing(item, Relief_List, scoring_list) + safe_indexing(item, Correlation_List, scoring_list) + safe_indexing(item, Symmetrical_Uncertainity_List, scoring_list)
    #print("===")
    #print(Information_Gain_ratio_List.index(item))
    #print(Relief_List.index(item))
    #print(Correlation_List.index(item))
    #print(Symmetrical_Uncertainity_List.index(item))
    sum = safe_indexing(item, Information_Gain_List, scoring_list) + safe_indexing(item, Information_Gain_ratio_List, scoring_list) + safe_indexing(item, Relief_List, scoring_list) + safe_indexing(item, Correlation_List, scoring_list) + safe_indexing(item, Symmetrical_Uncertainity_List, scoring_list)
    #print("sum for " + item + " is "+str(sum))
    #print("appending the sum to ensemble_sum_list")
    ensemble_sum_list.append(sum)

# create pandas data frame of scores
score_df = pd.DataFrame(list(zip(ensemble_list,ensemble_sum_list)))
# add column names
score_df.columns=["Features","score"]
#print(score_df)
# sort dataframe by second column
sorted_df =score_df.sort_values(by=['score'],ascending=False)
sorted_df.to_csv("./ensemble_output/Feature_Weights.csv", sep=',',index=False,header=True)
#print(sorted_df)


# getting features column into a list from pandas dataframe
ensemble_feature_list = sorted_df['Features'].tolist()
#print(ensemble_feature_list)
ensemble_feature_list.append("class")
transposed = Information_Gain.T
reversed = transposed.loc[ensemble_feature_list].T
#print(reversed)
reversed.to_csv("./ensemble_output/Ensemble.csv", sep=',',index=False,header=True)
print("Ensemble Creation Complete.")
#print()



##print(Information_Gain_List[0])

# permanent_dentition_DP2 = dict()
# #print(Information_Gain_List)
# # Iterate through each row of the data list.
# # Save each row in dictionary
# for item in Information_Gain_List:
#     #print(item)
#     ##print(row[0])
#     #permanent_dentition_DP2[row[0]] = row
#     ##print("*****")