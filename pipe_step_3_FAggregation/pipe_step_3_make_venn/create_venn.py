from matplotlib import pyplot as plt
from  matplotlib_venn import venn2
import pandas as pd
from os import getcwd,chdir
from glob import glob
from itertools import combinations
import sys




# thre cutoff needs to be the same value



print("Venn Diagram Creation and Feature Aggregate Creation Started.")
default_val = sys.argv[1]
# The argument is taken as string
# it needs to be converted to integer to be used as a slicing cutoff
cutoff = int(default_val)
print("You assigned top " +str(default_val) +" features to be investigated.")
#print("To change this setting, change the default_val variable in the script of ./pipe_step_3_FAggregation/pipe_step_3_make_venn/create_venn.py")

Information_Gain = pd.read_csv('../pipe_step_3_Fselected_input/info_gain_ranking.csv')
Correlation = pd.read_csv('../pipe_step_3_Fselected_input/correlation_ranking.csv')
Information_Gain_ratio = pd.read_csv('../pipe_step_3_Fselected_input/info_gain_ratio_ranking.csv')
Relief = pd.read_csv('../pipe_step_3_Fselected_input/relief_ranking.csv')
Symmetrical_Uncertainty = pd.read_csv('../pipe_step_3_Fselected_input/sym_uncertainty_ranking.csv')
MRMR = pd.read_csv('../pipe_step_3_Fselected_input/mrmr_ranking.csv')



# get header names without the class label
# and they need to be sets
Information_Gain_List=set(Information_Gain.columns.values[0:cutoff])
Correlation_List=set(Correlation.columns.values[0:cutoff])
Information_Gain_ratio_List=set(Information_Gain_ratio.columns.values[0:cutoff])
Relief_List=set(Relief.columns.values[0:cutoff])
Information_Gain_ratio_List=set(Information_Gain_ratio.columns.values[0:cutoff])
Symmetrical_Uncertainty_List=set(Symmetrical_Uncertainty.columns.values[0:cutoff])
MRMR_List=set(MRMR.columns.values[0:cutoff])

# Information_Gain_List=set(Information_Gain.columns.values[0:Information_Gain.shape[1]-1])
# Correlation_List=set(Correlation.columns.values[0:Correlation.shape[1]-1])
# Information_Gain_ratio_List=set(Information_Gain_ratio.columns.values[0:Information_Gain_ratio.shape[1]-1])
# Relief_List=set(Relief.columns.values[0:Relief.shape[1]-1])
# Information_Gain_ratio_List=set(Information_Gain_ratio.columns.values[0:Information_Gain_ratio.shape[1]-1])
# Symmetrical_Uncertainty_List=set(Symmetrical_Uncertainty.columns.values[0:Symmetrical_Uncertainty.shape[1]-1])
# ##print(Correlation.columns.values[0:Correlation.shape[1]-1])
##print(Information_Gain_ratio.columns.values[0:Information_Gain_ratio.shape[1]-1])

# dictionary is needed for venn diagram creation
venn_dictionary = dict()
venn_dictionary["Information_Gain"] = Information_Gain_List
#print("informationGain: "+ str(Information_Gain_List))
venn_dictionary["Correlation"] = Correlation_List
#print("Correlation: "+ str(Correlation_List))
venn_dictionary["Information_Gain_ratio"] = Information_Gain_ratio_List
#print("Information_Gain_ratio: "+ str(Information_Gain_ratio_List))
venn_dictionary["Relief"] = Relief_List
#print("Relief: "+ str(Relief_List))
venn_dictionary["Symmetrical_Uncertainty"] = Symmetrical_Uncertainty_List
#print("Symmetrical_Uncertainty: "+ str(Symmetrical_Uncertainty_List))
venn_dictionary["MRMR"] = MRMR_List

#venn2([set(['A', 'B', 'C', 'D']), set(['D', 'E', 'F'])])
##print(venn_dictionary)
# shows all possible combinations of intersections
sets=[Information_Gain_List,Correlation_List,Information_Gain_ratio_List,Relief_List,Symmetrical_Uncertainty_List,MRMR_List]
##print(len(sets))
from venn import venn
import pandas as pd
f_arrangement_file = open("./output_vennDiagram/feature_set_arrangement.txt", "a")
for comboSize in range(1,len(sets)+1):
    ##print("combosize is "+str(comboSize))
    tempset = list()
    unitedtempset = list()
    for combo in combinations(range(len(sets)),comboSize):
        intersection = sets[combo[0]]
        for i in combo[1:]: intersection = intersection & sets[i]
        statement =" and ".join(f"Set{i+1}" for i in combo),"=",intersection
        #print(statement)
        #f_arrangement_file.write(statement)
        f_arrangement_file.write(' '.join(str(s) for s in statement) + '\n')
        ##print("====")
        unitedtempset.append(list(intersection))
    #print("combosize is " + str(comboSize))
    ##print(unitedtempset)
    # make a flat list from list of lists
    flat_list = [item for sublist in unitedtempset for item in sublist]
    ##print(flat_list)
    ##print("get only uniques of flat list")
    flat_list_with_uniques=list(set(flat_list))
    #print("following is flat_list_with_uniques")
    flat_list_with_uniques.append("class")
    #print(flat_list_with_uniques)
    transposed = Information_Gain.T
    reversed = transposed.loc[flat_list_with_uniques].T
    #print(reversed)


    #need to transpose the list to be saved in csv file
    #df_list=pd.DataFrame(flat_list_with_uniques)
    ##print(df_list)
    #reversed.to_csv("../pipe_step_3_make_aggregates/at_Least.%s.csv" % comboSize, sep=',',index=False,header=True)

#This was a very good example of dictionary of sets
#this is what we need
#musicians = {
#    "Members of The Beatles": {"Paul McCartney", "John Lennon", "George Harrison", "Ringo Starr"},
#    "Guitarists": {"John Lennon", "George Harrison", "Jimi Hendrix", "Eric Clapton", "Carlos Santana"},
#    "Played at Woodstock": {"Jimi Hendrix", "Carlos Santana", "Keith Moon"},
#    "Played at hikka": {"miyu", "Carlos Santana", "Keith Moon"},
#    "Played at matara": {"miyu", "nadee", "Keith Moon"}
#}
venn(venn_dictionary)
plt.title('Venn Diagram - Top '+str(cutoff)+' Ranked Features')
plt.savefig("./output_vennDiagram/VennDiagram.png")
print("Venn Diagram Creation and Feature Aggregate Creation Complete.")
f_arrangement_file.close()
#plt.show()
