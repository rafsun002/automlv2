############################################
#*********Import Basic libraries*********
#############################################
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import math


############################################
#*********Getting the dataset*********
#############################################
tmpDF=pd.read_csv("Enter the file path/ file_name.extension")
print(tmpDF)


############################################
#********Gathering some information*********
############################################
print(tmpDF.head())

print(tmpDF.tail())

print(tmpDF.shape)

print(tmpDF.columns)

print(tmpDF.info())

print(tmpDF.isnull().sum()/tmpDF.shape[0]*100)

print(tmpDF.isnull().sum())


############################################
#**********Taking all the inputs***********
############################################

target_col_name="Item_Outlet_Sales"

drop_col_nm=['none']

dateTimeColsNames=['none']

drop_nan_col_bound=float(40.0)

num_col_nm_strip=['Item_Weight', 'Item_Visibility', 'Item_MRP', 'Item_Outlet_Sales', 'Outlet_age']

num_col_nm_strip_2=['Item_Fat_Content', 'Item_Type', 'Outlet_Size', 'Outlet_Location_Type', 'Outlet_Type']

rank_col=['None']

columns_to_split=['none']

delimiters=['none']

makedecision="none"


###############################################################
#*************change into numerical data type******************
###############################################################

tmpDF["Item_Weight"] = tmpDF["Item_Weight"].astype(float)


tmpDF["Item_Visibility"] = tmpDF["Item_Visibility"].astype(float)


tmpDF["Item_MRP"] = tmpDF["Item_MRP"].astype(float)


tmpDF["Item_Outlet_Sales"] = tmpDF["Item_Outlet_Sales"].astype(float)


tmpDF["Outlet_age"] = tmpDF["Outlet_age"].astype(int)


##########################################################
#************** Filling missing values *********************
###########################################################

df=tmpDF

df=df.reset_index(drop=True)


################################################
#********* EDA for Outliers Handling************
################################################


#******** distplot for see the distribution *******
plt.subplot(2,3,1)
sns.distplot(df["Item_Weight"])
plt.subplots_adjust(left=0.0,bottom=0.0,right=2.0,top=3.0,wspace=0.4,hspace=0.4)

plt.subplot(2,3,2)
sns.distplot(df["Item_Visibility"])
plt.subplots_adjust(left=0.0,bottom=0.0,right=2.0,top=3.0,wspace=0.4,hspace=0.4)

plt.subplot(2,3,3)
sns.distplot(df["Item_MRP"])
plt.subplots_adjust(left=0.0,bottom=0.0,right=2.0,top=3.0,wspace=0.4,hspace=0.4)

plt.subplot(2,3,4)
sns.distplot(df["Outlet_age"])
plt.subplots_adjust(left=0.0,bottom=0.0,right=2.0,top=3.0,wspace=0.4,hspace=0.4)


#******** Box plot for identify outliers*******
plt.subplot(2,3,1)
sns.boxplot(df["Item_Weight"])
plt.subplots_adjust(left=0.0,bottom=0.0,right=2.0,top=3.0,wspace=0.4,hspace=0.4)

plt.subplot(2,3,2)
sns.boxplot(df["Item_Visibility"])
plt.subplots_adjust(left=0.0,bottom=0.0,right=2.0,top=3.0,wspace=0.4,hspace=0.4)

plt.subplot(2,3,3)
sns.boxplot(df["Item_MRP"])
plt.subplots_adjust(left=0.0,bottom=0.0,right=2.0,top=3.0,wspace=0.4,hspace=0.4)

plt.subplot(2,3,4)
sns.boxplot(df["Outlet_age"])
plt.subplots_adjust(left=0.0,bottom=0.0,right=2.0,top=3.0,wspace=0.4,hspace=0.4)


########################################################
#********* Removing Outliers for numerical data *********
########################################################

data=df["Item_Weight"]
data=df["Item_Visibility"]
data=df["Item_MRP"]
data=df["Outlet_age"]
row_drp_rmv_outl=[5634, 8194, 521, 6670, 6674, 532, 8215, 1560, 9240, 2586, 9756, 5150, 2081, 5154, 6179, 1575, 2088, 7215, 4656, 49, 2613, 8765, 7744, 4674, 2122, 9803, 4175, 8273, 7250, 83, 4192, 3171, 5732, 8292, 5734, 108, 1644, 7278, 1651, 4219, 2177, 6786, 3206, 1159, 9876, 5784, 8345, 8856, 5795, 9381, 3750, 5287, 680, 2728, 8875, 174, 6833, 9394, 8371, 8883, 4789, 3767, 9913, 7866, 7368, 1225, 2251, 8908, 5837, 1754, 7388, 3811, 5354, 8432, 9969, 5366, 6903, 1272, 5880, 6909, 4350, 5374, 5891, 3336, 1291, 1805, 3341, 6926, 7949, 9486, 2324, 9492, 4382, 1311, 2336, 1827, 8997, 2855, 6953, 1324, 3884, 1841, 6966, 4408, 9017, 5946, 8509, 5445, 3399, 7499, 4941, 334, 847, 2895, 854, 7005, 8542, 2401, 2403, 9066, 7030, 6008, 8569, 4987, 6012, 9600, 3458, 7558, 2439, 6536, 9607, 10123, 2445, 9617, 3474, 6547, 1941, 8599, 3993, 1434, 8601, 3488, 7072, 8612, 4006, 5031, 3497, 7081, 6576, 7088, 434, 4530, 7603, 3001, 4538, 5050, 5057, 7107, 966, 9160, 3017, 7121, 3540, 8661, 6102, 4567, 1496, 7639, 5083, 1501, 6622, 9695, 9707, 502, 6647]
df=df.drop(row_drp_rmv_outl).reset_index(drop=True)


#######################################################
#*********************** Encoding ********************
#######################################################

from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import OrdinalEncoder

allCatCol=list(df.select_dtypes(include="object").columns)

categorical_column=None

categorical_column=['Item_Fat_Content', 'Item_Type', 'Outlet_Size', 'Outlet_Location_Type', 'Outlet_Type']
df["Item_Fat_Content"]=df["Item_Fat_Content"].map({'Low Fat': 0, 'Regular': 1})


df["Outlet_Size"]=df["Outlet_Size"].map({'Medium': 0, 'High': 1, 'Small': 2})

df["Outlet_Location_Type"]=df["Outlet_Location_Type"].map({'Tier 1': 0, 'Tier 3': 1, 'Tier 2': 2})


encoding= OneHotEncoder(sparse=False)
#*********** Performing one-Hot encoding ************
result_encod=encoding.fit_transform(df[['Item_Type', 'Outlet_Type']])
dataFrame_encode_col=pd.DataFrame(result_encod,columns=['Item_Type_Baking Goods', 'Item_Type_Breads', 'Item_Type_Breakfast', 'Item_Type_Canned', 'Item_Type_Dairy', 'Item_Type_Frozen Foods', 'Item_Type_Fruits and Vegetables', 'Item_Type_Hard Drinks', 'Item_Type_Health and Hygiene', 'Item_Type_Household', 'Item_Type_Meat', 'Item_Type_Others', 'Item_Type_Seafood', 'Item_Type_Snack Foods', 'Item_Type_Soft Drinks', 'Item_Type_Starchy Foods', 'Outlet_Type_Grocery Store', 'Outlet_Type_Supermarket Type1', 'Outlet_Type_Supermarket Type10', 'Outlet_Type_Supermarket Type100', 'Outlet_Type_Supermarket Type1000', 'Outlet_Type_Supermarket Type1001', 'Outlet_Type_Supermarket Type1002', 'Outlet_Type_Supermarket Type1003', 'Outlet_Type_Supermarket Type1004', 'Outlet_Type_Supermarket Type1005', 'Outlet_Type_Supermarket Type1006', 'Outlet_Type_Supermarket Type1007', 'Outlet_Type_Supermarket Type1008', 'Outlet_Type_Supermarket Type1009', 'Outlet_Type_Supermarket Type101', 'Outlet_Type_Supermarket Type1010', 'Outlet_Type_Supermarket Type1011', 'Outlet_Type_Supermarket Type1012', 'Outlet_Type_Supermarket Type1013', 'Outlet_Type_Supermarket Type1014', 'Outlet_Type_Supermarket Type1015', 'Outlet_Type_Supermarket Type1016', 'Outlet_Type_Supermarket Type1017', 'Outlet_Type_Supermarket Type1018', 'Outlet_Type_Supermarket Type1019', 'Outlet_Type_Supermarket Type102', 'Outlet_Type_Supermarket Type1020', 'Outlet_Type_Supermarket Type1021', 'Outlet_Type_Supermarket Type1022', 'Outlet_Type_Supermarket Type1023', 'Outlet_Type_Supermarket Type1024', 'Outlet_Type_Supermarket Type1025', 'Outlet_Type_Supermarket Type1026', 'Outlet_Type_Supermarket Type1027', 'Outlet_Type_Supermarket Type1028', 'Outlet_Type_Supermarket Type1029', 'Outlet_Type_Supermarket Type103', 'Outlet_Type_Supermarket Type1030', 'Outlet_Type_Supermarket Type1031', 'Outlet_Type_Supermarket Type1032', 'Outlet_Type_Supermarket Type1033', 'Outlet_Type_Supermarket Type1034', 'Outlet_Type_Supermarket Type1035', 'Outlet_Type_Supermarket Type1036', 'Outlet_Type_Supermarket Type1037', 'Outlet_Type_Supermarket Type1038', 'Outlet_Type_Supermarket Type1039', 'Outlet_Type_Supermarket Type104', 'Outlet_Type_Supermarket Type1040', 'Outlet_Type_Supermarket Type1041', 'Outlet_Type_Supermarket Type1042', 'Outlet_Type_Supermarket Type1043', 'Outlet_Type_Supermarket Type1044', 'Outlet_Type_Supermarket Type1045', 'Outlet_Type_Supermarket Type1046', 'Outlet_Type_Supermarket Type1047', 'Outlet_Type_Supermarket Type1048', 'Outlet_Type_Supermarket Type1049', 'Outlet_Type_Supermarket Type105', 'Outlet_Type_Supermarket Type1050', 'Outlet_Type_Supermarket Type1051', 'Outlet_Type_Supermarket Type1052', 'Outlet_Type_Supermarket Type1053', 'Outlet_Type_Supermarket Type1054', 'Outlet_Type_Supermarket Type1055', 'Outlet_Type_Supermarket Type1056', 'Outlet_Type_Supermarket Type1057', 'Outlet_Type_Supermarket Type1058', 'Outlet_Type_Supermarket Type1059', 'Outlet_Type_Supermarket Type106', 'Outlet_Type_Supermarket Type1060', 'Outlet_Type_Supermarket Type1061', 'Outlet_Type_Supermarket Type1062', 'Outlet_Type_Supermarket Type1063', 'Outlet_Type_Supermarket Type1064', 'Outlet_Type_Supermarket Type1065', 'Outlet_Type_Supermarket Type1066', 'Outlet_Type_Supermarket Type1067', 'Outlet_Type_Supermarket Type1068', 'Outlet_Type_Supermarket Type1069', 'Outlet_Type_Supermarket Type107', 'Outlet_Type_Supermarket Type1071', 'Outlet_Type_Supermarket Type1072', 'Outlet_Type_Supermarket Type1073', 'Outlet_Type_Supermarket Type1074', 'Outlet_Type_Supermarket Type1075', 'Outlet_Type_Supermarket Type1077', 'Outlet_Type_Supermarket Type1078', 'Outlet_Type_Supermarket Type1079', 'Outlet_Type_Supermarket Type108', 'Outlet_Type_Supermarket Type1080', 'Outlet_Type_Supermarket Type1081', 'Outlet_Type_Supermarket Type1082', 'Outlet_Type_Supermarket Type1083', 'Outlet_Type_Supermarket Type1084', 'Outlet_Type_Supermarket Type1085', 'Outlet_Type_Supermarket Type1086', 'Outlet_Type_Supermarket Type1087', 'Outlet_Type_Supermarket Type1088', 'Outlet_Type_Supermarket Type1089', 'Outlet_Type_Supermarket Type109', 'Outlet_Type_Supermarket Type1090', 'Outlet_Type_Supermarket Type1091', 'Outlet_Type_Supermarket Type1092', 'Outlet_Type_Supermarket Type1093', 'Outlet_Type_Supermarket Type1094', 'Outlet_Type_Supermarket Type1095', 'Outlet_Type_Supermarket Type1096', 'Outlet_Type_Supermarket Type1097', 'Outlet_Type_Supermarket Type1098', 'Outlet_Type_Supermarket Type1099', 'Outlet_Type_Supermarket Type11', 'Outlet_Type_Supermarket Type110', 'Outlet_Type_Supermarket Type1100', 'Outlet_Type_Supermarket Type1101', 'Outlet_Type_Supermarket Type1102', 'Outlet_Type_Supermarket Type1103', 'Outlet_Type_Supermarket Type1104', 'Outlet_Type_Supermarket Type1105', 'Outlet_Type_Supermarket Type1106', 'Outlet_Type_Supermarket Type1107', 'Outlet_Type_Supermarket Type1108', 'Outlet_Type_Supermarket Type1109', 'Outlet_Type_Supermarket Type111', 'Outlet_Type_Supermarket Type1110', 'Outlet_Type_Supermarket Type1111', 'Outlet_Type_Supermarket Type1112', 'Outlet_Type_Supermarket Type1113', 'Outlet_Type_Supermarket Type1114', 'Outlet_Type_Supermarket Type1115', 'Outlet_Type_Supermarket Type1116', 'Outlet_Type_Supermarket Type1117', 'Outlet_Type_Supermarket Type1118', 'Outlet_Type_Supermarket Type1119', 'Outlet_Type_Supermarket Type112', 'Outlet_Type_Supermarket Type1120', 'Outlet_Type_Supermarket Type1121', 'Outlet_Type_Supermarket Type1122', 'Outlet_Type_Supermarket Type1123', 'Outlet_Type_Supermarket Type1124', 'Outlet_Type_Supermarket Type1125', 'Outlet_Type_Supermarket Type1126', 'Outlet_Type_Supermarket Type1127', 'Outlet_Type_Supermarket Type1128', 'Outlet_Type_Supermarket Type1129', 'Outlet_Type_Supermarket Type113', 'Outlet_Type_Supermarket Type114', 'Outlet_Type_Supermarket Type115', 'Outlet_Type_Supermarket Type116', 'Outlet_Type_Supermarket Type117', 'Outlet_Type_Supermarket Type118', 'Outlet_Type_Supermarket Type119', 'Outlet_Type_Supermarket Type12', 'Outlet_Type_Supermarket Type120', 'Outlet_Type_Supermarket Type121', 'Outlet_Type_Supermarket Type122', 'Outlet_Type_Supermarket Type123', 'Outlet_Type_Supermarket Type124', 'Outlet_Type_Supermarket Type125', 'Outlet_Type_Supermarket Type127', 'Outlet_Type_Supermarket Type128', 'Outlet_Type_Supermarket Type129', 'Outlet_Type_Supermarket Type13', 'Outlet_Type_Supermarket Type130', 'Outlet_Type_Supermarket Type131', 'Outlet_Type_Supermarket Type132', 'Outlet_Type_Supermarket Type133', 'Outlet_Type_Supermarket Type134', 'Outlet_Type_Supermarket Type135', 'Outlet_Type_Supermarket Type136', 'Outlet_Type_Supermarket Type137', 'Outlet_Type_Supermarket Type138', 'Outlet_Type_Supermarket Type139', 'Outlet_Type_Supermarket Type14', 'Outlet_Type_Supermarket Type140', 'Outlet_Type_Supermarket Type141', 'Outlet_Type_Supermarket Type142', 'Outlet_Type_Supermarket Type143', 'Outlet_Type_Supermarket Type144', 'Outlet_Type_Supermarket Type145', 'Outlet_Type_Supermarket Type146', 'Outlet_Type_Supermarket Type147', 'Outlet_Type_Supermarket Type148', 'Outlet_Type_Supermarket Type149', 'Outlet_Type_Supermarket Type15', 'Outlet_Type_Supermarket Type150', 'Outlet_Type_Supermarket Type151', 'Outlet_Type_Supermarket Type152', 'Outlet_Type_Supermarket Type154', 'Outlet_Type_Supermarket Type155', 'Outlet_Type_Supermarket Type156', 'Outlet_Type_Supermarket Type157', 'Outlet_Type_Supermarket Type158', 'Outlet_Type_Supermarket Type159', 'Outlet_Type_Supermarket Type160', 'Outlet_Type_Supermarket Type161', 'Outlet_Type_Supermarket Type162', 'Outlet_Type_Supermarket Type163', 'Outlet_Type_Supermarket Type164', 'Outlet_Type_Supermarket Type165', 'Outlet_Type_Supermarket Type166', 'Outlet_Type_Supermarket Type167', 'Outlet_Type_Supermarket Type168', 'Outlet_Type_Supermarket Type169', 'Outlet_Type_Supermarket Type17', 'Outlet_Type_Supermarket Type170', 'Outlet_Type_Supermarket Type171', 'Outlet_Type_Supermarket Type172', 'Outlet_Type_Supermarket Type173', 'Outlet_Type_Supermarket Type174', 'Outlet_Type_Supermarket Type175', 'Outlet_Type_Supermarket Type176', 'Outlet_Type_Supermarket Type177', 'Outlet_Type_Supermarket Type178', 'Outlet_Type_Supermarket Type179', 'Outlet_Type_Supermarket Type18', 'Outlet_Type_Supermarket Type180', 'Outlet_Type_Supermarket Type181', 'Outlet_Type_Supermarket Type182', 'Outlet_Type_Supermarket Type184', 'Outlet_Type_Supermarket Type186', 'Outlet_Type_Supermarket Type187', 'Outlet_Type_Supermarket Type188', 'Outlet_Type_Supermarket Type189', 'Outlet_Type_Supermarket Type19', 'Outlet_Type_Supermarket Type190', 'Outlet_Type_Supermarket Type191', 'Outlet_Type_Supermarket Type192', 'Outlet_Type_Supermarket Type193', 'Outlet_Type_Supermarket Type194', 'Outlet_Type_Supermarket Type195', 'Outlet_Type_Supermarket Type197', 'Outlet_Type_Supermarket Type198', 'Outlet_Type_Supermarket Type199', 'Outlet_Type_Supermarket Type2', 'Outlet_Type_Supermarket Type20', 'Outlet_Type_Supermarket Type200', 'Outlet_Type_Supermarket Type201', 'Outlet_Type_Supermarket Type202', 'Outlet_Type_Supermarket Type203', 'Outlet_Type_Supermarket Type204', 'Outlet_Type_Supermarket Type205', 'Outlet_Type_Supermarket Type206', 'Outlet_Type_Supermarket Type207', 'Outlet_Type_Supermarket Type208', 'Outlet_Type_Supermarket Type209', 'Outlet_Type_Supermarket Type21', 'Outlet_Type_Supermarket Type210', 'Outlet_Type_Supermarket Type211', 'Outlet_Type_Supermarket Type212', 'Outlet_Type_Supermarket Type213', 'Outlet_Type_Supermarket Type214', 'Outlet_Type_Supermarket Type215', 'Outlet_Type_Supermarket Type216', 'Outlet_Type_Supermarket Type217', 'Outlet_Type_Supermarket Type218', 'Outlet_Type_Supermarket Type219', 'Outlet_Type_Supermarket Type22', 'Outlet_Type_Supermarket Type220', 'Outlet_Type_Supermarket Type221', 'Outlet_Type_Supermarket Type222', 'Outlet_Type_Supermarket Type223', 'Outlet_Type_Supermarket Type224', 'Outlet_Type_Supermarket Type225', 'Outlet_Type_Supermarket Type226', 'Outlet_Type_Supermarket Type227', 'Outlet_Type_Supermarket Type228', 'Outlet_Type_Supermarket Type229', 'Outlet_Type_Supermarket Type23', 'Outlet_Type_Supermarket Type230', 'Outlet_Type_Supermarket Type231', 'Outlet_Type_Supermarket Type232', 'Outlet_Type_Supermarket Type233', 'Outlet_Type_Supermarket Type234', 'Outlet_Type_Supermarket Type235', 'Outlet_Type_Supermarket Type236', 'Outlet_Type_Supermarket Type237', 'Outlet_Type_Supermarket Type238', 'Outlet_Type_Supermarket Type239', 'Outlet_Type_Supermarket Type24', 'Outlet_Type_Supermarket Type240', 'Outlet_Type_Supermarket Type241', 'Outlet_Type_Supermarket Type242', 'Outlet_Type_Supermarket Type243', 'Outlet_Type_Supermarket Type244', 'Outlet_Type_Supermarket Type246', 'Outlet_Type_Supermarket Type247', 'Outlet_Type_Supermarket Type248', 'Outlet_Type_Supermarket Type249', 'Outlet_Type_Supermarket Type25', 'Outlet_Type_Supermarket Type250', 'Outlet_Type_Supermarket Type251', 'Outlet_Type_Supermarket Type252', 'Outlet_Type_Supermarket Type253', 'Outlet_Type_Supermarket Type254', 'Outlet_Type_Supermarket Type255', 'Outlet_Type_Supermarket Type256', 'Outlet_Type_Supermarket Type257', 'Outlet_Type_Supermarket Type258', 'Outlet_Type_Supermarket Type259', 'Outlet_Type_Supermarket Type26', 'Outlet_Type_Supermarket Type260', 'Outlet_Type_Supermarket Type261', 'Outlet_Type_Supermarket Type262', 'Outlet_Type_Supermarket Type263', 'Outlet_Type_Supermarket Type264', 'Outlet_Type_Supermarket Type265', 'Outlet_Type_Supermarket Type266', 'Outlet_Type_Supermarket Type267', 'Outlet_Type_Supermarket Type268', 'Outlet_Type_Supermarket Type269', 'Outlet_Type_Supermarket Type27', 'Outlet_Type_Supermarket Type270', 'Outlet_Type_Supermarket Type271', 'Outlet_Type_Supermarket Type272', 'Outlet_Type_Supermarket Type273', 'Outlet_Type_Supermarket Type274', 'Outlet_Type_Supermarket Type275', 'Outlet_Type_Supermarket Type276', 'Outlet_Type_Supermarket Type277', 'Outlet_Type_Supermarket Type278', 'Outlet_Type_Supermarket Type279', 'Outlet_Type_Supermarket Type28', 'Outlet_Type_Supermarket Type280', 'Outlet_Type_Supermarket Type281', 'Outlet_Type_Supermarket Type282', 'Outlet_Type_Supermarket Type283', 'Outlet_Type_Supermarket Type284', 'Outlet_Type_Supermarket Type285', 'Outlet_Type_Supermarket Type286', 'Outlet_Type_Supermarket Type287', 'Outlet_Type_Supermarket Type288', 'Outlet_Type_Supermarket Type289', 'Outlet_Type_Supermarket Type29', 'Outlet_Type_Supermarket Type290', 'Outlet_Type_Supermarket Type291', 'Outlet_Type_Supermarket Type292', 'Outlet_Type_Supermarket Type293', 'Outlet_Type_Supermarket Type294', 'Outlet_Type_Supermarket Type295', 'Outlet_Type_Supermarket Type296', 'Outlet_Type_Supermarket Type297', 'Outlet_Type_Supermarket Type298', 'Outlet_Type_Supermarket Type299', 'Outlet_Type_Supermarket Type3', 'Outlet_Type_Supermarket Type30', 'Outlet_Type_Supermarket Type300', 'Outlet_Type_Supermarket Type301', 'Outlet_Type_Supermarket Type302', 'Outlet_Type_Supermarket Type303', 'Outlet_Type_Supermarket Type304', 'Outlet_Type_Supermarket Type305', 'Outlet_Type_Supermarket Type306', 'Outlet_Type_Supermarket Type307', 'Outlet_Type_Supermarket Type308', 'Outlet_Type_Supermarket Type309', 'Outlet_Type_Supermarket Type31', 'Outlet_Type_Supermarket Type310', 'Outlet_Type_Supermarket Type311', 'Outlet_Type_Supermarket Type312', 'Outlet_Type_Supermarket Type313', 'Outlet_Type_Supermarket Type314', 'Outlet_Type_Supermarket Type315', 'Outlet_Type_Supermarket Type316', 'Outlet_Type_Supermarket Type317', 'Outlet_Type_Supermarket Type318', 'Outlet_Type_Supermarket Type319', 'Outlet_Type_Supermarket Type32', 'Outlet_Type_Supermarket Type320', 'Outlet_Type_Supermarket Type321', 'Outlet_Type_Supermarket Type322', 'Outlet_Type_Supermarket Type323', 'Outlet_Type_Supermarket Type324', 'Outlet_Type_Supermarket Type325', 'Outlet_Type_Supermarket Type326', 'Outlet_Type_Supermarket Type327', 'Outlet_Type_Supermarket Type328', 'Outlet_Type_Supermarket Type329', 'Outlet_Type_Supermarket Type33', 'Outlet_Type_Supermarket Type330', 'Outlet_Type_Supermarket Type331', 'Outlet_Type_Supermarket Type332', 'Outlet_Type_Supermarket Type333', 'Outlet_Type_Supermarket Type334', 'Outlet_Type_Supermarket Type335', 'Outlet_Type_Supermarket Type336', 'Outlet_Type_Supermarket Type337', 'Outlet_Type_Supermarket Type338', 'Outlet_Type_Supermarket Type339', 'Outlet_Type_Supermarket Type34', 'Outlet_Type_Supermarket Type340', 'Outlet_Type_Supermarket Type341', 'Outlet_Type_Supermarket Type342', 'Outlet_Type_Supermarket Type343', 'Outlet_Type_Supermarket Type344', 'Outlet_Type_Supermarket Type345', 'Outlet_Type_Supermarket Type346', 'Outlet_Type_Supermarket Type347', 'Outlet_Type_Supermarket Type348', 'Outlet_Type_Supermarket Type35', 'Outlet_Type_Supermarket Type350', 'Outlet_Type_Supermarket Type351', 'Outlet_Type_Supermarket Type352', 'Outlet_Type_Supermarket Type353', 'Outlet_Type_Supermarket Type354', 'Outlet_Type_Supermarket Type355', 'Outlet_Type_Supermarket Type356', 'Outlet_Type_Supermarket Type357', 'Outlet_Type_Supermarket Type358', 'Outlet_Type_Supermarket Type359', 'Outlet_Type_Supermarket Type36', 'Outlet_Type_Supermarket Type360', 'Outlet_Type_Supermarket Type361', 'Outlet_Type_Supermarket Type362', 'Outlet_Type_Supermarket Type363', 'Outlet_Type_Supermarket Type364', 'Outlet_Type_Supermarket Type365', 'Outlet_Type_Supermarket Type366', 'Outlet_Type_Supermarket Type367', 'Outlet_Type_Supermarket Type368', 'Outlet_Type_Supermarket Type369', 'Outlet_Type_Supermarket Type37', 'Outlet_Type_Supermarket Type370', 'Outlet_Type_Supermarket Type371', 'Outlet_Type_Supermarket Type372', 'Outlet_Type_Supermarket Type373', 'Outlet_Type_Supermarket Type374', 'Outlet_Type_Supermarket Type375', 'Outlet_Type_Supermarket Type376', 'Outlet_Type_Supermarket Type377', 'Outlet_Type_Supermarket Type378', 'Outlet_Type_Supermarket Type379', 'Outlet_Type_Supermarket Type38', 'Outlet_Type_Supermarket Type380', 'Outlet_Type_Supermarket Type381', 'Outlet_Type_Supermarket Type382', 'Outlet_Type_Supermarket Type383', 'Outlet_Type_Supermarket Type384', 'Outlet_Type_Supermarket Type385', 'Outlet_Type_Supermarket Type386', 'Outlet_Type_Supermarket Type387', 'Outlet_Type_Supermarket Type388', 'Outlet_Type_Supermarket Type389', 'Outlet_Type_Supermarket Type39', 'Outlet_Type_Supermarket Type390', 'Outlet_Type_Supermarket Type391', 'Outlet_Type_Supermarket Type392', 'Outlet_Type_Supermarket Type393', 'Outlet_Type_Supermarket Type394', 'Outlet_Type_Supermarket Type395', 'Outlet_Type_Supermarket Type396', 'Outlet_Type_Supermarket Type397', 'Outlet_Type_Supermarket Type398', 'Outlet_Type_Supermarket Type399', 'Outlet_Type_Supermarket Type4', 'Outlet_Type_Supermarket Type40', 'Outlet_Type_Supermarket Type400', 'Outlet_Type_Supermarket Type401', 'Outlet_Type_Supermarket Type402', 'Outlet_Type_Supermarket Type403', 'Outlet_Type_Supermarket Type404', 'Outlet_Type_Supermarket Type405', 'Outlet_Type_Supermarket Type406', 'Outlet_Type_Supermarket Type407', 'Outlet_Type_Supermarket Type408', 'Outlet_Type_Supermarket Type409', 'Outlet_Type_Supermarket Type41', 'Outlet_Type_Supermarket Type410', 'Outlet_Type_Supermarket Type411', 'Outlet_Type_Supermarket Type412', 'Outlet_Type_Supermarket Type413', 'Outlet_Type_Supermarket Type414', 'Outlet_Type_Supermarket Type415', 'Outlet_Type_Supermarket Type416', 'Outlet_Type_Supermarket Type417', 'Outlet_Type_Supermarket Type418', 'Outlet_Type_Supermarket Type419', 'Outlet_Type_Supermarket Type42', 'Outlet_Type_Supermarket Type420', 'Outlet_Type_Supermarket Type421', 'Outlet_Type_Supermarket Type422', 'Outlet_Type_Supermarket Type423', 'Outlet_Type_Supermarket Type424', 'Outlet_Type_Supermarket Type425', 'Outlet_Type_Supermarket Type426', 'Outlet_Type_Supermarket Type427', 'Outlet_Type_Supermarket Type428', 'Outlet_Type_Supermarket Type429', 'Outlet_Type_Supermarket Type43', 'Outlet_Type_Supermarket Type430', 'Outlet_Type_Supermarket Type431', 'Outlet_Type_Supermarket Type432', 'Outlet_Type_Supermarket Type433', 'Outlet_Type_Supermarket Type434', 'Outlet_Type_Supermarket Type435', 'Outlet_Type_Supermarket Type436', 'Outlet_Type_Supermarket Type437', 'Outlet_Type_Supermarket Type438', 'Outlet_Type_Supermarket Type439', 'Outlet_Type_Supermarket Type44', 'Outlet_Type_Supermarket Type441', 'Outlet_Type_Supermarket Type442', 'Outlet_Type_Supermarket Type443', 'Outlet_Type_Supermarket Type444', 'Outlet_Type_Supermarket Type445', 'Outlet_Type_Supermarket Type446', 'Outlet_Type_Supermarket Type447', 'Outlet_Type_Supermarket Type448', 'Outlet_Type_Supermarket Type449', 'Outlet_Type_Supermarket Type45', 'Outlet_Type_Supermarket Type450', 'Outlet_Type_Supermarket Type451', 'Outlet_Type_Supermarket Type452', 'Outlet_Type_Supermarket Type453', 'Outlet_Type_Supermarket Type454', 'Outlet_Type_Supermarket Type455', 'Outlet_Type_Supermarket Type456', 'Outlet_Type_Supermarket Type457', 'Outlet_Type_Supermarket Type458', 'Outlet_Type_Supermarket Type46', 'Outlet_Type_Supermarket Type460', 'Outlet_Type_Supermarket Type461', 'Outlet_Type_Supermarket Type462', 'Outlet_Type_Supermarket Type463', 'Outlet_Type_Supermarket Type464', 'Outlet_Type_Supermarket Type465', 'Outlet_Type_Supermarket Type466', 'Outlet_Type_Supermarket Type468', 'Outlet_Type_Supermarket Type469', 'Outlet_Type_Supermarket Type47', 'Outlet_Type_Supermarket Type470', 'Outlet_Type_Supermarket Type471', 'Outlet_Type_Supermarket Type472', 'Outlet_Type_Supermarket Type473', 'Outlet_Type_Supermarket Type474', 'Outlet_Type_Supermarket Type475', 'Outlet_Type_Supermarket Type476', 'Outlet_Type_Supermarket Type477', 'Outlet_Type_Supermarket Type478', 'Outlet_Type_Supermarket Type479', 'Outlet_Type_Supermarket Type48', 'Outlet_Type_Supermarket Type480', 'Outlet_Type_Supermarket Type481', 'Outlet_Type_Supermarket Type482', 'Outlet_Type_Supermarket Type483', 'Outlet_Type_Supermarket Type484', 'Outlet_Type_Supermarket Type485', 'Outlet_Type_Supermarket Type486', 'Outlet_Type_Supermarket Type487', 'Outlet_Type_Supermarket Type488', 'Outlet_Type_Supermarket Type489', 'Outlet_Type_Supermarket Type49', 'Outlet_Type_Supermarket Type490', 'Outlet_Type_Supermarket Type491', 'Outlet_Type_Supermarket Type493', 'Outlet_Type_Supermarket Type494', 'Outlet_Type_Supermarket Type495', 'Outlet_Type_Supermarket Type496', 'Outlet_Type_Supermarket Type497', 'Outlet_Type_Supermarket Type498', 'Outlet_Type_Supermarket Type499', 'Outlet_Type_Supermarket Type5', 'Outlet_Type_Supermarket Type50', 'Outlet_Type_Supermarket Type500', 'Outlet_Type_Supermarket Type501', 'Outlet_Type_Supermarket Type502', 'Outlet_Type_Supermarket Type503', 'Outlet_Type_Supermarket Type504', 'Outlet_Type_Supermarket Type505', 'Outlet_Type_Supermarket Type506', 'Outlet_Type_Supermarket Type507', 'Outlet_Type_Supermarket Type508', 'Outlet_Type_Supermarket Type509', 'Outlet_Type_Supermarket Type51', 'Outlet_Type_Supermarket Type510', 'Outlet_Type_Supermarket Type511', 'Outlet_Type_Supermarket Type512', 'Outlet_Type_Supermarket Type513', 'Outlet_Type_Supermarket Type514', 'Outlet_Type_Supermarket Type515', 'Outlet_Type_Supermarket Type516', 'Outlet_Type_Supermarket Type517', 'Outlet_Type_Supermarket Type518', 'Outlet_Type_Supermarket Type519', 'Outlet_Type_Supermarket Type52', 'Outlet_Type_Supermarket Type520', 'Outlet_Type_Supermarket Type521', 'Outlet_Type_Supermarket Type522', 'Outlet_Type_Supermarket Type523', 'Outlet_Type_Supermarket Type524', 'Outlet_Type_Supermarket Type525', 'Outlet_Type_Supermarket Type526', 'Outlet_Type_Supermarket Type527', 'Outlet_Type_Supermarket Type528', 'Outlet_Type_Supermarket Type529', 'Outlet_Type_Supermarket Type53', 'Outlet_Type_Supermarket Type530', 'Outlet_Type_Supermarket Type531', 'Outlet_Type_Supermarket Type532', 'Outlet_Type_Supermarket Type533', 'Outlet_Type_Supermarket Type534', 'Outlet_Type_Supermarket Type535', 'Outlet_Type_Supermarket Type536', 'Outlet_Type_Supermarket Type537', 'Outlet_Type_Supermarket Type538', 'Outlet_Type_Supermarket Type539', 'Outlet_Type_Supermarket Type54', 'Outlet_Type_Supermarket Type540', 'Outlet_Type_Supermarket Type541', 'Outlet_Type_Supermarket Type542', 'Outlet_Type_Supermarket Type543', 'Outlet_Type_Supermarket Type544', 'Outlet_Type_Supermarket Type545', 'Outlet_Type_Supermarket Type546', 'Outlet_Type_Supermarket Type547', 'Outlet_Type_Supermarket Type548', 'Outlet_Type_Supermarket Type549', 'Outlet_Type_Supermarket Type55', 'Outlet_Type_Supermarket Type550', 'Outlet_Type_Supermarket Type551', 'Outlet_Type_Supermarket Type552', 'Outlet_Type_Supermarket Type553', 'Outlet_Type_Supermarket Type554', 'Outlet_Type_Supermarket Type555', 'Outlet_Type_Supermarket Type556', 'Outlet_Type_Supermarket Type557', 'Outlet_Type_Supermarket Type558', 'Outlet_Type_Supermarket Type559', 'Outlet_Type_Supermarket Type56', 'Outlet_Type_Supermarket Type560', 'Outlet_Type_Supermarket Type561', 'Outlet_Type_Supermarket Type562', 'Outlet_Type_Supermarket Type563', 'Outlet_Type_Supermarket Type564', 'Outlet_Type_Supermarket Type565', 'Outlet_Type_Supermarket Type566', 'Outlet_Type_Supermarket Type567', 'Outlet_Type_Supermarket Type568', 'Outlet_Type_Supermarket Type569', 'Outlet_Type_Supermarket Type57', 'Outlet_Type_Supermarket Type570', 'Outlet_Type_Supermarket Type571', 'Outlet_Type_Supermarket Type572', 'Outlet_Type_Supermarket Type573', 'Outlet_Type_Supermarket Type574', 'Outlet_Type_Supermarket Type575', 'Outlet_Type_Supermarket Type576', 'Outlet_Type_Supermarket Type577', 'Outlet_Type_Supermarket Type578', 'Outlet_Type_Supermarket Type579', 'Outlet_Type_Supermarket Type58', 'Outlet_Type_Supermarket Type580', 'Outlet_Type_Supermarket Type582', 'Outlet_Type_Supermarket Type583', 'Outlet_Type_Supermarket Type584', 'Outlet_Type_Supermarket Type585', 'Outlet_Type_Supermarket Type586', 'Outlet_Type_Supermarket Type587', 'Outlet_Type_Supermarket Type588', 'Outlet_Type_Supermarket Type589', 'Outlet_Type_Supermarket Type59', 'Outlet_Type_Supermarket Type590', 'Outlet_Type_Supermarket Type591', 'Outlet_Type_Supermarket Type592', 'Outlet_Type_Supermarket Type593', 'Outlet_Type_Supermarket Type594', 'Outlet_Type_Supermarket Type595', 'Outlet_Type_Supermarket Type596', 'Outlet_Type_Supermarket Type597', 'Outlet_Type_Supermarket Type598', 'Outlet_Type_Supermarket Type599', 'Outlet_Type_Supermarket Type6', 'Outlet_Type_Supermarket Type60', 'Outlet_Type_Supermarket Type600', 'Outlet_Type_Supermarket Type602', 'Outlet_Type_Supermarket Type603', 'Outlet_Type_Supermarket Type604', 'Outlet_Type_Supermarket Type605', 'Outlet_Type_Supermarket Type606', 'Outlet_Type_Supermarket Type607', 'Outlet_Type_Supermarket Type608', 'Outlet_Type_Supermarket Type609', 'Outlet_Type_Supermarket Type61', 'Outlet_Type_Supermarket Type610', 'Outlet_Type_Supermarket Type611', 'Outlet_Type_Supermarket Type612', 'Outlet_Type_Supermarket Type613', 'Outlet_Type_Supermarket Type614', 'Outlet_Type_Supermarket Type615', 'Outlet_Type_Supermarket Type616', 'Outlet_Type_Supermarket Type617', 'Outlet_Type_Supermarket Type618', 'Outlet_Type_Supermarket Type619', 'Outlet_Type_Supermarket Type62', 'Outlet_Type_Supermarket Type620', 'Outlet_Type_Supermarket Type621', 'Outlet_Type_Supermarket Type622', 'Outlet_Type_Supermarket Type623', 'Outlet_Type_Supermarket Type624', 'Outlet_Type_Supermarket Type625', 'Outlet_Type_Supermarket Type626', 'Outlet_Type_Supermarket Type627', 'Outlet_Type_Supermarket Type628', 'Outlet_Type_Supermarket Type629', 'Outlet_Type_Supermarket Type63', 'Outlet_Type_Supermarket Type630', 'Outlet_Type_Supermarket Type631', 'Outlet_Type_Supermarket Type632', 'Outlet_Type_Supermarket Type633', 'Outlet_Type_Supermarket Type634', 'Outlet_Type_Supermarket Type635', 'Outlet_Type_Supermarket Type636', 'Outlet_Type_Supermarket Type637', 'Outlet_Type_Supermarket Type638', 'Outlet_Type_Supermarket Type639', 'Outlet_Type_Supermarket Type64', 'Outlet_Type_Supermarket Type640', 'Outlet_Type_Supermarket Type641', 'Outlet_Type_Supermarket Type642', 'Outlet_Type_Supermarket Type643', 'Outlet_Type_Supermarket Type644', 'Outlet_Type_Supermarket Type645', 'Outlet_Type_Supermarket Type646', 'Outlet_Type_Supermarket Type647', 'Outlet_Type_Supermarket Type648', 'Outlet_Type_Supermarket Type649', 'Outlet_Type_Supermarket Type65', 'Outlet_Type_Supermarket Type651', 'Outlet_Type_Supermarket Type652', 'Outlet_Type_Supermarket Type653', 'Outlet_Type_Supermarket Type654', 'Outlet_Type_Supermarket Type655', 'Outlet_Type_Supermarket Type656', 'Outlet_Type_Supermarket Type657', 'Outlet_Type_Supermarket Type658', 'Outlet_Type_Supermarket Type659', 'Outlet_Type_Supermarket Type66', 'Outlet_Type_Supermarket Type660', 'Outlet_Type_Supermarket Type661', 'Outlet_Type_Supermarket Type662', 'Outlet_Type_Supermarket Type663', 'Outlet_Type_Supermarket Type664', 'Outlet_Type_Supermarket Type665', 'Outlet_Type_Supermarket Type666', 'Outlet_Type_Supermarket Type667', 'Outlet_Type_Supermarket Type668', 'Outlet_Type_Supermarket Type669', 'Outlet_Type_Supermarket Type67', 'Outlet_Type_Supermarket Type670', 'Outlet_Type_Supermarket Type671', 'Outlet_Type_Supermarket Type672', 'Outlet_Type_Supermarket Type673', 'Outlet_Type_Supermarket Type674', 'Outlet_Type_Supermarket Type675', 'Outlet_Type_Supermarket Type676', 'Outlet_Type_Supermarket Type677', 'Outlet_Type_Supermarket Type678', 'Outlet_Type_Supermarket Type679', 'Outlet_Type_Supermarket Type68', 'Outlet_Type_Supermarket Type680', 'Outlet_Type_Supermarket Type681', 'Outlet_Type_Supermarket Type682', 'Outlet_Type_Supermarket Type683', 'Outlet_Type_Supermarket Type684', 'Outlet_Type_Supermarket Type685', 'Outlet_Type_Supermarket Type686', 'Outlet_Type_Supermarket Type687', 'Outlet_Type_Supermarket Type688', 'Outlet_Type_Supermarket Type689', 'Outlet_Type_Supermarket Type69', 'Outlet_Type_Supermarket Type690', 'Outlet_Type_Supermarket Type691', 'Outlet_Type_Supermarket Type692', 'Outlet_Type_Supermarket Type693', 'Outlet_Type_Supermarket Type694', 'Outlet_Type_Supermarket Type695', 'Outlet_Type_Supermarket Type696', 'Outlet_Type_Supermarket Type697', 'Outlet_Type_Supermarket Type698', 'Outlet_Type_Supermarket Type699', 'Outlet_Type_Supermarket Type7', 'Outlet_Type_Supermarket Type70', 'Outlet_Type_Supermarket Type700', 'Outlet_Type_Supermarket Type701', 'Outlet_Type_Supermarket Type702', 'Outlet_Type_Supermarket Type703', 'Outlet_Type_Supermarket Type704', 'Outlet_Type_Supermarket Type705', 'Outlet_Type_Supermarket Type706', 'Outlet_Type_Supermarket Type707', 'Outlet_Type_Supermarket Type708', 'Outlet_Type_Supermarket Type709', 'Outlet_Type_Supermarket Type71', 'Outlet_Type_Supermarket Type710', 'Outlet_Type_Supermarket Type711', 'Outlet_Type_Supermarket Type712', 'Outlet_Type_Supermarket Type713', 'Outlet_Type_Supermarket Type714', 'Outlet_Type_Supermarket Type715', 'Outlet_Type_Supermarket Type716', 'Outlet_Type_Supermarket Type717', 'Outlet_Type_Supermarket Type718', 'Outlet_Type_Supermarket Type719', 'Outlet_Type_Supermarket Type72', 'Outlet_Type_Supermarket Type720', 'Outlet_Type_Supermarket Type721', 'Outlet_Type_Supermarket Type722', 'Outlet_Type_Supermarket Type723', 'Outlet_Type_Supermarket Type724', 'Outlet_Type_Supermarket Type725', 'Outlet_Type_Supermarket Type726', 'Outlet_Type_Supermarket Type727', 'Outlet_Type_Supermarket Type728', 'Outlet_Type_Supermarket Type729', 'Outlet_Type_Supermarket Type73', 'Outlet_Type_Supermarket Type730', 'Outlet_Type_Supermarket Type731', 'Outlet_Type_Supermarket Type732', 'Outlet_Type_Supermarket Type733', 'Outlet_Type_Supermarket Type734', 'Outlet_Type_Supermarket Type735', 'Outlet_Type_Supermarket Type736', 'Outlet_Type_Supermarket Type737', 'Outlet_Type_Supermarket Type738', 'Outlet_Type_Supermarket Type739', 'Outlet_Type_Supermarket Type74', 'Outlet_Type_Supermarket Type740', 'Outlet_Type_Supermarket Type741', 'Outlet_Type_Supermarket Type742', 'Outlet_Type_Supermarket Type743', 'Outlet_Type_Supermarket Type745', 'Outlet_Type_Supermarket Type746', 'Outlet_Type_Supermarket Type747', 'Outlet_Type_Supermarket Type748', 'Outlet_Type_Supermarket Type749', 'Outlet_Type_Supermarket Type75', 'Outlet_Type_Supermarket Type750', 'Outlet_Type_Supermarket Type751', 'Outlet_Type_Supermarket Type752', 'Outlet_Type_Supermarket Type753', 'Outlet_Type_Supermarket Type754', 'Outlet_Type_Supermarket Type755', 'Outlet_Type_Supermarket Type756', 'Outlet_Type_Supermarket Type757', 'Outlet_Type_Supermarket Type758', 'Outlet_Type_Supermarket Type759', 'Outlet_Type_Supermarket Type76', 'Outlet_Type_Supermarket Type760', 'Outlet_Type_Supermarket Type761', 'Outlet_Type_Supermarket Type762', 'Outlet_Type_Supermarket Type763', 'Outlet_Type_Supermarket Type764', 'Outlet_Type_Supermarket Type765', 'Outlet_Type_Supermarket Type766', 'Outlet_Type_Supermarket Type767', 'Outlet_Type_Supermarket Type768', 'Outlet_Type_Supermarket Type769', 'Outlet_Type_Supermarket Type77', 'Outlet_Type_Supermarket Type770', 'Outlet_Type_Supermarket Type771', 'Outlet_Type_Supermarket Type772', 'Outlet_Type_Supermarket Type773', 'Outlet_Type_Supermarket Type774', 'Outlet_Type_Supermarket Type775', 'Outlet_Type_Supermarket Type776', 'Outlet_Type_Supermarket Type777', 'Outlet_Type_Supermarket Type778', 'Outlet_Type_Supermarket Type779', 'Outlet_Type_Supermarket Type78', 'Outlet_Type_Supermarket Type780', 'Outlet_Type_Supermarket Type781', 'Outlet_Type_Supermarket Type782', 'Outlet_Type_Supermarket Type783', 'Outlet_Type_Supermarket Type784', 'Outlet_Type_Supermarket Type785', 'Outlet_Type_Supermarket Type786', 'Outlet_Type_Supermarket Type787', 'Outlet_Type_Supermarket Type788', 'Outlet_Type_Supermarket Type789', 'Outlet_Type_Supermarket Type79', 'Outlet_Type_Supermarket Type790', 'Outlet_Type_Supermarket Type791', 'Outlet_Type_Supermarket Type792', 'Outlet_Type_Supermarket Type793', 'Outlet_Type_Supermarket Type794', 'Outlet_Type_Supermarket Type795', 'Outlet_Type_Supermarket Type796', 'Outlet_Type_Supermarket Type797', 'Outlet_Type_Supermarket Type798', 'Outlet_Type_Supermarket Type799', 'Outlet_Type_Supermarket Type8', 'Outlet_Type_Supermarket Type80', 'Outlet_Type_Supermarket Type800', 'Outlet_Type_Supermarket Type801', 'Outlet_Type_Supermarket Type802', 'Outlet_Type_Supermarket Type803', 'Outlet_Type_Supermarket Type804', 'Outlet_Type_Supermarket Type805', 'Outlet_Type_Supermarket Type806', 'Outlet_Type_Supermarket Type807', 'Outlet_Type_Supermarket Type808', 'Outlet_Type_Supermarket Type809', 'Outlet_Type_Supermarket Type81', 'Outlet_Type_Supermarket Type810', 'Outlet_Type_Supermarket Type811', 'Outlet_Type_Supermarket Type812', 'Outlet_Type_Supermarket Type813', 'Outlet_Type_Supermarket Type814', 'Outlet_Type_Supermarket Type815', 'Outlet_Type_Supermarket Type816', 'Outlet_Type_Supermarket Type817', 'Outlet_Type_Supermarket Type818', 'Outlet_Type_Supermarket Type819', 'Outlet_Type_Supermarket Type82', 'Outlet_Type_Supermarket Type820', 'Outlet_Type_Supermarket Type821', 'Outlet_Type_Supermarket Type822', 'Outlet_Type_Supermarket Type823', 'Outlet_Type_Supermarket Type825', 'Outlet_Type_Supermarket Type826', 'Outlet_Type_Supermarket Type827', 'Outlet_Type_Supermarket Type828', 'Outlet_Type_Supermarket Type829', 'Outlet_Type_Supermarket Type83', 'Outlet_Type_Supermarket Type830', 'Outlet_Type_Supermarket Type831', 'Outlet_Type_Supermarket Type832', 'Outlet_Type_Supermarket Type833', 'Outlet_Type_Supermarket Type834', 'Outlet_Type_Supermarket Type835', 'Outlet_Type_Supermarket Type836', 'Outlet_Type_Supermarket Type837', 'Outlet_Type_Supermarket Type838', 'Outlet_Type_Supermarket Type839', 'Outlet_Type_Supermarket Type84', 'Outlet_Type_Supermarket Type840', 'Outlet_Type_Supermarket Type841', 'Outlet_Type_Supermarket Type842', 'Outlet_Type_Supermarket Type843', 'Outlet_Type_Supermarket Type844', 'Outlet_Type_Supermarket Type845', 'Outlet_Type_Supermarket Type846', 'Outlet_Type_Supermarket Type847', 'Outlet_Type_Supermarket Type848', 'Outlet_Type_Supermarket Type849', 'Outlet_Type_Supermarket Type85', 'Outlet_Type_Supermarket Type850', 'Outlet_Type_Supermarket Type851', 'Outlet_Type_Supermarket Type852', 'Outlet_Type_Supermarket Type853', 'Outlet_Type_Supermarket Type854', 'Outlet_Type_Supermarket Type855', 'Outlet_Type_Supermarket Type856', 'Outlet_Type_Supermarket Type857', 'Outlet_Type_Supermarket Type858', 'Outlet_Type_Supermarket Type859', 'Outlet_Type_Supermarket Type86', 'Outlet_Type_Supermarket Type860', 'Outlet_Type_Supermarket Type861', 'Outlet_Type_Supermarket Type862', 'Outlet_Type_Supermarket Type863', 'Outlet_Type_Supermarket Type864', 'Outlet_Type_Supermarket Type865', 'Outlet_Type_Supermarket Type866', 'Outlet_Type_Supermarket Type867', 'Outlet_Type_Supermarket Type868', 'Outlet_Type_Supermarket Type869', 'Outlet_Type_Supermarket Type87', 'Outlet_Type_Supermarket Type870', 'Outlet_Type_Supermarket Type871', 'Outlet_Type_Supermarket Type872', 'Outlet_Type_Supermarket Type873', 'Outlet_Type_Supermarket Type874', 'Outlet_Type_Supermarket Type875', 'Outlet_Type_Supermarket Type876', 'Outlet_Type_Supermarket Type877', 'Outlet_Type_Supermarket Type878', 'Outlet_Type_Supermarket Type879', 'Outlet_Type_Supermarket Type88', 'Outlet_Type_Supermarket Type880', 'Outlet_Type_Supermarket Type881', 'Outlet_Type_Supermarket Type882', 'Outlet_Type_Supermarket Type883', 'Outlet_Type_Supermarket Type884', 'Outlet_Type_Supermarket Type885', 'Outlet_Type_Supermarket Type886', 'Outlet_Type_Supermarket Type887', 'Outlet_Type_Supermarket Type888', 'Outlet_Type_Supermarket Type889', 'Outlet_Type_Supermarket Type89', 'Outlet_Type_Supermarket Type890', 'Outlet_Type_Supermarket Type891', 'Outlet_Type_Supermarket Type892', 'Outlet_Type_Supermarket Type893', 'Outlet_Type_Supermarket Type894', 'Outlet_Type_Supermarket Type895', 'Outlet_Type_Supermarket Type896', 'Outlet_Type_Supermarket Type897', 'Outlet_Type_Supermarket Type898', 'Outlet_Type_Supermarket Type899', 'Outlet_Type_Supermarket Type9', 'Outlet_Type_Supermarket Type90', 'Outlet_Type_Supermarket Type900', 'Outlet_Type_Supermarket Type901', 'Outlet_Type_Supermarket Type902', 'Outlet_Type_Supermarket Type903', 'Outlet_Type_Supermarket Type904', 'Outlet_Type_Supermarket Type905', 'Outlet_Type_Supermarket Type906', 'Outlet_Type_Supermarket Type907', 'Outlet_Type_Supermarket Type908', 'Outlet_Type_Supermarket Type909', 'Outlet_Type_Supermarket Type91', 'Outlet_Type_Supermarket Type910', 'Outlet_Type_Supermarket Type911', 'Outlet_Type_Supermarket Type912', 'Outlet_Type_Supermarket Type913', 'Outlet_Type_Supermarket Type914', 'Outlet_Type_Supermarket Type915', 'Outlet_Type_Supermarket Type916', 'Outlet_Type_Supermarket Type917', 'Outlet_Type_Supermarket Type918', 'Outlet_Type_Supermarket Type919', 'Outlet_Type_Supermarket Type92', 'Outlet_Type_Supermarket Type920', 'Outlet_Type_Supermarket Type921', 'Outlet_Type_Supermarket Type922', 'Outlet_Type_Supermarket Type923', 'Outlet_Type_Supermarket Type924', 'Outlet_Type_Supermarket Type925', 'Outlet_Type_Supermarket Type926', 'Outlet_Type_Supermarket Type927', 'Outlet_Type_Supermarket Type928', 'Outlet_Type_Supermarket Type929', 'Outlet_Type_Supermarket Type930', 'Outlet_Type_Supermarket Type931', 'Outlet_Type_Supermarket Type932', 'Outlet_Type_Supermarket Type933', 'Outlet_Type_Supermarket Type934', 'Outlet_Type_Supermarket Type935', 'Outlet_Type_Supermarket Type936', 'Outlet_Type_Supermarket Type937', 'Outlet_Type_Supermarket Type938', 'Outlet_Type_Supermarket Type939', 'Outlet_Type_Supermarket Type94', 'Outlet_Type_Supermarket Type940', 'Outlet_Type_Supermarket Type941', 'Outlet_Type_Supermarket Type942', 'Outlet_Type_Supermarket Type943', 'Outlet_Type_Supermarket Type944', 'Outlet_Type_Supermarket Type945', 'Outlet_Type_Supermarket Type946', 'Outlet_Type_Supermarket Type947', 'Outlet_Type_Supermarket Type948', 'Outlet_Type_Supermarket Type949', 'Outlet_Type_Supermarket Type95', 'Outlet_Type_Supermarket Type950', 'Outlet_Type_Supermarket Type951', 'Outlet_Type_Supermarket Type952', 'Outlet_Type_Supermarket Type953', 'Outlet_Type_Supermarket Type954', 'Outlet_Type_Supermarket Type955', 'Outlet_Type_Supermarket Type956', 'Outlet_Type_Supermarket Type957', 'Outlet_Type_Supermarket Type958', 'Outlet_Type_Supermarket Type959', 'Outlet_Type_Supermarket Type96', 'Outlet_Type_Supermarket Type960', 'Outlet_Type_Supermarket Type961', 'Outlet_Type_Supermarket Type962', 'Outlet_Type_Supermarket Type963', 'Outlet_Type_Supermarket Type964', 'Outlet_Type_Supermarket Type966', 'Outlet_Type_Supermarket Type967', 'Outlet_Type_Supermarket Type968', 'Outlet_Type_Supermarket Type969', 'Outlet_Type_Supermarket Type97', 'Outlet_Type_Supermarket Type970', 'Outlet_Type_Supermarket Type971', 'Outlet_Type_Supermarket Type972', 'Outlet_Type_Supermarket Type973', 'Outlet_Type_Supermarket Type974', 'Outlet_Type_Supermarket Type975', 'Outlet_Type_Supermarket Type976', 'Outlet_Type_Supermarket Type977', 'Outlet_Type_Supermarket Type979', 'Outlet_Type_Supermarket Type98', 'Outlet_Type_Supermarket Type980', 'Outlet_Type_Supermarket Type981', 'Outlet_Type_Supermarket Type982', 'Outlet_Type_Supermarket Type983', 'Outlet_Type_Supermarket Type984', 'Outlet_Type_Supermarket Type985', 'Outlet_Type_Supermarket Type986', 'Outlet_Type_Supermarket Type987', 'Outlet_Type_Supermarket Type988', 'Outlet_Type_Supermarket Type989', 'Outlet_Type_Supermarket Type99', 'Outlet_Type_Supermarket Type990', 'Outlet_Type_Supermarket Type991', 'Outlet_Type_Supermarket Type992', 'Outlet_Type_Supermarket Type993', 'Outlet_Type_Supermarket Type994', 'Outlet_Type_Supermarket Type995', 'Outlet_Type_Supermarket Type996', 'Outlet_Type_Supermarket Type997', 'Outlet_Type_Supermarket Type998', 'Outlet_Type_Supermarket Type999'])
numeric_col=df[['Item_Weight', 'Item_Fat_Content', 'Item_Visibility', 'Item_MRP', 'Outlet_Size', 'Outlet_Location_Type', 'Item_Outlet_Sales', 'Outlet_age']]
final_dataset=pd.merge(dataFrame_encode_col,numeric_col,left_index=True,right_index=True)


#######################################################
#**** Separating dependent and independent feature ****
#######################################################

x=final_dataset.drop(target_col_name, axis=1)
y=final_dataset[target_col_name]


############################################################################
#**************** Separating Data into train and test data *****************
############################################################################

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2, random_state=1)
#Change the random_state parameter value. It can change the accuracy


############################################################################
#****************************** model training *****************************
############################################################################

from sklearn import metrics
from sklearn.metrics import mean_squared_error

from xgboost import XGBRegressor

xgb = XGBRegressor()
xgb.fit(x_train,y_train)
xgbtd=xgb.predict(x_test)
print('r2 score',r2_score(y_test,xgbtd))


######################################
#****** Model evaluation ************
######################################

# make predictions on the testing set
y_pred = xgb.predict(x_test)

# calculate the evaluation metrics

#Mean Squared error
mse = mean_squared_error(y_test, y_pred)
print(mse)

#Root mean squared error
rmse = np.sqrt(mse)
print(rmse)

#r2_score
r2 = r2_score(y_test, y_pred)
print(r2)

mean absolute error
mae = mean_absolute_error(y_test, y_pred)
print(mae)

Mean absolute percentage error
mape = np.mean(np.abs((y_test - y_pred) / y_test)) * 100
print(mape)


############################################
#********** Plot the residuals *************
############################################

residuals = y_test - y_pred
plt.scatter(y_pred, residuals)
plt.xlabel("Predicted Values")
plt.ylabel("Residuals")
plt.show()


