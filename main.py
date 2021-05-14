# import required libraries
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder

colors = pd.read_csv('data/color.csv', low_memory=False)
print('\n\n****** Colors ******')
print(colors)

phones = pd.read_csv('data/phone.csv', low_memory=False)
print('\n\n****** Phones ******')
print(phones)

print('\n\n****** Merged ******')
merged = phones.merge(colors)
del(merged['Color'])
print(merged)

# creating initial dataframe
#color_df = pd.DataFrame(colors, columns=['Color'])
# converting type of columns to 'category'
#color_df['Bridge_Types'] = color_df['Bridge_Types'].astype('category')
# Assigning numerical values and storing in another column
#bridge_df['Bridge_Types_Cat'] = bridge_df['Bridge_Types'].cat.codes
#print('\n\n***** Converting columns to categories *****')
#print(bridge_df)



# creating initial dataframe
#bridge_types = ('Arch','Beam','Truss','Cantilever','Tied Arch','Suspension','Cable')
#bridge_df = pd.DataFrame(bridge_types, columns=['Bridge_Types'])
# creating instance of labelencoder
#labelencoder = LabelEncoder()
# Assigning numerical values and storing in another column
#bridge_df['Bridge_Types_Cat'] = labelencoder.fit_transform(bridge_df['Bridge_Types'])
#print('\n\n***** Label Encoder *****')
#print(bridge_df)


from sklearn.preprocessing import OneHotEncoder
# creating instance of one-hot-encoder
#enc = OneHotEncoder(handle_unknown='ignore')
# passing bridge-types-cat column (label encoded values of bridge_types)
#enc_df = pd.DataFrame(enc.fit_transform(bridge_df[['Bridge_Types_Cat']]).toarray())
# merge with main df bridge_df on key values
#bridge_df = bridge_df.join(enc_df)
#print('\n\n***** One Hot Encoding *****')
#print(bridge_df)


# creating initial dataframe
#bridge_types = ('Arch','Beam','Truss','Cantilever','Tied Arch','Suspension','Cable')
#bridge_df = pd.DataFrame(bridge_types, columns=['Bridge_Types'])
# generate binary values using get_dummies
#dum_df = pd.get_dummies(bridge_df, columns=["Bridge_Types"], prefix=["Type_is"] )
# merge with main df bridge_df on key values
#bridge_df = bridge_df.join(dum_df)
#bridge_df.to_csv('bridge_dummy.csv')
#print('\n\n***** Using Dummy Values *****')
#print(bridge_df)