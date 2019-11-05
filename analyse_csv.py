
import pandas as pd
import pickle
import numpy as np
import sys

df = pd.read_csv("imageLinksWithAgeAndGender.csv")
print(df.shape)
df['age' ] = df['age'].loc[df['age'].str.len() ==2]
df['age'] = df['age'].fillna(0)
strs = df['age'].value_counts()
pos = strs.loc[df['age'].value_counts() > 10]
pos = pos.index.tolist()

df['age'] = df['age'].loc[df['age'].isin(pos)]
df['age'].dropna(inplace=True)

df['ageInt'] = df['age'].astype(int)

select = df['ageInt'].loc[(df['ageInt'] > 15) & (df['ageInt'] < 80) & (df['ageInt'] != np.inf) & (df['ageInt'] != -np.inf)]

print(select.shape)

age = df['ageInt'][select]

indexes = select.index.values

images = df['imagePath'][indexes]

ages = {}

trainImages = open(sys.argv[1]).readlines()

trainImages = [image.strip().split('/')[-1] for image in trainImages]

for i in range(len(indexes)):
    image = df['imagePath'][indexes[i]].split('/')[-1]
    if image in trainImages:
        ages[image] = df['ageInt'][indexes[i]]

pickle.dump(ages, open("ages.pkl", "wb"))



