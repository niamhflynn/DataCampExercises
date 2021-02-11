import pandas as pd

data = pd.read_csv("VAR PATH")

missing_values_count = data.isnull().sum()
#gives sum of null in all columns
print(missing_values_count)

new_data = data.dropna(axis=1)
#axis = 1 -> python thinks you are performing function columnwise
#axis = 0 -> function row wise
print(new_data.shape)

cleaned_data = new_data.fillna(0)

cleaned_data = new_data.fillna(method='bfill', axis=0).fillna(0)
#bfill - method to take in next possible value in same col, and replace with this
drop_duplicates = new_data.dropduplicates(subset = ['type'])
#subset gives deciding factors of output
#drop duplicates doesn't work on columns (ie axes =1)



