import pandas as pd

file_name = input('Enter the file name. For example "down-4.txt" and should be in the same folder:   ')
#out_name = input('"Enter the output name for the file, For example "test.csv" it will generate "test.csv":   ')

df = pd.read_csv(file_name)

df.drop(df.columns[0], axis=1, inplace=True)

df.drop(df.iloc[:, 4:], axis=1, inplace=True)

df.to_csv(file_name, header=False, index=False)