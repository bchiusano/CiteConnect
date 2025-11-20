import pandas as pd

DEBUG = False

excel_df =pd.read_excel("data/Dataset Advice letters on objections towing of bicycles.xlsx")
letters = excel_df['geanonimiseerd_doc_inhoud']
case_numbers = excel_df['Octopus zaaknummer']

for i in range(len(case_numbers)):
    number = case_numbers[i]
    letter = letters[i]
    
    if DEBUG:
        print(number)
        print(letter)

    
    with open(file="data/letters/{}.txt".format(number), mode="w+") as f:
        f.write(letter)

    