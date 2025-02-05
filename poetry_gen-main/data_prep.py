import re
import pandas as pd
import datasets
import torch
from torch.utils.data import Dataset
import pandas as pd

datasources = ['Indic_ShareLlama','Dolly_T', 'OpenAssistant_T','IndoWordNet']

for item in datasources : 
  pattern = r"\['([^']+)', '([\s\S]+?)'\]"
  questions=[]
  answers=[]
  df = datasets.load_dataset("ai4bharat/indic-align", item)
  for i in range (0,len(df['train']['eng_Latn'])):
   data = str(df['train']['eng_Latn'][i])
   match = re.search(pattern, data)
   if match:
       question = match.group(1)
       answer = match.group(2)
       questions.append(question)
       answers.append(answer)

    ds = pd.DataFrame({
    'questions': questions,
    'answers': answers
    })

   ds.to_csv('ift_data.csv', index= False, mode ='a', header = False)









