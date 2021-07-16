import os

import numpy as np
import pandas as pd

def parse_toefl(type):
	path_index_dir = "/data/nlp/jeonso/dataset/toefl/data/text/"
	path_dir_text = path_index_dir + "responses/original/"
	path_output = "/data/nlp/jeonso/dataset/toefl/data/text/"
	name_output = type + "_toefl_essay.csv"  # training, dev, or test

	path_index_file = os.path.join(path_index_dir, "index-" + type + ".csv")  

	pd_index = None
	pd_out = None
	if type is not "test":
		pd_index = pd.read_csv(path_index_file, names=['Filename', 'Prompt', 'Language', 'Score Level'])
		pd_out = pd_index.copy()
		pd_out = pd_out.rename({'Filename': 'essay_id', 'Prompt':'prompt', 'Language':'native_lang', 'Score Level': 'essay_score'}, axis='columns')
	else:
		pd_index = pd.read_csv(path_index_file, names=['Filename', 'Prompt', 'Score Level'])
		pd_out = pd_index.copy()
		pd_out = pd_out.rename({'Filename': 'essay_id', 'Prompt':'prompt', 'Score Level': 'essay_score'}, axis='columns')

	print(pd_index.columns)  # Index(['Filename', 'Prompt', 'Language', 'Score Level'], dtype='object')
	list_essay_text = []
	for index, row in pd_index.iterrows():
		cur_filename = row["Filename"]
		cur_prompt = row["Prompt"]
		# if type is not "test":
		# 	cur_nlang = row["Language"]
		cur_label = row['Score Level']

		# id from filename
		cur_id = os.path.splitext(cur_filename)[0]
		pd_out.at[index,'essay_id'] = cur_id

		# prompt
		cur_prompt = cur_prompt[1:]
		pd_out.at[index,'prompt'] = cur_prompt

		# label
		if cur_label == "low": 
			cur_label = 0
		elif cur_label == "medium":
			cur_label = 1
		elif cur_label == "high":
			cur_label = 2
		pd_out.at[index,'essay_score'] = cur_label
		
		# text
		cur_essay_file = open(os.path.join(path_dir_text, cur_filename), 'r')
		cur_text = cur_essay_file.read()
		cur_essay_file.close()
		list_essay_text.append(cur_text)
	# end for index
	
	pd_out.insert(len(pd_out.columns), 'essay', list_essay_text)
	print(pd_out.head())

	pd_out.to_csv(os.path.join(path_output, name_output), index=False)

	# # verify result
	# pd_test = pd.read_csv(os.path.join(path_output, name_output))
	# for index, row in pd_test.iterrows():
	# 	print(row)

	return

#
if __name__ == "__main__":
	list_type = ["training", "dev", "test"]
	for cur_type in list_type:
		parse_toefl(cur_type)





