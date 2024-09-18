pytorch_reid_auto_train_and_eval_closed_and_open.ipynb
Notebook to train reid SCL model
Uses regular expressions to loop through the csv files inside of the premade directories 
(e.g. new_open_04_ids_all_colors_batch2, new_open_08_ids_monocolor_batch1, etc.)

With little modifications, it should save any desired model in its own directory inside of a models_trained directory. For each CSV it creates a yml_config that is saved in the model directory to perform training and evaluation using pytorch_train_and_eval_reid_2.py file. 
