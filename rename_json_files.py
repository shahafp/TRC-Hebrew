import os

res = list(os.walk('INCEPTION/inception_round_3/annotation'))[1:]
for file_dir,_,file_names in res:
    for f_name in file_names:
        if f_name.startswith('guy'):
            file_path = os.path.join(file_dir,f_name)
            new_file_path = os.path.join(file_dir,'guy.json')
            os.rename(file_path,new_file_path)