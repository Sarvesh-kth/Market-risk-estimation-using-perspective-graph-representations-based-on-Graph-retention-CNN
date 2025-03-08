import os

directory_path = 'Graphs'
file_names = os.listdir(directory_path)

file_names_sliced = [file_name[:-6] for file_name in file_names]

print(file_names_sliced)
