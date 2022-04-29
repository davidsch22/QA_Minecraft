from distutils.file_util import write_file
import os
import pandas as pd

def main():
    files = pd.DataFrame(columns=['RootFolder','DirPath','FileName'])
    for (dirpath, dirnames, filenames) in os.walk('KnowledgeDatabase/'):
        if filenames:
            for fileName in filenames:
                rootFolder = ""
                if "Sorted" in dirpath:
                    rootFolder = "Sorted"
                elif "GuidesTxt" in dirpath:
                    rootFolder = "GuidesTxt"
                elif "GamepediaTxt" in dirpath:
                    rootFolder = "GamepediaTxt"
                files.loc[len(files.index)] = [rootFolder, dirpath, fileName]

    duplicates = files.duplicated('FileName', keep=False)
    duplicates = files[duplicates]

    for fileName in duplicates['FileName'].unique():
        file_dup = duplicates[duplicates['FileName'] == fileName]
        write_path = ""
        read_path = ""
        for i in range(file_dup.shape[0]):
            path = file_dup['DirPath'][i] + "/" + file_dup['FileName'][i]
            if "Sorted" in path:
                write_path = path
            else:
                read_path = path
        with os.open(write_path, 'a') as write_file:
            with os.open(read_path, 'r') as read_file:
                lines = read_file.readlines()
                write_file.writelines(lines)
        return

if __name__ == "__main__":
    main()