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

    files['FileName'] = files['FileName'].str.lower()
    duplicates = files.duplicated('FileName', keep=False)
    duplicates = files[duplicates]

    for fileName in duplicates['FileName'].unique():
        file_dup = duplicates[duplicates['FileName'] == fileName].reset_index(drop=True)
        write_path = ""
        read_path = ""
        for i in range(file_dup.shape[0]):
            path = file_dup['DirPath'][i] + "/" + file_dup['FileName'][i]
            if write_path == "" and "Sorted" in path:
                write_path = path
            else:
                read_path = path
        with open(write_path, 'a', encoding='utf-8') as write_file:
            with open(read_path, 'r', encoding='utf-8') as read_file:
                lines = read_file.readlines()
                write_file.writelines(lines)
        os.remove(read_path)

if __name__ == "__main__":
    main()