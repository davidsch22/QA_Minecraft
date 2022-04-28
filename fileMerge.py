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
        sources = file_dup['RootFolder'].unique()
        if len(sources) < file_dup.shape[0]:
            for source in sources:
                source_files = file_dup[file_dup['RootFolder'] == source].reset_index(drop=True)
                if source_files.shape[0] > 1:
                    path = source_files['DirPath'][1] + "/" + source_files['FileName'][1]
                    os.remove(path)
        if file_dup['RootFolder'].unique() > 0:
            pass

if __name__ == "__main__":
    main()