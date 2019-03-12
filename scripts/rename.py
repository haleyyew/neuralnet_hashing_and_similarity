import os
import shutil

root_dir = "/Users/haoran/Google Drive/GoogleMusic/"
for root, dirs, files in os.walk(root_dir):
    path = root.split(os.sep)
    for file in files:
        from_path = os.path.join(root, file)

        foldername, file_extension = os.path.splitext(file)
        # print(foldername, file_extension)
        if file_extension == '.mp3':
            foldername = foldername.split('(')[0]
            to_path = os.path.join(root, foldername, file)
            print(from_path, to_path)
            shutil.move(from_path, to_path)