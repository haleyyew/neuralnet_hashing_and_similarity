from builtins import enumerate, max, open

from mutagen.mp3 import MP3
from mutagen.id3 import ID3, APIC, error
import sys
import os
from operator import itemgetter

root_dir = "/Users/haoran/Google Drive/GoogleMusic/"
pic_root_dir = "/Users/haoran/Google Drive/music_album_cover/"
for root, dirs, files in os.walk(root_dir):
    path = root.split(os.sep)
    for file in files:
        if file == '.DS_Store': continue
        mp3file = os.path.join(root, file)
        print(mp3file)

        audio = MP3(mp3file, ID3=ID3)

        try:
            audio.add_tags()
        except error:
            pass

        foldername, file_extension = os.path.splitext(file)
        foldername = foldername.split('(')[0]

        pic_file_folder = os.path.join(pic_root_dir, foldername)
        print(pic_file_folder)

        pic_files = [os.path.join(pic_file_folder, f) for f in os.listdir(pic_file_folder)]
        pic_files_sizes = [os.path.getsize(pic_file) for pic_file in pic_files]

        index, element = max(enumerate(pic_files_sizes), key=itemgetter(1))
        pic_file = pic_files[index]
        print(pic_file)

        imagedata = open(pic_file, 'rb').read()

        audio.tags.add(
           APIC(
              encoding=3,
              mime='image/jpeg',
              type=3,
              desc=u'Cover',
              data=imagedata
           )
        )
        audio.save()

        print('-----')