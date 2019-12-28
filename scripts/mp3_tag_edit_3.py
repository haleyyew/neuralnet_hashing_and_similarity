from builtins import enumerate, max, open

from mutagen.mp3 import MP3
from mutagen.id3 import ID3, APIC, error
import sys
import os
from operator import itemgetter
from mutagen.id3 import ID3NoHeaderError
from mutagen.id3 import ID3, TIT2, TALB, TPE1, TPE2, COMM, TCOM, TCON, TDRC, TRCK
import re
import pprint

from argparse import ArgumentParser

parser = ArgumentParser()
parser.add_argument("-nt", "--skip_track_listing", dest="skip_track_listing", type=int, choices=[0,1], default=0)
parser.add_argument("-nn", "--skip_titles", dest="skip_titles", type=int, choices=[0,1], default=0)

args = parser.parse_args()
# print(args.skip_track_listing)

root_dir = "/Users/haoran/Google Drive/untitled folder/"
directories = {}

for root, dirs, files in os.walk(root_dir):

    path = root.split(os.sep)
    directories[root] = []
    # print(root)
    for file in files:
        if file == '.DS_Store': continue
        if root == root_dir: continue
        if '.mp3' not in file: continue
        directories[root].append(file)
        art_alb = root.split('/')[-1]
        art_alb = art_alb.split(' - ')
        artist = art_alb[0]
        # print(art_alb)
        album_name = art_alb[1].split(' (')[0]
        try:
            year = re.search(r'\((.+?)\)', art_alb[1]).group(1)
        except Exception:
            year = None

        fname = os.path.join(root, file)
        try: 
            tags = ID3(fname)
            tags = ID3()
        except ID3NoHeaderError:
            print("Adding ID3 header")
            tags = ID3()

        file_title = file.split('.')
        if file_title != None and len(file_title) != 0: file_title = file_title[1]

        tags["TALB"] = TALB(encoding=3, text=album_name)
        tags["TPE1"] = TPE1(encoding=3, text=artist)
        if args.skip_titles == 0: tags["TIT2"] = TIT2(encoding=3, text=file_title)
        if year != None:
            tags["DATE"] = TRCK(encoding=3, text=year)

        tags.save(fname)
        # print(fname.split('/')[-1], album_name, artist, year)


# pprint.pprint(directories)
for direc in directories:
    directory = directories[direc]
    directory.sort()
    track_number = 1

    tracks_p = direc + '/tracks.txt'
    tracks = ['']*len(directory)    # Hack
    if os.path.exists(tracks_p):
        tracksdata = open(tracks_p, 'r')
        tracks = tracksdata.readlines()
    print(tracks)

    for file in directory:
        if '.mp3' not in file: continue
        fname = direc + '/' + file

        try: 
            tags = ID3(fname)
        except ID3NoHeaderError:
            print("Adding ID3 header")
            tags = ID3()        

        title_new = tracks[track_number-1]
        title_new = title_new.strip()
        if args.skip_titles == 0: tags["TIT2"] = TIT2(encoding=3, text=title_new)   # Get name from txt

        track_num = str(track_number)    
        if args.skip_track_listing == 0: tags["TRCK"] = TRCK(encoding=3, text=track_num)

        tags.save(fname)
        print(tags["TIT2"], track_num)
        track_number += 1


for direc in directories:
    directory = directories[direc]
    directory.sort()
    # print(direc, directory)
    if len(directory) == 0: continue

    pic_file = direc + '/folder.jpg'
    if not os.path.exists(pic_file): continue
    try:
        imagedata = open(pic_file, 'rb').read()
    except Exception:
        print(pic_file)
        continue

    for file in directory:
        if '.mp3' not in file: continue
        mp3file = direc + '/' + file
        audio = MP3(mp3file, ID3=ID3)

        try:
            audio.add_tags()
        except error:
            pass

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