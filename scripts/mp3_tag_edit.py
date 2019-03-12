from builtins import enumerate, max, open

from mutagen.mp3 import MP3
from mutagen.id3 import ID3, APIC, error
import sys
import os
from operator import itemgetter
from mutagen.id3 import ID3NoHeaderError
from mutagen.id3 import ID3, TIT2, TALB, TPE1, TPE2, COMM, TCOM, TCON, TDRC, TRCK

root_dir = "/Users/haoran/Downloads/浜崎あゆみ - A BALLADS/"
album_name = u"Ultimate Piano Concertos The Essential Masterpieces (CDs 1-2)"
artist = u"Decca"
track_number = 1
# print(album_name, artist)
fname_list = []
for root, dirs, files in os.walk(root_dir):
    path = root.split(os.sep)
    for file in files:
        if file == '.DS_Store': continue
        fname = os.path.join(root, file)
        fname_list.append(fname)

fname_list.sort()
for fname in fname_list:
  try: 
      tags = ID3(fname)
  except ID3NoHeaderError:
      print("Adding ID3 header")
      tags = ID3()

  track_num = str(track_number)
  # tags["TIT2"] = TIT2(encoding=3, text=title)
  # tags["TALB"] = TALB(encoding=3, text=album_name)
  # tags["TPE1"] = TPE1(encoding=3, text=artist)
  tags["TRCK"] = TRCK(encoding=3, text=track_num)

  tags.save(fname)

  print(track_number, fname)
  track_number += 1