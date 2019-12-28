mypath = '/Users/haoran/Downloads/松田聖子 - Seiko Train/'
from os import listdir
from os.path import isfile, join
onlyfiles = [f for f in listdir(mypath) if isfile(join(mypath, f))]
import pprint

onlyfiles.sort()

import re
# all chars except -, until -
onlyfiles = [re.sub(r'^([^-]+)-', '', f) for f in onlyfiles]

onlyfiles2 = []
for f in onlyfiles:
	f1 = None

	# all chars between ( and )
	pat = re.search(r'.*?\((.*)\).*', f)
	if pat != None:
		# chars are in group 1
		f2 = f.replace(pat.group(1), '')
		f1 = f2
	else:
		f1 = f

	f1 = f1.replace('()', '')
	f1 = f1.replace('.mp3', '')
	f1 = f1.strip()
	onlyfiles2 += [f1]

pprint.pprint(onlyfiles2)

with open('titles.txt', 'a') as the_file:
	for f in onlyfiles2:
		the_file.write(f + '\n')

