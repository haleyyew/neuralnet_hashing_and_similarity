from google_images_download import google_images_download
import os

root_dir = "/Users/haoran/Google Drive/GoogleMusic/"
for root, dirs, files in os.walk(root_dir):
    path = root.split(os.sep)
    for file in files:
        down_path = os.path.join(root, file)
        filename, file_extension = os.path.splitext(file)

        filename = filename.split('(')[0]

        response = google_images_download.googleimagesdownload()
        absolute_image_paths = response.download({"keywords":filename,
        	"limit":4,
        	"print_urls":True,
        	"aspect_ratio":"square",
        	"format":"jpg",
        	"size":"medium"})
        print(absolute_image_paths)

# export PATH="/Library/Frameworks/Python.framework/Versions/3.6/bin"