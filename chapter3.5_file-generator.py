import urllib.request
import zipfile

url = "https://storage.googleapis.com/learning-datasets/horse-or-human.zip"
url2 = "https://storage.googleapis.com/learning-datasets/validation-horse-or-human.zip"

file_name = "horse-or-human_training.zip"
file_name2 = "horse-or-human_validation.zip"

valid_dir = "horse-or-human/training/"
training_dir = "horse-or-human/validation/"

urllib.request.urlretrieve(url, file_name)
urllib.request.urlretrieve(url2, file_name2)

zip_ref = zipfile.ZipFile(file_name, "r")
zip_ref.extractall(training_dir)
zip_ref.close()

zip_ref = zipfile.ZipFile(file_name2, "r")
zip_ref.extractall(valid_dir)
zip_ref.close()