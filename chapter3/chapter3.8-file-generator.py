import urllib.request
import zipfile
import os

# 다운로드할 파일 URL
train_url = "https://storage.googleapis.com/learning-datasets/rps.zip"
valid_url = "https://storage.googleapis.com/learning-datasets/rps-test-set.zip"

# 저장할 파일 이름
train_zip = "rps.zip"
valid_zip = "rps-test-set.zip"

# 압축 해제할 경로 (폴더명 수정)
train_dir = "rps/training/"
valid_dir = "rps/validation/"

# 데이터 폴더 생성 (없으면 생성)
os.makedirs(train_dir, exist_ok=True)
os.makedirs(valid_dir, exist_ok=True)

# 데이터 다운로드
urllib.request.urlretrieve(train_url, train_zip)
urllib.request.urlretrieve(valid_url, valid_zip)

# 압축 풀기 (올바른 폴더에 저장)
with zipfile.ZipFile(train_zip, "r") as zip_ref:
    zip_ref.extractall(train_dir)
    zip_ref.close()

with zipfile.ZipFile(valid_zip, "r") as zip_ref:
    zip_ref.extractall(valid_dir)
    zip_ref.close()

print("✅ 압축 해제 완료!")