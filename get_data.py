
import requests
import zipfile

def download_and_extract_zip(url, extract_to='.'):
    print("Downloading and extracting data...")
    response = requests.get(url)
    if response.status_code == 200:
        with open('data.zip', 'wb') as f:
            f.write(response.content)
            print("successfully downloaded data.zip")
        with zipfile.ZipFile('data.zip', 'r') as zip_ref:
            zip_ref.extractall(extract_to)
            print("successfully extracted data.zip")
    else:
        print(f"Failed to download data. Status code: {response.status_code}")
url = "https://github.com/karoldvl/ESC-50/archive/master.zip"
data_dir = "data"
import os 
os.makedirs(data_dir, exist_ok=True)
download_and_extract_zip(url, extract_to=data_dir)

