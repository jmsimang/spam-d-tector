import requests
import shutil


def get_data_from_url(url):
    filename = url.split('/')[-1]
    with requests.get(url, stream=True) as request:
        with open(filename, 'wb') as file:
            shutil.copyfileobj(request.raw, file)
