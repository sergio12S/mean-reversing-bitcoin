import requests
import json


def get_data():
    data = requests.get('https://fapi.bybt.com/api/futures/coins/markets')
    if data.status_code == 200:
        return json.loads(data.text).get('data')


data = get_data()
