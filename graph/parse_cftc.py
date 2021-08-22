import requests
from bs4 import BeautifulSoup

URL = 'https://www.cftc.gov/dea/futures/financial_lf.htm'
HEADERS = {'accept':
           'text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,\
        image/webp,image/apng,*/*;q=0.8,application/signed-exchange;\
        v=b3;q=0.9', 'user-agent':
           'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, \
        like Gecko) Chrome/86.0.4240.111 Safari/537.36'}


def get_html(url, params=None):
    return requests.get(url, headers=HEADERS, params=params)


def get_data_from_html(URL):
    html = get_html(URL)
    soup = BeautifulSoup(html.text, 'html.parser')
    items = soup.find('pre')
    return items.text.split(
        "-----------------------------------------------------------------------------------------------------------------------------------------------------------"
    )


def processing_data(data):
    bitcoin_data = [i for i in data if i.find('BITCOIN') != -1]
    bitcoin_data = bitcoin_data[0].split('\r')
    bitcoin_data = [i.split('\n') for i in bitcoin_data]
    bitcoin_data = [[s for s in i if s not in ["", ' ']]
                    for i in bitcoin_data]
    bitcoin_data = list(filter(None, bitcoin_data))
    bitcoin_data = [[" ".join(s.split()) for s in i]for i in bitcoin_data]
    bitcoin_data = [[s.replace('.', '0') for s in i]for i in bitcoin_data]
    return bitcoin_data


def get_rows(bitcoin_data):
    key_word = ['positions', 'changes', 'percent', 'number']
    name_rows = []
    data_f = []
    for k in key_word:
        for i in range(len(bitcoin_data)):
            if bitcoin_data[i][0].lower().find(k) != -1:
                name_rows.append(bitcoin_data[i][0])
                data_f.append(bitcoin_data[i+1][0].split())
    return name_rows, data_f


def get_timestano(name_rows):
    timestamp = list(filter(lambda s: s.find("from: ") != -1, name_rows))
    if timestamp:
        index = timestamp[0].split(',')
        index = [i.strip() for i in index]
        index = [i.split() for i in index]
        timestamp = f'{" ".join(index[0][-2:])} {index[1][0]}'
    return timestamp


def get_headers(data):
    headers = list(filter(None, data[1].split(':')))
    headers = [i.strip() for i in headers]
    headers = list(filter(None, headers))
    return headers


def parse_cftc_to_json():

    data = get_data_from_html(URL)
    bitcoin_data = processing_data(data)
    name_rows, data_f = get_rows(bitcoin_data)
    timestamp = get_timestano(name_rows)
    headers = get_headers(data)

    return {headers[i]: {
        'timestamp': timestamp,
        'values': data_f[i],
        'name_rows': name_rows[i],
        'name_columns': headers[-12:]
    } for i in range(len(headers[:4]))}


df = parse_cftc_to_json()
