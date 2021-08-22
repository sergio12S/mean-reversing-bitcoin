import requests


URL = 'https://api.glassnode.com/v1/metrics/addresses/active_count?a=BTC&i=24h'
HEADERS = {'accept':
           'text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,\
        image/webp,image/apng,*/*;q=0.8,application/signed-exchange;\
        v=b3;q=0.9', 'user-agent':
           'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, \
        like Gecko) Chrome/86.0.4240.111 Safari/537.36'}


df = requests.get(URL, headers=HEADERS, params=None)
df.status_code