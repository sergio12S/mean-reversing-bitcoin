import requests


s = requests.Session()


URL = 'https://fundingchoicesmessages.google.com/f/AGSKWxXC-OySB6QmfjE38h0U9oyi5t6kIzJcAE50S1IKOUYqqTBMfTcCtrwHg2QUeZV4L6a-DPUQiXuxVBHXK4yLGP0=?fccs=W1siQUtzUm9sLV9IRkEtQjlzZ0lEVm9YSzF4X1ZxYzA3TDByeENxTEdHYmlPWF82a09aV2luMENRZktSYThSRnEtRmZzT3JROGJRY2h0LVlNOV9OWnpWQUpVSkxUckdtb0pxX1drSzV3UTAweXNWUTkydXlzal9hWlplbmIweUxMdklPMTRsSWZSZHlQRlViVTZjZGNwYWk4RmNtU0pXZW53UVRBPT0iXSxbW10sW11dLG51bGwsbnVsbCxudWxsLDIsWzE2Mjk2NTQzMjIsNTQ5MDAwMDAwXSwiNDkyNTRFQjktQzcxQS00NEIyLTlBMEMtMEJDMkZEOTUxODJEIiwiNDgyN0I3RjctQTM0Qy00M0ZCLTkyMDgtNkIwOTgwQTYzNUQ5IixudWxsLFtudWxsLFs3XV0sImh0dHBzOi8vd3d3LnRyYWRpbmdzdGVyLmNvbS9jb3QvZnV0dXJlcy9maW4vMTMzNzQxIl0'
HEADERS = {'accept':
           'text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,\
        image/webp,image/apng,*/*;q=0.8,application/signed-exchange;\
        v=b3;q=0.9', 'user-agent':
           'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, \
        like Gecko) Chrome/86.0.4240.111 Safari/537.36'}


df = s.get(URL, headers=HEADERS, cookies=None)
df.status_code
