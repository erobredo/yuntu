import json
import urllib3

def http_client():
    return urllib3.PoolManager()

def get_sync(client, url, params=None, auth=None):
    headers=None
    if auth is not None:
        headers = urllib3.make_headers(basic_auth=auth)
    res = client.request('GET',  url,
                         fields=params,
                         headers=headers
                         )
    if res.status != 200:
        res = client.request('GET', url,
                             fields=params,
                             headers=headers
                             )
        if res.status != 200:
            raise ValueError(f"Server error {res.status}")

    return json.loads(res.data.decode('utf-8'))

def post_sync(client, url, params=None, auth=None):
    headers=None
    if auth is not None:
        headers = urllib3.make_headers(basic_auth=auth)
    body = json.dumps(params)
    res = client.request('POST',  url,
                         body=body,
                         headers=headers
                         )
    if res.status != 200:
        res = client.request('POST', url,
                             body=body,
                             headers=headers
                             )
        if res.status != 200:
            raise ValueError(f"Server error {res.status}")

    return json.loads(res.data.decode('utf-8'))
