import json
import urllib3

def http_client(timeout=30.0, retry=10, forcelist=(504, )):
    rtimeout = urllib3.util.Timeout(total=timeout)
    retries = urllib3.util.Retry(total=retry, status_forcelist=forcelist)

    return urllib3.PoolManager(retries=retries, timeout=rtimeout)

def get_sync(client, url, params=None, auth=None, headers=None):

    if auth is not None:
        req_headers = urllib3.make_headers(basic_auth=auth)
        if headers is not None:
            req_headers.update(headers)
    else:
        req_headers = headers

    res = client.request('GET',  url,
                         fields=params,
                         headers=req_headers
                         )
    if res.status != 200:
        res = client.request('GET', url,
                             fields=params,
                             headers=req_headers
                             )
        if res.status != 200:
            raise ValueError(f"Server error {res.status}")

    return json.loads(res.data.decode('utf-8'))

def post_sync(client, url, params=None, auth=None, headers=None):

    if auth is not None:
        req_headers = urllib3.make_headers(basic_auth=auth)
        if headers is not None:
            req_headers.update(headers)
    else:
        req_headers = headers

    body = json.dumps(params)
    res = client.request('POST',  url,
                         body=body,
                         headers=req_headers
                         )

    if res.status != 200:
        res = client.request('POST', url,
                             body=body,
                             headers=req_headers
                             )
        if res.status != 200:
            raise ValueError(f"Server error {res.status}")

    return json.loads(res.data.decode('utf-8'))
