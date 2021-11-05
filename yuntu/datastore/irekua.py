import requests
from dateutil.parser import parse as dateutil_parse
import datetime

from yuntu.datastore.base import RemoteStorage

class IrekuaDatastore(RemoteStorage):

    def __init__(self, *args, page_size=10, page_start=0, page_end=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.page_start = page_start
        self.page_end = page_end
        self.page_size = page_size

    def iter_pages(self):
        page_size = self.page_size
        for page_number in range(self.page_start, self.page_end):

            url = self.metadata_url
            if "?" not in url:
                url = url + "?"
            elif url[-1] != "&":
                url = url + "&"
            url = url + f"page_size={page_size}&page={page_number}"

            res = requests.get(url, auth=self.auth)

            if res.status_code != 200:
                res = requests.get(url, auth=self.auth)
                raise ValueError(str(res))

            res_json = res.json()
            res_json["page_url"] = url

            yield res_json

    def iter(self):
        for page in self.iter_pages():
            for item in page["results"]:
                item["page_url"] = page["page_url"]
                yield item

    def iter_annotations(self, datum):
        return []

    def prepare_annotation(self, datum, annotation):
        pass

    def prepare_datum(self, datum):
        path = datum["item_file"]
        samplerate = datum["media_info"]["sampling_rate"]
        media_info = {
            'nchannels': datum["media_info"]["channels"],
            'sampwidth': datum["media_info"]["sampwidth"],
            'samplerate': samplerate,
            'length': datum["media_info"]["frames"],
            'filesize': datum["filesize"],
            'duration': datum["media_info"]["duration"]
        }
        spectrum = 'ultrasonic' if samplerate > 50000 else 'audible'
        metadata = {
            'item_url': datum["url"],
            'page_url': datum["page_url"]
        }

        dtime_zone = datum["captured_on_timezone"]
        dtime = dateutil_parse(datum["captured_on"])
        dtime_format = "%H:%M:%S %d/%m/%Y (%z)"
        dtime_raw = datetime.datetime.strftime(dtime, format=dtime_format)

        return {
            'id': datum['id'],
            'path': path,
            'hash': datum["hash"],
            'timeexp': 1,
            'media_info': media_info,
            'metadata': metadata,
            'spectrum': spectrum,
            'time_raw': dtime_raw,
            'time_format': dtime_format,
            'time_zone': dtime_zone,
            'time_utc': dtime
        }


    def get_metadata(self):
        meta = super().get_metadata()
        meta = {"type":"IrekuaDatastore"}
        return meta

    @property
    def size(self):
        if self._size is None:
            self._size = (self.page_end - self.page_start)*self.page_size
        return self._size
