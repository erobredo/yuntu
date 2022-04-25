from abc import ABC
from abc import abstractmethod

import math
from yuntu.core.database.REST.utils import http_client, post_sync

MAX_PAGE_SIZE = 10000

class as_object:
    def __init__(self, datum):
        for key in datum:
            setattr(self, key, datum[key])

    def to_dict(self):
        return self.__dict__

class RESTModel(ABC):
    """A base class for all REST models"""

    def __init__(self, target_url,
                 page_size=1000, auth=None,
                 base_filter=None, **kwargs):
        self.target_url = target_url
        self._auth = auth
        self._page_size = min(page_size, MAX_PAGE_SIZE)
        self._http = http_client()
        self.base_filter = {}

        if base_filter is not None:
            self.base_filter = base_filter


    @property
    def page_size(self):
        return self._page_size

    @property
    def auth(self):
        return self._auth

    def fetch_sync(self, url, params=None, auth=None, headers=None):
        """HTTP method for information retrieval"""
        return post_sync(client=self._http, url=url, params=params, auth=auth, headers=headers)

    def set_page_size(self, page_size):
        self._page_size = page_size

    def set_auth(self, auth):
        self._auth = auth

    def select(self, query=None, limit=None, offset=None):
        """Request results and return"""
        # Add limits to request and traverse pages
        if limit is not None:
            count = 0
            for page in self.iter_pages(query, limit, offset):
                meta_arr = self.extract_entries(page)
                for meta in meta_arr:
                    if count > limit:
                        break
                    parsed = self.parse(meta)
                    count += 1
                    yield as_object(parsed)
        else:
            for page in self.iter_pages(query, limit, offset):
                meta_arr = self.extract_entries(page)
                for meta in meta_arr:
                    parsed = self.parse(meta)
                    yield as_object(parsed)

    def count(self, query=None, **kwargs):
        """Request results count"""
        vquery = self.validate_query(query)
        return self.result_size(vquery, **kwargs)

    def iter_pages(self, query=None, limit=None, offset=None, **kwargs):
        vquery = self.validate_query(query)
        rec_count = self.count(query)

        if limit is None:
            npages = math.ceil(float(rec_count)/float(self.page_size))
        else:
            npages = math.ceil(float(limit)/float(self.page_size))

        if offset is None:
            offset = 0

        headers = self.build_headers(**kwargs)

        for page in range(npages):
            params = self.build_request_params(vquery,
                                               self.page_size,
                                               offset + page*self.page_size)

            yield self.fetch_sync(self.target_url,
                                  params=params, auth=self.auth,
                                  headers=headers)

    def build_request_params(self, query, limit, offset, sortby=None):
        """Use query to build specific HTTP parameters"""
        params = self.build_query_params(query)
        params.update(self.build_paging_params(limit, offset))
        #params.update(self.build_sorting_params(sortby))

        return params

    @abstractmethod
    def extract_entries(self, page):
        """Process page and produce a list of entries"""

    @abstractmethod
    def validate_query(self, query):
        """Check if query is valid for REST service"""

    def build_headers(self, **kwargs):
        """Use query to build specific HTTP parameters"""
        return None

    @abstractmethod
    def build_query_params(self, query):
        """Use query to build specific HTTP parameters"""

    @abstractmethod
    def build_paging_params(self, limit, offset):
        """Use query to build specific HTTP parameters"""

    @abstractmethod
    def build_sorting_params(self, sortby):
        """Use query to build specific HTTP parameters"""

    @abstractmethod
    def result_size(self, query=None, **kwargs):
        """Fetch the number of results in query"""

    @abstractmethod
    def parse(self, datum, **kwargs):
        """Parse incoming data to a common format"""


class RESTAnnotation(RESTModel, ABC):

    @abstractmethod
    def get_recording(self, datum):
        """Get annotation's recording from reference"""

    @abstractmethod
    def parse_annotation(self, datum):
        """Parse annotation's labels from datum"""

    def parse(self, datum, **kwargs):
        """Parse incoming data to a common format"""
        recording = self.get_recording(datum)
        meta = self.parse_annotation(datum, **kwargs)
        meta["recording"] = recording
        return meta

class RESTRecording(RESTModel, ABC):

    @abstractmethod
    def parse_recording(self, datum, **kwargs):
        """Parse yuntu fields from datum"""

    def parse(self, datum, **kwargs):
        """Parse incoming data to a common format"""
        meta = self.parse_recording(datum, **kwargs)
        return meta
