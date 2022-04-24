import os
import json
from collections import namedtuple
from dateutil.parser import parse as dateutil_parse
import datetime
import math

from yuntu.core.database.REST.base import RESTManager
from yuntu.core.database.REST.models import RESTRecording, RESTAnnotation

MODELS = [
    'recording',
    # 'annotation'
]

Models = namedtuple('Models', MODELS)

class AlfrescoMixin:
    """A mixin that controls general aspects of Alfresco
    interactions"""

    include = None

    def result_size(self, query=None, **kwargs):
        """Fetch the number of results in query"""
        params = {
            "query": {
                "query": query
            },
            "paging": {
                "maxItems": "1",
                "skipCount": "0"
            }
        }

        headers = self.build_headers(**kwargs)

        return self.fetch_sync(self.target_url,
                               params=params,
                               auth=self.auth,
                               headers=headers)["list"]["pagination"]["totalItems"]

    def validate_query(self, query):
        """Check if query is valid for REST service"""
        if query is None:
            return self.base_filter

        if not isinstance(query, str):
            raise ValueError("When using Alfresco collections, queries should " +
                             "be specified with a string holding filters.")

        return self.base_filter + " AND " + query

    def build_query_params(self, query):
        """Use query to build specific HTTP parameters"""
        return {
            "query": {
                "query": query,
                "language": "afts"
                }
        }

    def build_paging_params(self, limit, offset):
        """Use limit and offset to build specific HTTP parameters"""
        return {
            "paging": {
                "maxItems": str(limit),
                "skipCount": str(offset)
            }
        }

    def build_sorting_params(self, sortby):
        """Use sorting specification to build specific HTTP parameters"""
        return {
            "sort": sortby
        }

    def build_headers(self, **kwargs):
        """Use query to build specific HTTP parameters"""
        if self.api_key is None:
            return None
        return {"x-api-key": self.api_key}

class AlfrescoAnnotation(AlfrescoMixin, RESTAnnotation):

    def get_recording(self, datum):
        """Get annotation's recording from reference"""
        pass

    def parse_annotation(self, datum):
        """Parse annotation's labels from datum"""
        pass


class AlfrescoRecording(AlfrescoMixin, RESTRecording):
    include = ["path", "properties"]

    def __init__(self, *args, path_extractor, parser, **kwargs):
        self.path_extractor = path_extractor
        self.parser = parser
        super().__init__(*args,*kwargs)

    def get_path(self, datum):
        """Get path to recording from datum"""
        return self.path_extractor(datum)

    def parse_recording(self, datum):
        """Parse yuntu fields from datum"""
        return self.parser(datum)

    def build_request_params(self, query, limit, offset, sortby=None):
        """Use query to build specific HTTP parameters"""
        params = super().build_request_params(query, limit, offset, sortby)
        params.update({"include": self.include})

        return params

    def extract_entries(self, page):
        return page["list"]["entries"]

class AlfrescoREST(RESTManager):

    def init_configs(self, config):
        super().init_configs(config)

        if "api_key" in config:
            self.api_key = config["api_key"]
        else:
            self.api_key = None

        self.path_extractor = config["path_extractor"]
        self.recording_parser = config["recording_parser"]

    def build_recordings_url(self):
        return f"{self.api_url}/alfresco/search/"

    def build_models(self):
        """Construct all database entities."""
        recording = self.build_recording_model()
        models = {
            'recording': recording,
        }
        return Models(**models)

    def build_recording_model(self):
        """Build REST recording model"""
        return AlfrescoRecording(target_url=self.recordings_url,
                                 target_attr="list",
                                 page_size=self.page_size,
                                 base_filter=self.base_filter,
                                 auth=self.auth,
                                 api_key=self.api_key,
                                 path_extractor=self.path_extractor,
                                 parser=self.recording_parser)

    #def build_annotation_model(self):
