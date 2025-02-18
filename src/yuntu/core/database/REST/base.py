from abc import ABC
from abc import abstractmethod

class RESTManager(ABC):
    """Managet to fetch information on the fly from irekua REST api"""

    def __init__(self, provider, config):
        self.provider = provider
        self.init_configs(config)
        self.recordings_url = self.build_recordings_url()
        self.models = self.build_models()

    def init_configs(self, config):
        self.api_url = config["api_url"]
        if "version" in config:
            self.version = config["version"]
        else:
            self.version = None
        self.page_size = config["page_size"]

        if "auth" in config:
            self.auth = config["auth"]
        else:
            self.auth = None

        self.base_filter = config["base_filter"]

    @abstractmethod
    def build_recordings_url(self):
        pass

    def select(self, query=None, limit=None, offset=None, model="recording"):
        """Query entries from database."""

        model_class = self.get_model_class(model)

        return model_class.select(query=query, limit=limit, offset=offset)

    def get_model_class(self, model):
        """Return model class if exists or raise error."""
        model_dict = self.models._asdict()

        if model not in model_dict:
            options = model_dict.keys()
            options_str = ', '.join(options)
            message = (
                f'The model {model} is not available in this REST database. '
                f'Admisible options: {options_str}')
            raise NotImplementedError(message)
        return model_dict[model]

    @abstractmethod
    def build_models(self):
        """Construct all database entities."""

    @abstractmethod
    def build_recording_model(self):
        """Build REST recording model"""
