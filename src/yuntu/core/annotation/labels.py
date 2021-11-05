from collections import OrderedDict


class Label:
    def __init__(self, key, value, type=None):
        self.key = key
        self.value = value
        self.type = type

    @classmethod
    def from_dict(cls, data):
        return cls(**data)

    def to_dict(self):
        data = {
            'key': self.key,
            'value': self.value
        }

        if self.type is not None:
            data['type'] = self.type

        return data

    def is_type(self, type=None):
        if self.type is None and type is None:
            return True

        return self.type == type

    def __repr__(self):
        if self.type is None:
            return f'Label(key={self.key}, value={self.value})'

        return f'Label(key={self.key}, value={self.value}, type={self.type})'

    def __str__(self):
        return f'{self.key}: {self.value}'


class Labels:
    def __init__(self, labels=None):
        self.labels_dict = OrderedDict()

        if labels is None:
            labels = []

        for label in labels:
            if not isinstance(label, Label):
                label = Label.from_dict(label)

            if label.key in self.labels_dict:
                message = 'Label list has two values for the same key.'
                raise ValueError(message)
            self.labels_dict[label.key] = label

    def to_dict(self):
        return [
            label.to_dict()
            for label in self.labels_dict.values()
        ]

    @classmethod
    def from_dict(cls, data):
        return cls(labels=data)

    def add(self, key=None, value=None, type=None, data=None, label=None):
        if label is None:
            if data is None:
                data = {
                    'key': key,
                    'value': value,
                    'type': type}

            label = Label.from_dict(data)

        if not isinstance(label, Label):
            message = 'The provided label is not of the correct type'
            raise ValueError(message)

        if label.key in self.labels_dict:
            message = 'The provided label would overwrite another label'
            raise ValueError(message)

        self.labels_dict[label.key] = label

    def get(self, key):
        return self.labels_dict[key]

    def get_value(self, key, default=None):
        try:
            return self.labels_dict[key].value
        except KeyError:
            return default

    def get_type(self, key):
        return self.labels_dict[key].type

    def get_by_type(self, type=None):
        return [
            label for label in self.labels_dict.values()
            if label.is_type(type)
        ]

    def __contains__(self, key):
        return key in self.labels_dict

    def __getitem__(self, key):
        return self.labels_dict[key]

    def remove(self, key):
        del self.labels_dict[key]

    def get_text_repr(self, key):
        if key is None:
            return str(self)

        if isinstance(key, (tuple, list)):
            return '\n'.join([
                str(self.get(subkey)) for subkey in key
                if subkey in self.labels_dict
            ])

        return self.get_value(key)

    def __iter__(self):
        for label in self.labels_dict.values():
            yield label

    def iter_values(self):
        for label in self.labels_dict.values():
            yield label.value

    def __repr__(self):
        arguments = ', '.join([
            repr(label) for label in self
        ])
        return f'Labels([{arguments}])'

    def __str__(self):
        return '\n'.join([str(label) for label in self])
