from functools import wraps


def set_property(decorated):
    """
    Sets the keyword arguments of this method as class variables.
    """
    @wraps(decorated)
    def assigner(self, *args, **kwargs):
        self.__dict__.update(kwargs)
        decorated(self, *args, **kwargs)
    return assigner


def set_property_hidden(decorated):
    """
    Sets the keyword arguments of this method as hidden class variables.
    """
    @wraps(decorated)
    def assigner(self, *args, **kwargs):
        self.__dict__.update({f"_{k}": v for k, v in kwargs.items()})
        decorated(self, *args, **kwargs)
    return assigner


def set_properties(self, **properties):
    properties.pop("self", None)
    self.__dict__.update(properties)


def set_properties_hidden(self, **properties):
    properties.pop("self", None)
    d = {}
    for key, val in properties.items():
        d[f"_{key}"] = val
    self.__dict__.update(d)
