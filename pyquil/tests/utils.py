from pyquil.parser import parse


def parse_equals(quil_string, *instructions):
    assert list(instructions) == parse(quil_string)
