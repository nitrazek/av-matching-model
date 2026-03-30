import argparse
from dataclasses import fields


def build_parser_from_dataclass(cls):
    parser = argparse.ArgumentParser()
    for f in fields(cls):
        parser.add_argument(f"--{f.name}", type=f.type, default=f.default)
    return parser