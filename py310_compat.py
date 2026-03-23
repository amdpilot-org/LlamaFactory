"""Python 3.10 compatibility shim for LLaMA-Factory (requires 3.11+).

Patches typing.Self, typing.NotRequired, and enum.StrEnum into stdlib
so LLaMA-Factory code works on Python 3.10 with typing_extensions.
"""

import sys

if sys.version_info < (3, 11):
    import typing
    import enum
    from typing_extensions import Self, NotRequired

    if not hasattr(typing, "Self"):
        typing.Self = Self
    if not hasattr(typing, "NotRequired"):
        typing.NotRequired = NotRequired
    if not hasattr(enum, "StrEnum"):
        class StrEnum(str, enum.Enum):
            pass
        enum.StrEnum = StrEnum
