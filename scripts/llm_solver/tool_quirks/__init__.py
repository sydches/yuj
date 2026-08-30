"""Non-bash tool-result quirks.

This is the L2 sibling of ``bash_quirks``. Bash quirks owns transforms over
bash output. Tool quirks owns supported transforms over other tool results.

The current data surface is ``glob.toml``, which owns the two model-facing
glob hints. Numeric limits remain in ``Config``. ``transforms.py`` owns the
cap decision, guarded envelope, and savings record. The glob handler owns
filesystem enumeration, normal pagination, and explicit transform wiring.

This package is not a general tool plugin loader. Adding another result
transform requires Python here and an explicit call from its tool handler.
"""
