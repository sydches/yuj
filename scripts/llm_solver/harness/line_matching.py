"""Exact whole-line matching through Python's native substring search."""


def matching_line_starts(lines: list[str], needle: list[str], start_idx: int = 0):
    """Yield every matching window, including overlaps, in source order.

    Inputs are lines already split by the patch parsers. Newline sentinels
    prevent partial-line matches; substring search avoids copying a window
    at every source line. Count each intervening newline only once.
    """
    if not needle:
        yield from range(start_idx, len(lines) + 1)
        return
    if len(needle) > len(lines) - start_idx:
        return
    text = '\n' + '\n'.join(lines[start_idx:]) + '\n'
    block = '\n' + '\n'.join(needle) + '\n'
    position, line = 0, start_idx
    while True:
        match = text.find(block, position)
        if match < 0:
            return
        line += text.count('\n', position, match)
        yield line
        position = match + 1
        line += 1
