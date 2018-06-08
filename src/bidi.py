import json
from collections import deque
from unicodedata import bidirectional

from textutils import uxxxx_to_utf8


# Description of issue
#  We might have a sring that is predominantly right-to-left, but contains some left-to-right spans in it
#  e.g.  abc 123 xyz
#  where let's suppose a, b, c, x, y, and z are all right-to-left Arabic characters, but 123 is a left-to-right digit sequence
#  Then what is stored in memory is this (think of it as "writing order"):
#     zyx 123 cba
#  and the bidi algorithm will display the correct display order shown above when displaying text to a user
#
#  Our problem is what we need for OCR is literal order of appearance of the letters, in particular we need to 
#  reverse the string so that it maps up to the display image; but in doing so we reverse the left-to-right digit string
#  which *shouldn't* be reversed.
#
#  This can get complicated with certain punctuation marks sometimes needing to be reversed and sometimes not.
#  Here we tryto 'undo' the Unicode bidi algorithm to get what we want

def _reverse_contiguous_sequence(chars, line_start, line_end, highest_level,
                                 lowest_odd_level):
    """L2. From the highest level found in the text to the lowest odd
    level on each line, including intermediate levels not actually
    present in the text, reverse any contiguous sequence of characters
    that are at that level or higher.

    """
    for level in range(highest_level, lowest_odd_level - 1, -1):
        _start = _end = None

        for run_idx in range(line_start, line_end + 1):
            run_ch = chars[run_idx]

            if run_ch['level'] >= level:
                if _start is None:
                    _start = _end = run_idx
                else:
                    _end = run_idx
            else:
                if _end:
                    chars[_start:+_end + 1] = \
                        reversed(chars[_start:+_end + 1])
                    _start = _end = None

        # anything remaining ?
        if _start is not None:
            chars[_start:+_end + 1] = \
                reversed(chars[_start:+_end + 1])


PARAGRAPH_LEVELS = {'L': 0, 'AL': 1, 'R': 1}
EXPLICIT_LEVEL_LIMIT = 62


def _LEAST_GREATER_ODD(x):
    return (x + 1) | 1


def _LEAST_GREATER_EVEN(x):
    return (x + 2) & ~1


X2_X5_MAPPINGS = {
    'RLE': (_LEAST_GREATER_ODD, 'N'),
    'LRE': (_LEAST_GREATER_EVEN, 'N'),
    'RLO': (_LEAST_GREATER_ODD, 'R'),
    'LRO': (_LEAST_GREATER_EVEN, 'L'),
}

# Added 'B' so X6 won't execute in that case and X8 will run it's course
X6_IGNORED = list(X2_X5_MAPPINGS.keys()) + ['BN', 'PDF', 'B']
X9_REMOVED = list(X2_X5_MAPPINGS.keys()) + ['BN', 'PDF']


def undo_bidi(uxxxx_str, base_level=1):
    # Step 0: attach the unicode bidi type to each char
    augmented_char_array = []
    for char in uxxxx_str.split():
        bidi_type = bidirectional(uxxxx_to_utf8(char))
        # For now, hard-coding base level to always be RTL, because this is for Arabic corpus. Revisit this later!
        augmented_char_array.append(
            {'char': char, 'bidi-type': bidi_type, 'bidi-orig-type': bidi_type, 'level': base_level})

    # Step 1: Resolve Explicit embed and overrides
    #   See:  http://unicode.org/reports/tr9/#Explicit_Levels_and_Directions
    overflow_counter = almost_overflow_counter = 0
    directional_override = 'N'
    levels = deque()

    # X1
    embedding_level = base_level

    for _ch in augmented_char_array:
        bidi_type = _ch['bidi-type']

        level_func, override = X2_X5_MAPPINGS.get(bidi_type, (None, None))

        if level_func:
            # So this is X2 to X5
            # if we've past EXPLICIT_LEVEL_LIMIT, note it and do nothing

            if overflow_counter != 0:
                overflow_counter += 1
                continue

            new_level = level_func(embedding_level)
            if new_level < EXPLICIT_LEVEL_LIMIT:
                levels.append((embedding_level, directional_override))
                embedding_level, directional_override = new_level, override

            elif embedding_level == EXPLICIT_LEVEL_LIMIT - 2:
                # The new level is invalid, but a valid level can still be
                # achieved if this level is 60 and we encounter an RLE or
                # RLO further on.  So record that we 'almost' overflowed.
                almost_overflow_counter += 1

            else:
                overflow_counter += 1
        else:
            # X6
            if bidi_type not in X6_IGNORED:
                _ch['level'] = embedding_level
                if directional_override != 'N':
                    _ch['bidi-type'] = directional_override

            # X7
            elif bidi_type == 'PDF':
                if overflow_counter:
                    overflow_counter -= 1
                elif almost_overflow_counter and \
                                embedding_level != EXPLICIT_LEVEL_LIMIT - 1:
                    almost_overflow_counter -= 1
                elif levels:
                    embedding_level, directional_override = levels.pop()

            # X8
            elif bidi_type == 'B':
                levels.clear()
                overflow_counter = almost_overflow_counter = 0
                embedding_level = _ch['level'] = base_level
                directional_override = 'N'

    # Removes the explicit embeds and overrides of types
    # RLE, LRE, RLO, LRO, PDF, and BN. Adjusts extended chars
    # next and prev as well

    # Applies X9. See http://unicode.org/reports/tr9/#X9
    augmented_char_array = [_ch for _ch in augmented_char_array if _ch['bidi-type'] not in X9_REMOVED]

    # Step 2: determine LTR / RTL runs
    #  See: See http://unicode.org/reports/tr9/#X10

    # First define utility function: Basically, RTL takes preference over LTR ... if either left/right boundary is RTL then all is RTL
    def calc_level_run(b_l, b_r):
        return ['L', 'R'][max(b_l, b_r) % 2]

    runs = []
    first_char = augmented_char_array[0]
    run_start_level = calc_level_run(first_char['level'], base_level)
    run_end_level = None
    run_start = run_length = 0
    prev_level, prev_type = first_char['level'], first_char['bidi-type']

    for char in augmented_char_array:
        curr_level, curr_type = char['level'], char['bidi-type']

        if curr_level == prev_level:
            run_length += 1
        else:
            run_end_level = calc_level_run(prev_level, curr_level)
            runs.append({'sor': run_start_level, 'eor': run_end_level, 'start': run_start,
                         'type': prev_type, 'length': run_length})
            run_start_level = run_end_level
            run_start += run_length
            run_length = 1

        prev_level, prev_type = curr_level, curr_type

    # for the last char/runlevel
    run_end_level = calc_level_run(curr_level, base_level)
    runs.append({'sor': run_start_level, 'eor': run_end_level, 'start': run_start,
                 'type': curr_type, 'length': run_length})

    # Step 3: Resolve weak LTR/RTL types
    #   See: http://unicode.org/reports/tr9/#Resolving_Weak_Types
    for run in runs:
        prev_strong = prev_type = run['sor']
        start, length = run['start'], run['length']
        chars = augmented_char_array[start:start + length]

        for char in chars:
            # W1. Examine each nonspacing mark (NSM) in the level run, and
            # change the type of the NSM to the type of the previous character.
            # If the NSM is at the start of the level run, it will get the type
            # of sor.
            bidi_type = char['bidi-type']

            if bidi_type == 'NSM':
                char['bidi-type'] = bidi_type = prev_type

            # W2. Search backward from each instance of a European number until
            # the first strong type (R, L, AL, or sor) is found. If an AL is
            # found, change the type of the European number to Arabic number.
            if bidi_type == 'EN' and prev_strong == 'AL':
                char['bidi-type'] = 'AN'

            # update prev_strong if needed
            if bidi_type in ('R', 'L', 'AL'):
                prev_strong = bidi_type

            prev_type = char['bidi-type']

        # W3. Change all ALs to R
        for char in chars:
            if char['bidi-type'] == 'AL':
                char['bidi-type'] = 'R'

        # W4. A single European separator between two European numbers changes
        # to a European number. A single common separator between two numbers of
        # the same type changes to that type.
        for idx in range(1, len(chars) - 1):
            bidi_type = chars[idx]['bidi-type']
            prev_type = chars[idx - 1]['bidi-type']
            next_type = chars[idx + 1]['bidi-type']

            if bidi_type == 'ES' and (prev_type == next_type == 'EN'):
                chars[idx]['bidi-type'] = 'EN'

            if bidi_type == 'CS' and prev_type == next_type and \
                            prev_type in ('AN', 'EN'):
                chars[idx]['bidi-type'] = prev_type

        # W5. A sequence of European terminators adjacent to European numbers
        # changes to all European numbers.
        for idx in range(len(chars)):
            if chars[idx]['bidi-type'] == 'EN':
                for et_idx in range(idx - 1, -1, -1):
                    if chars[et_idx]['bidi-type'] == 'ET':
                        chars[et_idx]['bidi-type'] = 'EN'
                    else:
                        break
                for et_idx in range(idx + 1, len(chars)):
                    if chars[et_idx]['bidi-type'] == 'ET':
                        chars[et_idx]['bidi-type'] = 'EN'
                    else:
                        break

        # W6. Otherwise, separators and terminators change to Other Neutral.
        for char in chars:
            if char['bidi-type'] in ('ET', 'ES', 'CS'):
                char['bidi-type'] = 'ON'

        # W7. Search backward from each instance of a European number until the
        # first strong type (R, L, or sor) is found. If an L is found, then
        # change the type of the European number to L.
        prev_strong = run['sor']
        for char in chars:
            if char['bidi-type'] == 'EN' and prev_strong == 'L':
                char['bidi-type'] = 'L'

            if char['bidi-type'] in ('L', 'R'):
                prev_strong = char['bidi-type']

    # Step 4: Resolve Neutral Types
    #   See: http://unicode.org/reports/tr9/#Resolving_Neutral_Types
    for run in runs:
        start, length = run['start'], run['length']
        # use sor and eor
        chars = [{'bidi-type': run['sor']}] + augmented_char_array[start:start + length] + [{'bidi-type': run['eor']}]
        total_chars = len(chars)

        seq_start = None
        for idx in range(total_chars):
            _ch = chars[idx]
            if _ch['bidi-type'] in ('B', 'S', 'WS', 'ON'):
                # N1. A sequence of neutrals takes the direction of the
                # surrounding strong text if the text on both sides has the same
                # direction. European and Arabic numbers act as if they were R
                # in terms of their influence on neutrals. Start-of-level-run
                # (sor) and end-of-level-run (eor) are used at level run
                # boundaries.
                if seq_start is None:
                    seq_start = idx
                    prev_bidi_type = chars[idx - 1]['bidi-type']
            else:
                if seq_start is not None:
                    next_bidi_type = chars[idx]['bidi-type']

                    if prev_bidi_type in ('AN', 'EN'):
                        prev_bidi_type = 'R'

                    if next_bidi_type in ('AN', 'EN'):
                        next_bidi_type = 'R'

                    for seq_idx in range(seq_start, idx):
                        if prev_bidi_type == next_bidi_type:
                            chars[seq_idx]['bidi-type'] = prev_bidi_type
                        else:
                            # N2. Any remaining neutrals take the embedding
                            # direction. The embedding direction for the given
                            # neutral character is derived from its embedding
                            # level: L if the character is set to an even level,
                            # and R if the level is odd.
                            if chars[seq_idx]['level'] % 2 == 0:
                                chars[seq_idx]['bidi-type'] = 'L'
                            else:
                                chars[seq_idx]['bidi-type'] = 'R'

                    seq_start = None

    # Step 5: Resolve Implicit Levels
    #   See: http://unicode.org/reports/tr9/#Resolving_Implicit_Levels
    def _embedding_direction(x):
        return ('L', 'R')[x % 2]

    for run in runs:
        start, length = run['start'], run['length']
        chars = augmented_char_array[start:start + length]

        for _ch in chars:
            # only those types are allowed at this stage
            assert _ch['bidi-type'] in ('L', 'R', 'EN', 'AN'), \
                '%s not allowed here' % _ch['bidi-type']

            if _embedding_direction(_ch['level']) == 'L':
                # I1. For all characters with an even (left-to-right) embedding
                # direction, those of type R go up one level and those of type
                # AN or EN go up two levels.
                if _ch['bidi-type'] == 'R':
                    _ch['level'] += 1
                elif _ch['bidi-type'] != 'L':
                    _ch['level'] += 2
            else:
                # I2. For all characters with an odd (right-to-left) embedding
                # direction, those of type L, EN or AN  go up one level.
                if _ch['bidi-type'] != 'R':
                    _ch['level'] += 1

    # Step 6: Reorder Resolved Levels
    #   See: http://unicode.org/reports/tr9/#I2

    # Applies L1.

    should_reset = True
    chars = augmented_char_array

    for _ch in chars[::-1]:
        # L1. On each line, reset the embedding level of the following
        # characters to the paragraph embedding level:
        if _ch['bidi-orig-type'] in ('B', 'S'):
            # 1. Segment separators,
            # 2. Paragraph separators,
            _ch['level'] = base_level
            should_reset = True
        elif should_reset and _ch['bidi-orig-type'] in ('BN', 'WS'):
            # 3. Any sequence of whitespace characters preceding a segment
            # separator or paragraph separator
            # 4. Any sequence of white space characters at the end of the
            # line.
            _ch['level'] = base_level
        else:
            should_reset = False

    max_len = len(chars)

    # L2 should be per line
    # Calculates highest level and loweset odd level on the fly.
    line_start = line_end = 0
    highest_level = 0
    lowest_odd_level = EXPLICIT_LEVEL_LIMIT

    for idx in range(max_len):
        _ch = chars[idx]

        # calc the levels
        char_level = _ch['level']
        if char_level > highest_level:
            highest_level = char_level

        if char_level % 2 and char_level < lowest_odd_level:
            lowest_odd_level = char_level

        if _ch['bidi-orig-type'] == 'B' or idx == max_len - 1:
            line_end = idx
            # omit line breaks
            if _ch['bidi-orig-type'] == 'B':
                line_end -= 1

            _reverse_contiguous_sequence(chars, line_start, line_end,
                                         highest_level, lowest_odd_level)

            # reset for next line run
            line_start = idx + 1
            highest_level = 0
            lowest_odd_level = EXPLICIT_LEVEL_LIMIT

    # Finally, reverse entire string
    return ' '.join([char['char'] for char in reversed(chars)])
