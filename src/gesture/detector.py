"""Rectangle gesture detector. Determines which show is active based on
which finger pair forms the rectangle on both hands."""
from __future__ import annotations

from enum import Enum

import numpy as np

THUMB_TIP, INDEX_TIP, MIDDLE_TIP, RING_TIP, PINKY_TIP = 4, 8, 12, 16, 20
WRIST = 0
THUMB_IP = 2
# MCP joints (knuckles) for extension check
INDEX_MCP, MIDDLE_MCP, RING_MCP, PINKY_MCP = 5, 9, 13, 17
# PIP joints (middle knuckle) — the reliable pivot for the extension test.
INDEX_PIP, MIDDLE_PIP, RING_PIP, PINKY_PIP = 6, 10, 14, 18


class Show(Enum):
    NONE = "none"
    NARUTO = "naruto"
    ATLA = "atla"
    JJK = "jjk"


# Which finger pair forms the rectangle corners for each show.
SHOW_FINGER_PAIRS = {
    Show.NARUTO: (THUMB_TIP, INDEX_TIP),
    Show.ATLA:   (INDEX_TIP, MIDDLE_TIP),
    Show.JJK:    (MIDDLE_TIP, PINKY_TIP),
}

# Per show: the fingers that must be extended ("up") and the disambiguating
# fingers that must be curled ("down"). The ring finger is intentionally left
# out of both sets — it is hard to control independently — which keeps the three
# finger-pairs cleanly separable.
SHOW_RULES = {
    Show.NARUTO: (frozenset({THUMB_TIP, INDEX_TIP}),  frozenset({MIDDLE_TIP, PINKY_TIP})),
    Show.ATLA:   (frozenset({INDEX_TIP, MIDDLE_TIP}), frozenset({THUMB_TIP, PINKY_TIP})),
    Show.JJK:    (frozenset({MIDDLE_TIP, PINKY_TIP}), frozenset({THUMB_TIP, INDEX_TIP})),
}

FINGER_LABELS = {
    THUMB_TIP: "T", INDEX_TIP: "I", MIDDLE_TIP: "M", RING_TIP: "R", PINKY_TIP: "P",
}


def _finger_extended(hand: np.ndarray, tip_idx: int, pip_idx: int,
                     mcp_idx: int) -> bool:
    """A finger is extended when its tip reaches beyond its own PIP joint,
    measured from the wrist. Comparing the tip against the finger's PIP (rather
    than its MCP) works for every finger regardless of length — including the
    short pinky, whose tip barely clears its MCP even when fully extended."""
    wrist = hand[WRIST, :2]
    tip_dist = np.linalg.norm(hand[tip_idx, :2] - wrist)
    pip_dist = np.linalg.norm(hand[pip_idx, :2] - wrist)
    mcp_dist = np.linalg.norm(hand[mcp_idx, :2] - wrist)
    # Tip past the PIP, and the PIP itself past the MCP (finger not folded flat).
    return tip_dist > pip_dist * 1.05 and pip_dist > mcp_dist * 0.9


def _extended_fingers(hand: np.ndarray) -> set[int]:
    """Return the set of extended fingertip indices."""
    extended = set()
    # Palm width (index MCP to pinky MCP) keeps the checks scale-invariant.
    palm = np.linalg.norm(hand[INDEX_MCP, :2] - hand[PINKY_MCP, :2]) + 1e-6
    # Thumb: extended only when the tip sits well away from the index knuckle.
    # A tucked thumb collapses toward the palm and must read as "down", otherwise
    # every index-up pose looks like NARUTO (thumb+index).
    if np.linalg.norm(hand[THUMB_TIP, :2] - hand[INDEX_MCP, :2]) > palm * 0.7:
        extended.add(THUMB_TIP)
    for tip, pip, mcp in [(INDEX_TIP, INDEX_PIP, INDEX_MCP),
                          (MIDDLE_TIP, MIDDLE_PIP, MIDDLE_MCP),
                          (RING_TIP, RING_PIP, RING_MCP),
                          (PINKY_TIP, PINKY_PIP, PINKY_MCP)]:
        if _finger_extended(hand, tip, pip, mcp):
            extended.add(tip)
    return extended


def extended_labels(hand: np.ndarray | None) -> str:
    """Compact debug string of extended fingers, e.g. "IM" for index+middle."""
    if hand is None:
        return "-"
    ext = _extended_fingers(hand)
    return "".join(lbl for tip, lbl in FINGER_LABELS.items() if tip in ext) or "-"


def detect_show(left_hand: np.ndarray | None,
                right_hand: np.ndarray | None) -> Show:
    """
    Determine the active show. A show requires its two "up" fingers extended on
    BOTH hands and its disambiguating "down" fingers curled on both hands, which
    makes the overlapping finger-pairs unambiguous.
    """
    if left_hand is None or right_hand is None:
        return Show.NONE

    left_ext = _extended_fingers(left_hand)
    right_ext = _extended_fingers(right_hand)

    for show, (up, down) in SHOW_RULES.items():
        if (up <= left_ext and up <= right_ext
                and not (down & left_ext) and not (down & right_ext)):
            return show
    return Show.NONE


def rectangle_corners(left_hand: np.ndarray, right_hand: np.ndarray,
                      show: Show) -> np.ndarray | None:
    """Return the 4 corner points of the rectangle for compositing/masking."""
    if show == Show.NONE:
        return None
    a, b = SHOW_FINGER_PAIRS[show]
    return np.array([
        left_hand[a, :2],
        left_hand[b, :2],
        right_hand[b, :2],
        right_hand[a, :2],
    ], dtype=np.float32)
