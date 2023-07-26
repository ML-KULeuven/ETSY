"""Defines ETSY configuration."""

FIELD_LENGTH = 105.0  # unit: meters
FIELD_WIDTH = 68.0  # unit: meters

SPADL_TYPES = [
    "pass",
    "cross",
    "throw_in",
    "freekick_crossed",
    "freekick_short",
    "corner_crossed",
    "corner_short",
    "take_on",
    "foul",
    "tackle",
    "interception",
    "shot",
    "shot_penalty",
    "shot_freekick",
    "keeper_save",
    "keeper_claim",
    "keeper_punch",
    "keeper_pick_up",
    "clearance",
    "bad_touch",
    "non_action",
    "dribble",
    "goalkick",
]

SPADL_BODYPARTS = ["foot", "head", "other", "head/other", "foot_left", "foot_right"]

PASS_LIKE_OPEN = ["pass", "cross", "shot", "clearance", "keeper_punch", "take_on"]

SET_PIECE = [
    "throw_in",
    "freekick_crossed",
    "freekick_short",
    "corner_crossed",
    "corner_short",
    "shot_penalty",
    "shot_freekick",
    "goalkick",
]

FAULT_LIKE = ["foul", "tackle"]

BAD_TOUCH = ["bad_touch"]

INCOMING_LIKE = ["interception", "keeper_save", "keeper_claim", "keeper_pick_up"]

NOT_HANDLED = ["non_action", "dribble"]

TIME_PASS_LIKE_OPEN = 5  # unit: seconds
TIME_SET_PIECE = 10  # unit: seconds
TIME_FAULT_LIKE = 5  # unit: seconds
TIME_BAD_TOUCH = 5  # unit: seconds
TIME_INCOMING_LIKE = 5  # unit: seconds
