"""Defines ETSY scoring functions."""
import numpy as np
from etsy.config import FIELD_LENGTH, FIELD_WIDTH


def down_lin_func(mini: float, maxi: float, minval: float, maxval: float):
    if maxi == mini:
        return lambda x: maxval

    a = (minval - maxval) / (maxi - mini)
    b = (maxi * maxval - mini * minval) / (maxi - mini)

    return lambda x: a * x + b


score_dist = down_lin_func(0.0, np.sqrt(FIELD_LENGTH**2 + FIELD_WIDTH**2), 0, 100 / 3)
score_dist_player = down_lin_func(0.0, np.sqrt(FIELD_LENGTH**2 + FIELD_WIDTH**2), 0, 100 / 3)
score_dist_ball = down_lin_func(0.0, np.sqrt(FIELD_LENGTH**2 + FIELD_WIDTH**2), 0, 100 / 3)


def score_frames(
    mask_func,
    dist_to_ball,
    height_ball,
    dist_event_player,
    dist_event_ball,
    acceleration,
    timestamps,
    bodypart,
):
    scores = np.zeros(len(dist_to_ball))

    idx = mask_func(
        dist_to_ball,
        height_ball,
        acceleration,
        timestamps,
        bodypart,
    )

    if len(idx[0]) > 0:
        scores[idx] += (
            score_dist(dist_to_ball[idx])
            + score_dist_player(dist_event_player[idx])
            + score_dist_ball(dist_event_ball[idx])
        )

    return scores
