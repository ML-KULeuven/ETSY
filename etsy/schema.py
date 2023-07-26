"""Schemas for the event and tracking data."""
from pandera import Column, DataFrameSchema, Check, Index
import numpy as np
from etsy import config

event_schema = DataFrameSchema(
    {
        "period_id": Column(int, Check(lambda s: s.isin([1, 2]))),
        "timestamp": Column(np.dtype("datetime64[ns]")),
        "player_id": Column(object),
        "type_name": Column(str, Check(lambda s: s.isin(config.SPADL_TYPES))),
        "start_x": Column(float, Check(lambda s: (s >= 0) & (s <= config.FIELD_LENGTH))),
        "start_y": Column(float, Check(lambda s: (s >= 0) & (s <= config.FIELD_WIDTH))),
        "bodypart_id": Column(int, Check(lambda s: s.isin(range(len(config.SPADL_BODYPARTS))))),
    },
    index=Index(int),
)

tracking_schema = DataFrameSchema(
    {
        "period_id": Column(int, Check(lambda s: s.isin([1, 2]))),
        "timestamp": Column(np.dtype("datetime64[ns]")),
        "frame": Column(int, Check(lambda s: s >= 0)),
        "player_id": Column(object, nullable=True),  # Mandatory for players (not ball)
        "ball": Column(bool),
        "x": Column(float),
        "y": Column(float),
        "z": Column(
            float, Check(lambda s: s >= 0), nullable=True
        ),  # Mandatory for ball (not players)
        "acceleration": Column(float),
    },
    index=Index(int),
)
