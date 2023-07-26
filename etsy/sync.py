"""Implements the ETSY algorithm."""
import numpy as np
import pandas as pd
from tqdm import tqdm
from etsy import config, scoring, schema


class EventTrackingSynchronizer:
    """Synchronize event and tracking data using the ETSY algorithm.

    Parameters
    ----------
    events : pd.DataFrame
        Event data to synchronize, according to schema etsy.schema.event_schema.
    tracking : pd.DataFrame
        Tracking data to synchronize, according to schema etsy.schema.tracking_schema.
    fps : int
        Recording frequency (frames per second) of the tracking data.
    kickoff_time : int
        Length of the window (in seconds) at the start of a playing period in which to search for the kickoff frame.
    """

    def __init__(
        self,
        events: pd.DataFrame,
        tracking: pd.DataFrame,
        fps: int = 10,
        kickoff_time: int = 5,
    ):
        schema.event_schema.validate(events)
        schema.tracking_schema.validate(tracking)

        # Ensure unique index
        assert list(events.index.unique()) == [i for i in range(len(events))]
        assert list(tracking.index.unique()) == [i for i in range(len(tracking))]

        # Ensure frame identifiers are increasing by 1
        assert list(tracking.frame.unique()) == [
            i for i in range(max(tracking[tracking.period_id == 2].frame) + 1)
        ]

        self.events = events
        self.tracking = tracking
        self.fps = fps
        self.kickoff_time = kickoff_time
        self.last_matched_ts = pd.Timestamp("2000-01-01 01:00:00")

        # Store synchronization results
        self.shifted_timestamp = pd.Series(pd.NaT, index=[i for i in range(len(self.tracking))])
        self.matched_frames = pd.Series(np.nan, index=[i for i in range(len(self.events))])
        self.scores = pd.Series(np.nan, index=[i for i in range(len(self.events))])

    def find_kickoff(self, period: int):
        """Searches for the kickoff frame in a given playing period.

        Parameters
        ----------
        period: int
            The given playing period.

        Returns
        -------
            The found kickoff frame.
        """
        kickoff_event = self.events[self.events.period_id == period].iloc[0]

        if kickoff_event.type_name != "pass":
            raise Exception("First event is not a pass!")

        # Frames to search
        frame = self.tracking[self.tracking.period_id == period].frame.iloc[0]
        frames_to_check = [j for j in range(frame, frame + self.fps * self.kickoff_time, 1)]

        df_selection_player = self.tracking[
            (
                (self.tracking.frame.isin(frames_to_check))
                & (self.tracking.period_id == period)
                & (self.tracking.player_id == kickoff_event.player_id)
            )
        ]
        df_selection_ball = self.tracking[
            (
                (self.tracking.frame.isin(frames_to_check))
                & (self.tracking.period_id == period)
                & self.tracking.ball
            )
        ]

        # Mask of frames that have both player and ball in it
        mask_player = np.isin(
            df_selection_player.frame.to_numpy(),
            sorted(set(df_selection_player.frame.values) & set(df_selection_ball.frame.values)),
        )
        mask_ball = np.isin(
            df_selection_ball.frame.to_numpy(),
            sorted(set(df_selection_player.frame.values) & set(df_selection_ball.frame.values)),
        )

        dists = np.ones(len(df_selection_player)) * np.inf
        dists[mask_player] = np.sqrt(
            (df_selection_player[mask_player].x.values - df_selection_ball[mask_ball].x.values)
            ** 2
            + (df_selection_player[mask_player].y.values - df_selection_ball[mask_ball].y.values)
            ** 2
        )

        dist_idx = np.where(dists <= 2.0)
        if len(dist_idx[0]) == 0:
            best_idx = np.argmin(dists)
        else:
            max_accel = np.nanmax(df_selection_player.acceleration.values[dist_idx])
            best_idx = np.where(df_selection_player.acceleration == max_accel)[0][0]

        return df_selection_player.iloc[best_idx]

    def _find_matching_frame(
        self,
        event_idx: int,
        player_window: pd.DataFrame,
        ball_window: pd.DataFrame,
        mask_func,
    ):
        """Finds the matching frame of the given event within the given window.

        Parameters
        ----------
        event_idx: int
            The index of the event to be matched.
        player_window: pd.DataFrame
            All frames of the acting player within a certain window.
        ball_window: pd.DataFrame
            All frames of the ball within the same window.
        mask_func:
            One of the action-specific filters, depending on the event's type.

        Returns
        -------
            int
                Index of the matching frame in the tracking dataframe.
            float
                Score associated with the matching frame.
        """

        event = self.events.loc[event_idx]

        # Mask of frames that have both player and ball in it
        mask_player = np.isin(
            player_window.frame.to_numpy(),
            sorted(set(player_window.frame.values) & set(ball_window.frame.values)),
        )
        mask_ball = np.isin(
            ball_window.frame.to_numpy(),
            sorted(set(player_window.frame.values) & set(ball_window.frame.values)),
        )

        # Retrieve features
        acceleration = np.ones(len(player_window)) * np.nan
        acceleration[mask_player] = ball_window[mask_ball].acceleration.values

        height_ball = np.ones(len(player_window)) * np.inf
        height_ball[mask_player] = ball_window[mask_ball].z.values

        dist_event_player = np.ones(len(player_window)) * np.inf
        dist_event_player[mask_player] = np.sqrt(
            (player_window[mask_player].x.values - event.start_x) ** 2
            + (player_window[mask_player].y.values - event.start_y) ** 2
        )

        dist_event_ball = np.ones(len(player_window)) * np.inf
        dist_event_ball[mask_player] = np.sqrt(
            (ball_window[mask_ball].x.values - event.start_x) ** 2
            + (ball_window[mask_ball].y.values - event.start_y) ** 2
        )

        dists = np.ones(len(player_window)) * np.inf
        dists[mask_player] = np.sqrt(
            (player_window[mask_player].x.values - ball_window[mask_ball].x.values) ** 2
            + (player_window[mask_player].y.values - ball_window[mask_ball].y.values) ** 2
        )

        # Score frames
        scores = scoring.score_frames(
            mask_func,
            dists,
            height_ball,
            dist_event_player,
            dist_event_ball,
            acceleration,
            self.shifted_timestamp.loc[player_window.index.values],
            event.bodypart_id,
        )
        id_max = np.argmax(scores)

        return player_window.index[id_max], scores[id_max]

    def _window_of_frames(
        self,
        player_id,
        event_timestamp: np.dtype("datetime64[ns]"),
        period: int,
        s: int,
    ):
        """Identifies the qualifying window of frames around the event's timestamp.

        Parameters
        ----------
        player_id: Any
            The player executing the action.
        event_timestamp: np.datetime64[ns]
            The event's timestamp.
        period: int
            The playing period in which the event was performed.
        s: int
            Window length (in seconds).

        Returns
        -------
            pd.DataFrame
                All frames of the acting player in the given window.
            pd.DataFrame
                All frames containing the ball in the given window.
        """

        # Find closest frame time-wise
        frame = self.tracking.loc[
            [(abs(self.shifted_timestamp - event_timestamp)).idxmin()]
        ].frame.values[0]

        # Get all frames to search through
        all_frames = [j for j in range(frame, frame + self.fps * s, 1)]
        all_frames.extend([j for j in range(frame - 1, frame - self.fps * s, -1)])

        # Select all player and ball frames within window range
        player_frames = self.tracking[
            (
                (self.tracking.player_id == player_id)
                & (self.tracking.frame.isin(all_frames))
                & (self.tracking.period_id == period)
            )
        ]
        ball_frames = self.tracking[
            (
                self.tracking.ball
                & (self.tracking.frame.isin(all_frames))
                & (self.tracking.period_id == period)
            )
        ]

        return player_frames, ball_frames

    def _adjust_time_bias_tracking(
        self, kickoff_timestamp: np.dtype("datetime64[ns]"), period: int
    ):
        """Corrects the tracking data timestamps by removing the constant bias.

        Parameters
        ----------
        kickoff_timestamp: np.datetime64[ns]
            Timestamp of the kickoff in the tracking data.
        period: int
            The playing period in which the given kickoff is situated.
        """

        start_time_events = self.events[self.events.period_id == period].iloc[0].timestamp

        kickoff_diff = abs(start_time_events - kickoff_timestamp)

        if start_time_events > kickoff_timestamp:
            self.shifted_timestamp.loc[
                self.tracking[self.tracking.period_id == period]
                .index[0] : self.tracking[self.tracking.period_id == period]
                .index[-1]
            ] = (self.tracking[self.tracking.period_id == period].timestamp + kickoff_diff)

        else:
            self.shifted_timestamp.loc[
                self.tracking[self.tracking.period_id == period]
                .index[0] : self.tracking[self.tracking.period_id == period]
                .index[-1]
            ] = (self.tracking[self.tracking.period_id == period].timestamp - kickoff_diff)

    def _mask_incoming_like(
        self,
        dist_to_ball,
        height_ball,
        acceleration,
        timestamps,
        bodypart,
    ):
        mask_dist_ball = dist_to_ball <= 2
        mask_height_ball = height_ball <= 3
        mask_timestamps = timestamps > self.last_matched_ts
        mask_acceleration = acceleration <= 0

        return np.where(mask_dist_ball & mask_height_ball & mask_timestamps & mask_acceleration)

    def _mask_fault_like(
        self,
        dist_to_ball,
        height_ball,
        acceleration,
        timestamps,
        bodypart,
    ):
        mask_dist_ball = dist_to_ball <= 3
        mask_height_ball = height_ball <= 4
        mask_timestamps = timestamps > self.last_matched_ts

        return np.where(mask_dist_ball & mask_height_ball & mask_timestamps)

    def _mask_bad_touch(
        self,
        dist_to_ball,
        height_ball,
        acceleration,
        timestamps,
        bodypart,
    ):
        mask_dist_ball = dist_to_ball <= 3
        mask_height_ball = height_ball <= 3
        mask_timestamps = timestamps > self.last_matched_ts

        if bodypart == 0:
            mask_height = height_ball <= 1.5
        elif bodypart == 1:
            mask_height = height_ball > 1.0
        else:
            mask_height = np.ones(len(dist_to_ball))

        return np.where(mask_dist_ball & mask_height_ball & mask_timestamps & mask_height)

    def _mask_pass_like(
        self,
        dist_to_ball,
        height_ball,
        acceleration,
        timestamps,
        bodypart,
    ):
        mask_dist_ball = dist_to_ball <= 2.5
        mask_height_ball = height_ball <= 3
        mask_timestamps = timestamps > self.last_matched_ts

        if bodypart == 0:
            mask_height = height_ball <= 1.5
            mask_acceleration = acceleration >= 0
        elif bodypart == 1:
            mask_height = height_ball > 1.0
            mask_acceleration = np.ones(len(dist_to_ball))
        else:
            mask_height = np.ones(len(dist_to_ball))
            mask_acceleration = acceleration >= 0

        return np.where(
            mask_dist_ball & mask_height_ball & mask_timestamps & mask_acceleration & mask_height
        )

    def _sync_events_of_period(self, period: int):
        """Synchronizes the event and tracking data of a given playing period.

        Parameters
        ----------
        period: int
            The playing period of which to synchronize the event and tracking data.

        Returns
        -------
            np.array(float)
                Array containing all matched frame identifiers for the events, or NaN if no match could be found.
            np.array(float)
                Array containing the score for each match, or NaN is no match could be found.
        """

        matched_frames = np.ones(len(self.events[self.events.period_id == period]) - 1) * np.nan
        scores = np.ones(len(self.events[self.events.period_id == period]) - 1) * np.nan

        idx_start = self.events[self.events.period_id == period].index[0] + 1

        if period == 1:
            idxs = self.events.index[
                (
                    (self.events.index >= idx_start)
                    & (self.events.index < self.events[self.events.period_id == 2].index[0])
                )
            ]
        else:
            idxs = self.events.index[self.events.index >= idx_start]

        for k, idx in enumerate(tqdm(idxs)):
            player_id = self.events.loc[idx].player_id
            event_time = self.events.loc[idx].timestamp
            type_action = self.events.loc[idx].type_name

            if type_action in config.PASS_LIKE_OPEN:
                s = config.TIME_PASS_LIKE_OPEN
                score_fn = self._mask_pass_like
            elif type_action in config.SET_PIECE:
                s = config.TIME_SET_PIECE
                score_fn = self._mask_pass_like
            elif type_action in config.INCOMING_LIKE:
                s = config.TIME_INCOMING_LIKE
                score_fn = self._mask_incoming_like
            elif type_action in config.BAD_TOUCH:
                s = config.TIME_BAD_TOUCH
                score_fn = self._mask_bad_touch
            elif type_action in config.FAULT_LIKE:
                s = config.TIME_FAULT_LIKE
                score_fn = self._mask_fault_like
            elif type_action in config.NOT_HANDLED:
                continue
            else:
                raise Exception(f"Event type {type_action} unknown!")

            player_window, ball_window = self._window_of_frames(player_id, event_time, period, s)

            if len(player_window) > 0:
                frame_idx, score = self._find_matching_frame(
                    idx, player_window, ball_window, score_fn
                )

                if score > 0.0:
                    matched_frames[k] = self.tracking.loc[frame_idx].frame
                    scores[k] = score
                    self.last_matched_ts = self.shifted_timestamp.loc[frame_idx]
            else:
                print(f"No window found at {event_time}!")

        return matched_frames, scores

    def synchronize(self):
        """
        Applies the ETSY synchronization algorithm on the instantiated class.
        """

        # Find kickoff & adjust time bias between events and tracking
        kickoff_frame_p1 = self.find_kickoff(period=1)

        self._adjust_time_bias_tracking(kickoff_frame_p1.timestamp, 1)
        self.last_matched_ts = self.shifted_timestamp.loc[
            self.tracking[self.tracking.timestamp == kickoff_frame_p1.timestamp].index.values[0]
        ]

        # Sync events of playing period 1
        matched_frames_p1, scores_p1 = self._sync_events_of_period(1)

        # Find kickoff & adjust time bias between events and tracking
        kickoff_frame_p2 = self.find_kickoff(period=2)
        self._adjust_time_bias_tracking(kickoff_frame_p2.timestamp, 2)
        self.last_matched_ts = self.shifted_timestamp.loc[
            self.tracking[self.tracking.timestamp == kickoff_frame_p2.timestamp].index.values[0]
        ]

        # Sync events of playing period 2
        matched_frames_p2, scores_p2 = self._sync_events_of_period(2)

        # Store result
        self.matched_frames.loc[self.events.index[0]] = kickoff_frame_p1.frame
        self.matched_frames.loc[
            self.events.index[1 : len(self.events[self.events.period_id == 1])]
        ] = matched_frames_p1
        self.matched_frames.loc[
            self.events.index[len(self.events[self.events.period_id == 1])]
        ] = kickoff_frame_p2.frame
        self.matched_frames.loc[
            self.events.index[len(self.events[self.events.period_id == 1]) + 1 :]
        ] = matched_frames_p2

        self.scores.loc[
            self.events.index[1 : len(self.events[self.events.period_id == 1])]
        ] = scores_p1
        self.scores.loc[
            self.events.index[len(self.events[self.events.period_id == 1]) + 1 :]
        ] = scores_p2
