from RTWhisper import Pipeline


class Classifier(Pipeline):
    def _get_prev_timestamps(self, timestamps: list[dict], anchor: int):
        prev_timestamps = []
        t_index = 0
        while t_index < len(timestamps) and timestamps[t_index]["end"] < anchor:
            t_index += 1
        if t_index < len(timestamps) and timestamps[t_index]["start"] < anchor:
            t = timestamps[t_index]
            prev_timestamps.append({"start": 0, "end": t["end"] - anchor})
            t_index += 1
        while t_index < len(timestamps):
            t = timestamps[t_index]
            prev_timestamps.append(
                {"start": t["start"] - anchor, "end": t["end"] - anchor}
            )
            t_index += 1
        return prev_timestamps

    def _get_prev_timestamps_mapping(self, timestamps_mapping: list[dict], anchor: int):
        prev_timestamps_mapping = []
        t_index = 0
        while (
            t_index < len(timestamps_mapping)
            and timestamps_mapping[t_index]["end"] < anchor
        ):
            t_index += 1
        offset = (
            timestamps_mapping[t_index]["offset"]
            if t_index < len(timestamps_mapping)
            else 0
        )
        if (
            t_index < len(timestamps_mapping)
            and timestamps_mapping[t_index]["start"] < anchor
        ):
            t = timestamps_mapping[t_index]
            prev_timestamps_mapping.append(
                {"start": 0, "end": t["end"] - anchor, "offset": 0}
            )
            t_index += 1
        while t_index < len(timestamps_mapping):
            t = timestamps_mapping[t_index]
            prev_timestamps_mapping.append(
                {
                    "start": t["start"] - anchor,
                    "end": t["end"] - anchor,
                    "offset": t["offset"] - offset,
                }
            )
            t_index += 1
        return prev_timestamps_mapping
