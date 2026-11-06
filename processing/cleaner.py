def process_segments(transcript):
    """
    Convert YouTube transcript objects into simple
    timestamped segments.

    No sentence splitting happens here.
    The LLM will handle that.
    """

    processed = []

    for index, segment in enumerate(transcript):

        text = segment.text.strip()

        if not text:
            continue

        start = float(segment.start)
        end = start + float(segment.duration)

        processed.append({
            "index": index,
            "text": text,
            "start": start,
            "end": end
        })

    return processed




