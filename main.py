from youtube.transcript import (
    extract_video_id,
    fetch_transcript
)

from processing.cleaner import process_segments

from processing.llm_segmenter import (
    clean_and_segment,
    attach_timestamps
)

from database.neon import insert_corpus_records

from models.corpus import generate_pair_id


DIALECTS = {
    "1": "Varhadi",
    "2": "Ahirani",
    "3": "Kolhapuri",
    "4": "Marathwadi"
}


def get_dialect():

    print("\nSelect dialect:")

    for number, dialect in DIALECTS.items():

        print(
            f"{number}. {dialect}"
        )

    while True:

        choice = input(
            "\nEnter option (1-4): "
        ).strip()

        if choice in DIALECTS:

            return DIALECTS[choice]

        print(
            "❌ Invalid choice. "
            "Please enter 1, 2, 3 or 4."
        )


def run_pipeline(
    youtube_url,
    dialect_label,
    domain_context=None
):

    print("\nStarting transcript pipeline...\n")

    # --------------------------------
    # 1. Extract video ID
    # --------------------------------

    video_id = extract_video_id(youtube_url)

    print(f"Video ID: {video_id}")

    # --------------------------------
    # 2. Fetch transcript
    # --------------------------------

    print("Fetching transcript...")

    transcript = fetch_transcript(video_id)

    print(
        f"Transcript segments: {len(transcript)}"
    )

    # --------------------------------
    # 3. Prepare raw segments
    # --------------------------------

    processed_segments = process_segments(
        transcript
    )

    print(
        f"Prepared segments: "
        f"{len(processed_segments)}"
    )

    # --------------------------------
    # 4. LLM segmentation
    # --------------------------------

    print("\nSending transcript to LLM...")
    print("Cleaning and splitting sentences...\n")

    llm_sentences = clean_and_segment(
        processed_segments
    )

    print(
        f"LLM sentences generated: "
        f"{len(llm_sentences)}"
    )

    # --------------------------------
    # 5. Attach timestamps
    # --------------------------------

    sentences = attach_timestamps(
        llm_sentences,
        processed_segments
    )

    print(
        f"Timestamped sentences: "
        f"{len(sentences)}"
    )

    # --------------------------------
    # 6. Preview
    # --------------------------------

    print("\n--------------------------------")
    print("PREVIEW")
    print("--------------------------------\n")

    for index, sentence in enumerate(
        sentences[:20],
        start=1
    ):

        print(
            f"{index}. {sentence['text']}"
        )

        print(
            f"   {sentence['start']:.2f}s"
            f" → "
            f"{sentence['end']:.2f}s"
        )

    if len(sentences) > 20:

        print(
            f"\n... and "
            f"{len(sentences) - 20} more sentences."
        )

    # --------------------------------
    # 7. Prepare DB records
    # --------------------------------

    records = []

    for sentence in sentences:

        record = {

            "pair_id": generate_pair_id(),

            "dialect_label": dialect_label,

            "source_text": sentence["text"],

            "target_text": None,

            "generation_method":
                "youtube_llm_segmentation",

            "domain_context":
                domain_context,

            "dataset_split": "raw",

            "source_type":
                "youtube_video",

            "source_url":
                youtube_url,

            "video_id":
                video_id,

            "timestamp_start":
                sentence["start"],

            "timestamp_end":
                sentence["end"],

            "transcript_source":
                "youtube_auto_generated",

            "verification_status":
                "llm_cleaned"
        }

        records.append(record)

    # --------------------------------
    # 8. Save to Neon
    # --------------------------------

    print("\nSaving records to Neon...")

    insert_corpus_records(records)

    print(
        f"\n✓ {len(records)} sentences saved to Neon."
    )

    print(
        "\nPipeline completed successfully!"
    )

    return records


if __name__ == "__main__":

    url = input(
        "\nEnter YouTube URL: "
    ).strip()

    dialect = get_dialect()

    print(
        f"\nSelected dialect: {dialect}"
    )

    run_pipeline(
        youtube_url=url,
        dialect_label=dialect
    )


