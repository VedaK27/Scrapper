# import os
# import json

# from openai import OpenAI
# from dotenv import load_dotenv


# load_dotenv()

# client = OpenAI(
#     api_key=os.getenv("OPENAI_API_KEY")
# )


# MODEL = os.getenv(
#     "OPENAI_MODEL",
#     "gpt-5.6-luna"
# )


# SYSTEM_PROMPT = """
# You are a Marathi transcript segmentation and cleaning system.

# Your task is to process Marathi conversational speech transcripts.

# The transcript may contain:
# - Marathi dialects such as Varhadi
# - missing punctuation
# - YouTube auto-transcription errors
# - repeated words
# - speaker markers such as >>
# - annotations such as [हास्य] or [संगीत]
# - multiple spoken sentences inside one transcript segment

# Your job is ONLY to clean and segment the transcript.

# VERY IMPORTANT:

# 1. Preserve the original dialect.

# 2. DO NOT translate dialect Marathi into Standard Marathi.

# 3. DO NOT replace dialect words with standard Marathi words.

# Example:
# "मले" must NOT become "मला".
# "तुले" must NOT become "तुला".

# 4. DO NOT paraphrase.

# 5. DO NOT summarize.

# 6. DO NOT add information that was not spoken.

# 7. You MAY:
#    - remove >> speaker markers
#    - remove [हास्य], [संगीत], [टाळ्या], etc.
#    - add punctuation
#    - split multiple sentences
#    - remove obvious non-speech transcript artifacts
#    - make very minor corrections when an obvious transcription artifact
#      prevents understanding, but preserve the dialect wording whenever possible.

# 8. Each output item should represent ONE natural spoken sentence or
#    complete conversational utterance.

# 9. Short utterances are valid.

# Examples:
# "काय दादा?"
# "हो."
# "बस आलोच मी."
# "चल."
# "कुठं जातोय?"

# 10. DO NOT combine several independent spoken sentences into one item.

# 11. DO NOT split a single natural sentence unnecessarily.

# 12. Preserve the wording and dialect as closely as possible.

# 13. Every output sentence MUST reference the original transcript
#     segment indexes from which it was created.

# Return ONLY valid JSON.
# """


# def clean_and_segment(segments):
#     """
#     Send timestamped transcript segments to the LLM
#     and receive natural sentence/utterance boundaries.
#     """

#     if not segments:
#         return []

#     transcript_text = []

#     for segment in segments:

#         transcript_text.append(
#             f"[{segment['index']}] {segment['text']}"
#         )

#     joined_transcript = "\n".join(transcript_text)

#     user_prompt = f"""
# Segment the following Marathi conversational transcript.

# Remember:

# - Preserve dialect exactly.
# - Do not translate to Standard Marathi.
# - Do not paraphrase.
# - Do not summarize.
# - Split into natural complete spoken sentences/utterances.
# - One sentence/utterance per output item.
# - Keep the original segment indexes.

# TRANSCRIPT:

# {joined_transcript}
# """

#     response = client.responses.create(
#         model=MODEL,
#         instructions=SYSTEM_PROMPT,
#         input=user_prompt
#     )

#     raw_output = response.output_text.strip()

#     try:
#         result = json.loads(raw_output)

#     except json.JSONDecodeError as error:

#         print("\n❌ LLM returned invalid JSON.")
#         print("\nRaw response:")
#         print(raw_output)

#         raise RuntimeError(
#             "LLM returned invalid JSON."
#         ) from error

#     return result


# def attach_timestamps(llm_sentences, original_segments):
#     """
#     Attach timestamps from original YouTube segments
#     to the sentences returned by the LLM.
#     """

#     segment_map = {
#         segment["index"]: segment
#         for segment in original_segments
#     }

#     final_sentences = []

#     for sentence in llm_sentences:

#         text = sentence.get("text", "").strip()

#         indexes = sentence.get("segments", [])

#         if not text or not indexes:
#             continue

#         valid_segments = [
#             segment_map[index]
#             for index in indexes
#             if index in segment_map
#         ]

#         if not valid_segments:
#             continue

#         start = min(
#             segment["start"]
#             for segment in valid_segments
#         )

#         end = max(
#             segment["end"]
#             for segment in valid_segments
#         )

#         final_sentences.append({
#             "text": text,
#             "start": start,
#             "end": end
#         })

#     return final_sentences

import os
from typing import List

from dotenv import load_dotenv
from google import genai
from pydantic import BaseModel

load_dotenv()


# ---------------------------------------------------------
# Gemini configuration
# ---------------------------------------------------------

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

if not GEMINI_API_KEY:
    raise ValueError(
        "GEMINI_API_KEY is not set in .env"
    )

GEMINI_MODEL = os.getenv(
    "GEMINI_MODEL",
    "gemini-3.7-flash"
)

client = genai.Client(
    api_key=GEMINI_API_KEY
)


# ---------------------------------------------------------
# Structured output models
# ---------------------------------------------------------

class Sentence(BaseModel):
    text: str
    segments: List[int]


class SegmentationResult(BaseModel):
    sentences: List[Sentence]


# ---------------------------------------------------------
# System instructions
# ---------------------------------------------------------

SYSTEM_PROMPT = """
You are a Marathi transcript segmentation and cleaning system.

Your task is to process Marathi conversational speech transcripts.

The transcript may contain:
- Marathi dialects such as Varhadi
- missing punctuation
- YouTube auto-transcription errors
- repeated words
- speaker markers such as >>
- annotations such as [हास्य], [संगीत], [टाळ्या]
- multiple spoken sentences inside one transcript segment

Your job is ONLY to clean and segment the transcript.

VERY IMPORTANT RULES:

1. PRESERVE THE ORIGINAL DIALECT.

2. DO NOT translate dialect Marathi into Standard Marathi.

3. DO NOT replace dialect words with standard Marathi words.

For example:

"मले" must remain "मले".
DO NOT change it to "मला".

"तुले" must remain "तुले".
DO NOT change it to "तुला".

4. DO NOT paraphrase.

5. DO NOT summarize.

6. DO NOT add information that was not spoken.

7. You MAY:
   - remove >> speaker markers
   - remove [हास्य], [संगीत], [टाळ्या], etc.
   - add punctuation
   - split multiple sentences
   - remove obvious non-speech transcript artifacts
   - make very minor corrections when an obvious transcription
     artifact prevents understanding

8. Preserve dialect wording whenever possible.

9. Each output item must represent ONE natural spoken
   sentence or complete conversational utterance.

Examples:

"काय दादा?"

"हो."

"बस आलोच मी."

"चल."

"कुठं जातोय?"

10. DO NOT combine several independent spoken sentences
    into one item.

11. DO NOT split a single natural sentence unnecessarily.

12. Every sentence MUST contain the indexes of the original
    transcript segments used to create it.

13. The "segments" field must contain ONLY original segment indexes.

14. Do not invent segment indexes.

15. Preserve the original wording and dialect as closely
    as possible.

Your output will be inserted into a dialect NLP dataset.
Accuracy of dialect preservation is more important than
grammatical correction.
"""


# ---------------------------------------------------------
# Process one chunk
# ---------------------------------------------------------

def process_chunk(segments):
    """
    Send one chunk of transcript segments to Gemini.
    """

    transcript_text = []

    for segment in segments:
        transcript_text.append(
            f"[{segment['index']}] {segment['text']}"
        )

    joined_transcript = "\n".join(
        transcript_text
    )

    prompt = f"""
{SYSTEM_PROMPT}

Now process the following transcript.

IMPORTANT:
- Split every natural spoken sentence/utterance separately.
- Preserve dialect.
- Do not standardize dialect words.
- Keep the original segment indexes.

TRANSCRIPT:

{joined_transcript}
"""

    response = client.models.generate_content(
        model=GEMINI_MODEL,
        contents=prompt,
        config={
            "response_mime_type": "application/json",
            "response_schema": SegmentationResult,
        }
    )

    result = SegmentationResult.model_validate_json(
        response.text
    )

    return result.sentences


# ---------------------------------------------------------
# Main segmentation function
# ---------------------------------------------------------

def clean_and_segment(
    segments,
    chunk_size=30
):
    """
    Split the transcript into smaller chunks and process
    each chunk using Gemini.

    Example:

    335 segments
        ↓
    30 segments
        ↓
    Gemini
        ↓
    next 30
        ↓
    Gemini
        ↓
    ...
    """

    if not segments:
        return []

    all_sentences = []

    total_segments = len(segments)

    for start in range(
        0,
        total_segments,
        chunk_size
    ):

        end = min(
            start + chunk_size,
            total_segments
        )

        chunk = segments[start:end]

        print(
            f"\nProcessing segments "
            f"{start + 1}-{end} "
            f"of {total_segments}..."
        )

        try:

            chunk_sentences = process_chunk(
                chunk
            )

            all_sentences.extend(
                chunk_sentences
            )

            print(
                f"✓ Gemini generated "
                f"{len(chunk_sentences)} sentences"
            )

        except Exception as error:

            print(
                f"❌ Error processing "
                f"segments {start + 1}-{end}"
            )

            print(error)

            raise

    return all_sentences


# ---------------------------------------------------------
# Attach timestamps
# ---------------------------------------------------------

def attach_timestamps(
    llm_sentences,
    original_segments
):
    """
    Attach YouTube timestamps to each sentence.
    """

    segment_map = {
        segment["index"]: segment
        for segment in original_segments
    }

    final_sentences = []

    for sentence in llm_sentences:

        text = sentence.text.strip()

        indexes = sentence.segments

        if not text or not indexes:
            continue

        valid_segments = [
            segment_map[index]
            for index in indexes
            if index in segment_map
        ]

        if not valid_segments:
            continue

        start = min(
            segment["start"]
            for segment in valid_segments
        )

        end = max(
            segment["end"]
            for segment in valid_segments
        )

        final_sentences.append({
            "text": text,
            "start": start,
            "end": end
        })

    return final_sentences



