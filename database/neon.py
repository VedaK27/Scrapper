import os
import psycopg2
from dotenv import load_dotenv

load_dotenv()

DATABASE_URL = os.getenv("DATABASE_URL")


def get_connection():
    if not DATABASE_URL:
        raise ValueError("DATABASE_URL is not set in .env")

    return psycopg2.connect(DATABASE_URL)



def insert_corpus_records(records):
    connection = get_connection()

    try:
        cursor = connection.cursor()

        query = """
            INSERT INTO dialect_corpus (
                pair_id,
                dialect_label,
                source_text,
                target_text,
                generation_method,
                domain_context,
                dataset_split,
                source_type,
                source_url,
                video_id,
                timestamp_start,
                timestamp_end,
                transcript_source,
                verification_status
            )
            VALUES (
                %s, %s, %s, %s, %s, %s, %s,
                %s, %s, %s, %s, %s, %s, %s
            )
        """

        for record in records:

            cursor.execute(
                query,
                (
                    record["pair_id"],
                    record["dialect_label"],
                    record["source_text"],
                    record["target_text"],
                    record["generation_method"],
                    record["domain_context"],
                    record["dataset_split"],
                    record["source_type"],
                    record["source_url"],
                    record["video_id"],
                    record["timestamp_start"],
                    record["timestamp_end"],
                    record["transcript_source"],
                    record["verification_status"],
                )
            )

        connection.commit()

        print(f"Inserted {len(records)} records into Neon.")

    except Exception as e:

        connection.rollback()

        raise e

    finally:

        connection.close()

