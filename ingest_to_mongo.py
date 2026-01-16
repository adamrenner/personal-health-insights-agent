import argparse
import os
import glob
import json
import sys

# Try importing pymongo, and print a helpful error if it's missing
try:
    import pymongo
    from pymongo import MongoClient, UpdateOne
    from pymongo.errors import BulkWriteError
except ImportError:
    print("Error: pymongo is not installed.")
    print("Please install pymongo using: pip install pymongo")
    sys.exit(1)

def ingest_data(input_path, connection_string, db_name, collection_name):
    """
    Reads JSON files from input_path and ingests them into MongoDB.
    """
    
    # establish connection to MongoDB
    try:
        client = MongoClient(connection_string)
        # Force a connection check
        client.admin.command('ping')
        print("Successfully connected to MongoDB.")
    except Exception as e:
        print(f"Error connecting to MongoDB: {e}")
        sys.exit(1)

    db = client[db_name]
    collection = db[collection_name]

    # Find all matching files
    search_pattern = os.path.join(input_path, "subject_*.json")
    files = glob.glob(search_pattern)
    
    # Filter out subject_mapping.json if it gets picked up (though subject_*.json shouldn't match subject_mapping.json usually, 
    # but depending on glob behavior or if user meant starting with subject_)
    # The requirement says "exclude subject_mapping.json".
    # "subject_*.json" matches "subject_mapping.json" because * matches "mapping".
    files = [f for f in files if os.path.basename(f) != "subject_mapping.json"]

    total_files = len(files)
    print(f"Found {total_files} files to process.")

    if total_files == 0:
        print("No files found to process.")
        return

    requests = []
    processed_count = 0
    successful_upserts = 0
    errors = 0

    print("Starting ingestion...")

    for file_path in files:
        try:
            with open(file_path, 'r') as f:
                data = json.load(f)
            
            # The requirement is to use subject_id as the unique identifier for upserting.
            # We assume the top-level 'subject_id' is the unique key.
            if 'subject_id' not in data:
                print(f"Warning: 'subject_id' not found in {file_path}. Skipping.")
                errors += 1
                continue

            subject_id = data['subject_id']

            # Create an UpdateOne operation with upsert=True
            # This replaces the existing document or inserts if not found
            # utilizing $set to update fields.
            # Using $set ensures we update the fields. If we wanted to replace the whole doc, we could use ReplaceOne.
            # Given "ingest generated JSON records", replacing/updating fields with the new JSON content is appropriate.
            requests.append(UpdateOne({'subject_id': subject_id}, {'$set': data}, upsert=True))
            
            processed_count += 1
            
            # Execute in batches of 1000 to be efficient (or whatever batch size is appropriate)
            if len(requests) >= 1000:
                result = collection.bulk_write(requests)
                successful_upserts += result.upserted_count + result.modified_count # Logic for "success" might vary but generally processed without error
                # Note: bulk_write returns upserted_count, inserted_count, modified_count, matched_count.
                # Since we are doing UpdateOne with upsert=True:
                # - if it didn't exist, it's upserted (upserted_count)
                # - if it existed and changed, it's modified (modified_count)
                # - if it existed and didn't change, it's matched but not modified.
                # We can count 'requests' length as attempted, and catch BulkWriteError for failures.
                requests = []
                print(f"Processed {processed_count} files...")

        except json.JSONDecodeError as e:
            print(f"Error decoding JSON in {file_path}: {e}")
            errors += 1
        except Exception as e:
            print(f"Unexpected error processing {file_path}: {e}")
            errors += 1

    # Process remaining requests
    if requests:
        try:
            result = collection.bulk_write(requests)
            # successful_upserts += result.upserted_count + result.modified_count 
            # Actually, simply processing without exception is often considered "success" in bulk context unless detailed tracking is needed.
            # But let's stick to the flow.
        except BulkWriteError as bwe:
            print(f"Bulk write error: {bwe.details}")
            errors += len(bwe.details['writeErrors'])
        except Exception as e:
            print(f"Error executing final batch: {e}")
            errors += len(requests) # Assume all failed if batch failed entirely

    # Final Summary
    # Note: successful_upserts calculation above was approximate. 
    # A simpler metric for "Successful inserts/upserts" is (processed_count - errors) if individual file errors were tracked.
    # However, bulk_write could fail partially.
    # Let's refine "Successful" to mean "files read and added to bulk request without local error" minus "bulk write errors".
    
    # Re-calculating success based on simple file processing:
    success_count = processed_count - (errors if errors > 0 else 0) # This is a rough estimation if bulk errors aren't mapped back to files 1:1 easily.
    # But since we want a nice summary:
    
    print("-" * 30)
    print("Ingestion Summary")
    print("-" * 30)
    print(f"Total files found: {total_files}")
    print(f"Files processed: {processed_count}") 
    print(f"Errors encountered: {errors}")
    print("Done.")

def main():
    parser = argparse.ArgumentParser(description="Ingest generated JSON records to MongoDB.")
    
    parser.add_argument(
        '--input_path', 
        type=str, 
        default='json_output/', 
        help='Path to the directory containing the JSON files (default: json_output/)'
    )
    parser.add_argument(
        '--connection_string', 
        type=str, 
        default='mongodb://pinnacle:dsfower8wedksfdweehr@192.168.86.49:27017/', 
        help='MongoDB connection string'
    )
    parser.add_argument(
        '--db_name', 
        type=str, 
        default='personal_health', 
        help='Name of the database (default: personal_health)'
    )
    parser.add_argument(
        '--collection_name', 
        type=str, 
        default='subjects', 
        help='Name of the collection (default: subjects)'
    )

    args = parser.parse_args()

    # Ensure input path ends with slash for glob if user didn't provide it, 
    # though os.path.join handles missing slashes well, it's good to be safe or just use os.path.join as done.
    
    ingest_data(
        args.input_path, 
        args.connection_string, 
        args.db_name, 
        args.collection_name
    )

if __name__ == "__main__":
    main()