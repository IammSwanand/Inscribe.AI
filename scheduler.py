# scheduler.py
import os
import chromadb
import datetime
import schedule
import time

from dotenv import load_dotenv
load_dotenv()

CHROMA_DIR = os.getenv("CHROMA_PERSIST_DIR", "./chroma_db")
COLLECTION_NAME = "legal_docs"

def delete_old_docs():
    print(f"[{datetime.datetime.now()}] Running scheduled cleanup...")
    client = chromadb.PersistentClient(path=CHROMA_DIR)
    
    if COLLECTION_NAME not in [c.name for c in client.list_collections()]:
        print("Collection does not exist, skipping cleanup.")
        return

    collection = client.get_collection(name=COLLECTION_NAME)
    
    Seven_Days_ago_timestamp = int((datetime.datetime.now() -datetime.timedelta(days=7)).timestamp())

    results = collection.get(
        where={
            "created_at": {
                "$lt": Seven_Days_ago_timestamp  # Compare as timestamp
            }
        },
        include=['metadatas']
    )
    
    if results['ids']:
        print(f"Found {len(results['ids'])} documents older than 7 days. Deleting...")
        collection.delete(ids=results['ids'])
        print("Deletion complete.")
    else:
        print("No documents found to delete.")

schedule.every().day.at("02:00").do(delete_old_docs)  # Schedule daily at 2 AM
print("Scheduler started. Deletion will run every 7 Days at 2:00 AM")

while True:
    schedule.run_pending()
    time.sleep(1)