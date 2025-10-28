"""
Test Database Connection from Backend
"""

import sys
import os

# Add backend path
sys.path.insert(0, "f:\\Maritime_NLU_Repo\\backend\\nlu_chatbot\\src\\app")

from db_handler import MaritimeDB

DB_PATH = "f:\\Maritime_NLU_Repo\\backend\\nlu_chatbot\\maritime_sample_0104.db"

print(f"Testing database: {DB_PATH}")
print(f"Database exists: {os.path.exists(DB_PATH)}")

try:
    db = MaritimeDB(DB_PATH)
    print(f"✅ Connected to database")
    
    # Get all vessel names
    vessels = db.get_all_vessel_names()
    print(f"✅ Retrieved {len(vessels)} vessels")
    print(f"Vessels: {vessels}")
    
except Exception as e:
    print(f"❌ Error: {e}")
    import traceback
    traceback.print_exc()

