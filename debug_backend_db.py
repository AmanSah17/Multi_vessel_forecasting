"""
Debug Backend Database Issue
"""

import sys
import os

# Add backend path
sys.path.insert(0, "f:\\Maritime_NLU_Repo\\backend\\nlu_chatbot\\src\\app")

# Simulate what the backend does
base_dir = "f:\\Maritime_NLU_Repo\\backend\\nlu_chatbot"
default_db = os.path.join(base_dir, "maritime_data.db")
sample_db = os.path.join(base_dir, "maritime_sample_0104.db")

print(f"Base dir: {base_dir}")
print(f"Default DB: {default_db}")
print(f"Sample DB: {sample_db}")
print(f"Default DB exists: {os.path.exists(default_db)}")
print(f"Sample DB exists: {os.path.exists(sample_db)}")

# Now import and test
from db_handler import MaritimeDB

# Test 1: Check default DB
print("\n1️⃣  Testing default DB")
try:
    db1 = MaritimeDB(default_db)
    vessels1 = db1.get_all_vessel_names()
    print(f"   Vessels: {len(vessels1)}")
except Exception as e:
    print(f"   Error: {e}")

# Test 2: Check sample DB
print("\n2️⃣  Testing sample DB")
try:
    db2 = MaritimeDB(sample_db)
    vessels2 = db2.get_all_vessel_names()
    print(f"   Vessels: {len(vessels2)}")
    print(f"   Vessel list: {vessels2}")
except Exception as e:
    print(f"   Error: {e}")

# Test 3: Check what backend would use
print("\n3️⃣  Simulating backend logic")
db_path = default_db
try:
    if not os.path.exists(db_path):
        raise FileNotFoundError(f"DB not found: {db_path}")
    
    temp_db = MaritimeDB(db_path)
    vessel_count = len(temp_db.get_all_vessel_names())
    print(f"   Default DB vessel count: {vessel_count}")
    
    if vessel_count == 0:
        print(f"   Main DB is empty, switching to sample DB")
        db_path = sample_db
        if os.path.exists(sample_db):
            print(f"   Using sample DB: {sample_db}")
except Exception as e:
    print(f"   Error: {e}")
    sample_db_path = os.path.join(base_dir, "maritime_sample_0104.db")
    if os.path.exists(sample_db_path):
        print(f"   Using sample DB: {sample_db_path}")
        db_path = sample_db_path

print(f"\n   Final DB path: {db_path}")

# Test 4: Create final DB instance
print("\n4️⃣  Creating final DB instance")
try:
    db = MaritimeDB(db_path)
    vessels = db.get_all_vessel_names()
    print(f"   Vessels: {len(vessels)}")
    print(f"   Vessel list: {vessels}")
    print(f"   DB path: {db.db_path}")
    print(f"   Engine: {db.engine}")
except Exception as e:
    print(f"   Error: {e}")
    import traceback
    traceback.print_exc()

