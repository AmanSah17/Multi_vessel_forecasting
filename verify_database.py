"""
Verify Database Contents
"""

import sqlite3
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

DB_PATH = "f:\\Maritime_NLU_Repo\\backend\\nlu_chatbot\\maritime_sample_0104.db"

try:
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    # Check tables
    logger.info("📋 Tables in database:")
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
    tables = cursor.fetchall()
    for table in tables:
        logger.info(f"   • {table[0]}")
    
    # Check vessel_data
    logger.info("\n📊 vessel_data table:")
    cursor.execute("SELECT COUNT(*) FROM vessel_data")
    count = cursor.fetchone()[0]
    logger.info(f"   Total records: {count}")
    
    # Check distinct vessels
    cursor.execute("SELECT DISTINCT VesselName FROM vessel_data")
    vessels = cursor.fetchall()
    logger.info(f"   Distinct vessels: {len(vessels)}")
    
    for vessel in vessels[:10]:
        logger.info(f"      • {vessel[0]}")
    
    # Check sample data
    logger.info("\n📈 Sample data:")
    cursor.execute("SELECT VesselName, MMSI, LAT, LON, SOG, COG FROM vessel_data LIMIT 3")
    rows = cursor.fetchall()
    for row in rows:
        logger.info(f"   {row}")
    
    conn.close()
    logger.info("\n✅ Database verification complete")
    
except Exception as e:
    logger.error(f"❌ Error: {e}")
    import traceback
    traceback.print_exc()

