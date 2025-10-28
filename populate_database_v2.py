"""
Populate Maritime Database with Sample Data - Version 2
Uses actual vessel data from the training dataset
"""

import pandas as pd
import sqlite3
import logging
from pathlib import Path
import numpy as np

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Database path
DB_PATH = "f:\\Maritime_NLU_Repo\\backend\\nlu_chatbot\\maritime_sample_0104.db"

def populate_database_v2():
    """Populate database with vessel data from training results"""
    
    logger.info("🚀 Starting database population (v2)...")
    
    # Find the training data
    csv_paths = [
        Path("results/xgboost_advanced_50_vessels/all_predictions.csv"),
        Path("results/kalman_lstm_predictions.csv"),
    ]
    
    df = None
    for csv_path in csv_paths:
        if csv_path.exists():
            logger.info(f"📂 Loading data from {csv_path}")
            df = pd.read_csv(csv_path)
            logger.info(f"✅ Loaded {len(df)} records")
            break
    
    if df is None:
        logger.error("❌ No CSV found")
        return False
    
    try:
        logger.info(f"📊 Columns: {list(df.columns)}")
        
        # Connect to database
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        
        # Create vessel_data table
        logger.info("📋 Creating vessel_data table...")
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS vessel_data (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                VesselName TEXT,
                MMSI INTEGER,
                IMO INTEGER,
                LAT REAL,
                LON REAL,
                SOG REAL,
                COG REAL,
                BaseDateTime TEXT,
                Status TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        cursor.execute("DELETE FROM vessel_data")
        logger.info("🗑️  Cleared existing data")
        
        # Create sample vessels with realistic data
        sample_vessels = [
            {"name": "CHAMPAGNE CHER", "mmsi": 228339611, "imo": 9400000},
            {"name": "MAERSK SEALAND", "mmsi": 219014969, "imo": 9400001},
            {"name": "EVER GIVEN", "mmsi": 353136000, "imo": 9400002},
            {"name": "MSC GULSUN", "mmsi": 636014407, "imo": 9400003},
            {"name": "OOCL HONG KONG", "mmsi": 563099700, "imo": 9400004},
            {"name": "COSCO SHIPPING", "mmsi": 413393000, "imo": 9400005},
            {"name": "PACIFIC PRINCESS", "mmsi": 310627000, "imo": 9400006},
            {"name": "QUEEN MARY 2", "mmsi": 311000000, "imo": 9400007},
            {"name": "CARNIVAL VISTA", "mmsi": 367671820, "imo": 9400008},
            {"name": "ROYAL CARIBBEAN", "mmsi": 319000000, "imo": 9400009},
        ]
        
        inserted = 0
        
        # Use actual data from predictions
        if 'actual_LAT' in df.columns and 'actual_LON' in df.columns:
            logger.info("✅ Using actual position data from predictions")
            
            # Group by MMSI and take samples
            for idx, (mmsi, group) in enumerate(df.groupby('MMSI')):
                if idx >= len(sample_vessels):
                    break
                
                vessel = sample_vessels[idx]
                
                # Take last 50 records for this vessel
                sample_data = group.tail(50)
                
                for _, row in sample_data.iterrows():
                    try:
                        cursor.execute('''
                            INSERT INTO vessel_data 
                            (VesselName, MMSI, IMO, LAT, LON, SOG, COG, BaseDateTime, Status)
                            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                        ''', (
                            vessel['name'],
                            vessel['mmsi'],
                            vessel['imo'],
                            float(row.get('actual_LAT', 0)),
                            float(row.get('actual_LON', 0)),
                            float(row.get('actual_SOG', 0)),
                            float(row.get('actual_COG', 0)),
                            f"2024-01-{(inserted % 28) + 1:02d} {(inserted % 24):02d}:00:00",
                            'Active'
                        ))
                        inserted += 1
                    except Exception as e:
                        logger.warning(f"⚠️  Error: {e}")
                        continue
        
        conn.commit()
        logger.info(f"✅ Inserted {inserted} records")
        
        # Verify
        cursor.execute("SELECT COUNT(*) FROM vessel_data")
        count = cursor.fetchone()[0]
        logger.info(f"📊 Total records: {count}")
        
        # Show vessels
        cursor.execute("SELECT DISTINCT VesselName, MMSI, COUNT(*) as records FROM vessel_data GROUP BY VesselName")
        vessels = cursor.fetchall()
        logger.info(f"\n📋 Vessels in Database:")
        for vessel_name, mmsi, records in vessels:
            logger.info(f"   • {vessel_name} (MMSI: {mmsi}) - {records} records")
        
        conn.close()
        logger.info("\n✅ Database population completed!")
        return True
        
    except Exception as e:
        logger.error(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    populate_database_v2()

