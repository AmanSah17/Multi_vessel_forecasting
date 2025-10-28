"""
Populate Maritime Database with Sample Data
Loads data from predictions CSV and populates the database
"""

import pandas as pd
import sqlite3
import logging
from pathlib import Path
import os

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Database path
DB_PATH = "f:\\Maritime_NLU_Repo\\backend\\nlu_chatbot\\maritime_sample_0104.db"

def populate_database():
    """Populate database with sample vessel data"""
    
    logger.info("🚀 Starting database population...")
    
    # Find prediction CSV
    csv_path = Path("results/xgboost_advanced_50_vessels/all_predictions.csv")
    
    if not csv_path.exists():
        logger.error(f"❌ CSV not found: {csv_path}")
        return False
    
    try:
        # Load predictions
        logger.info(f"📂 Loading data from {csv_path}")
        df = pd.read_csv(csv_path)
        logger.info(f"✅ Loaded {len(df)} records")
        
        # Show columns
        logger.info(f"📊 Columns: {list(df.columns)}")
        
        # Connect to database
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        
        # Create vessel_data table if not exists
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
        
        # Clear existing data
        cursor.execute("DELETE FROM vessel_data")
        logger.info("🗑️  Cleared existing data")
        
        # Prepare data for insertion
        # Assuming columns: vessel_name, mmsi, imo, lat, lon, sog, cog, timestamp
        required_cols = ['vessel_name', 'mmsi', 'lat', 'lon', 'sog', 'cog']
        
        # Check which columns exist
        available_cols = [col for col in required_cols if col in df.columns]
        logger.info(f"✅ Available columns: {available_cols}")
        
        if len(available_cols) < 4:
            logger.warning("⚠️  Not enough columns. Checking alternative names...")
            # Try alternative column names
            col_mapping = {
                'vessel_name': ['VesselName', 'vessel_name', 'Vessel Name'],
                'mmsi': ['MMSI', 'mmsi'],
                'lat': ['LAT', 'lat', 'Latitude'],
                'lon': ['LON', 'lon', 'Longitude'],
                'sog': ['SOG', 'sog', 'Speed'],
                'cog': ['COG', 'cog', 'Course']
            }
            
            for key, alternatives in col_mapping.items():
                for alt in alternatives:
                    if alt in df.columns:
                        df[key] = df[alt]
                        logger.info(f"   Mapped {alt} -> {key}")
                        break
        
        # Insert data
        inserted = 0
        for idx, row in df.iterrows():
            try:
                cursor.execute('''
                    INSERT INTO vessel_data 
                    (VesselName, MMSI, IMO, LAT, LON, SOG, COG, BaseDateTime, Status)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    str(row.get('vessel_name', 'Unknown')),
                    int(row.get('mmsi', 0)) if pd.notna(row.get('mmsi')) else 0,
                    int(row.get('imo', 0)) if pd.notna(row.get('imo')) else 0,
                    float(row.get('lat', 0)) if pd.notna(row.get('lat')) else 0,
                    float(row.get('lon', 0)) if pd.notna(row.get('lon')) else 0,
                    float(row.get('sog', 0)) if pd.notna(row.get('sog')) else 0,
                    float(row.get('cog', 0)) if pd.notna(row.get('cog')) else 0,
                    str(row.get('timestamp', '')),
                    'Active'
                ))
                inserted += 1
            except Exception as e:
                logger.warning(f"⚠️  Error inserting row {idx}: {e}")
                continue
        
        conn.commit()
        logger.info(f"✅ Inserted {inserted} records into database")
        
        # Verify
        cursor.execute("SELECT COUNT(*) FROM vessel_data")
        count = cursor.fetchone()[0]
        logger.info(f"📊 Total records in database: {count}")
        
        # Show sample vessels
        cursor.execute("SELECT DISTINCT VesselName FROM vessel_data LIMIT 10")
        vessels = cursor.fetchall()
        logger.info(f"\n📋 Sample Vessels in Database:")
        for i, (vessel,) in enumerate(vessels, 1):
            logger.info(f"   {i}. {vessel}")
        
        conn.close()
        logger.info("\n✅ Database population completed successfully!")
        return True
        
    except Exception as e:
        logger.error(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    populate_database()

