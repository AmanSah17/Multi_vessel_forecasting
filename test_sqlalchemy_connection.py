"""
Test SQLAlchemy Connection
"""

import pandas as pd
from sqlalchemy import create_engine
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

DB_PATH = "f:\\Maritime_NLU_Repo\\backend\\nlu_chatbot\\maritime_sample_0104.db"

logger.info(f"Testing SQLAlchemy connection to: {DB_PATH}")

try:
    # Create engine like the backend does
    engine = create_engine(
        f"sqlite:///{DB_PATH}",
        connect_args={"check_same_thread": False},
        pool_pre_ping=True
    )
    
    logger.info("✅ Engine created")
    
    # Test connection
    with engine.connect() as conn:
        logger.info("✅ Connected")
        
        # Check tables
        query = "SELECT name FROM sqlite_master WHERE type='table'"
        df = pd.read_sql_query(query, con=engine)
        logger.info(f"Tables: {df['name'].tolist()}")
        
        # Check vessel_data
        query = "SELECT COUNT(*) as count FROM vessel_data"
        df = pd.read_sql_query(query, con=engine)
        logger.info(f"vessel_data records: {df['count'].iloc[0]}")
        
        # Get distinct vessels
        query = "SELECT DISTINCT VesselName FROM vessel_data WHERE VesselName IS NOT NULL"
        df = pd.read_sql_query(query, con=engine)
        logger.info(f"Distinct vessels: {len(df)}")
        logger.info(f"Vessels: {df['VesselName'].tolist()}")
    
    logger.info("\n✅ SQLAlchemy connection test passed")
    
except Exception as e:
    logger.error(f"❌ Error: {e}")
    import traceback
    traceback.print_exc()

