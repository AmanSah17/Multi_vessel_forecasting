"""
Test NLP Parser to debug duration_minutes extraction
"""

import sys
import os

# Add backend path
sys.path.insert(0, r"f:\Maritime_NLU_Repo\backend\nlu_chatbot\src\app")

from nlp_interpreter import MaritimeNLPInterpreter

print("\n" + "="*80)
print("TESTING NLP PARSER - DURATION MINUTES EXTRACTION")
print("="*80 + "\n")

nlp = MaritimeNLPInterpreter(vessel_list=['CHAMPAGNE CHER', 'MAERSK SEALAND', 'EVER GIVEN'])

test_queries = [
    "Predict CHAMPAGNE CHER position after 30 minutes",
    "Predict MAERSK SEALAND position in 45 minutes",
    "Predict EVER GIVEN position after 1 hour",
    "Show CHAMPAGNE CHER position",
]

for query in test_queries:
    print(f"\nQuery: '{query}'")
    print("-" * 80)
    
    result = nlp.parse_query(query)
    
    print(f"  Intent: {result.get('intent')}")
    print(f"  Vessel: {result.get('vessel_name')}")
    print(f"  Time Horizon: {result.get('time_horizon')}")
    print(f"  Duration Minutes: {result.get('duration_minutes')}")
    print(f"  End DT: {result.get('end_dt')}")
    print(f"  DateTime: {result.get('datetime')}")

print("\n" + "="*80)
print("TEST COMPLETE")
print("="*80 + "\n")

