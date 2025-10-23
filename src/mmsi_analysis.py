"""
MMSI Distribution Analysis Module

Analyzes Maritime Mobile Service Identity (MMSI) distribution and formatting.
Identifies suspicious patterns and formatting issues.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, Tuple
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class MMSIAnalyzer:
    """Analyzes MMSI distribution and identifies anomalies."""
    
    # MID (Maritime Identification Digits) mapping
    MID_COUNTRY_MAP = {
        '201': 'Albania', '202': 'Andorra', '203': 'Austria',
        '204': 'Azores', '205': 'Denmark', '206': 'Malta',
        '207': 'Cyprus', '208': 'Germany', '209': 'Malta',
        '210': 'Greece', '211': 'Netherlands', '212': 'Belgium',
        '213': 'France', '214': 'Spain', '215': 'Hungary',
        '216': 'Switzerland', '218': 'Germany', '219': 'Denmark',
        '220': 'Croatia', '224': 'Russia', '230': 'Cyprus',
        '231': 'Russia', '232': 'Russia', '233': 'Ukraine',
        '234': 'Turkey', '235': 'Cyprus', '236': 'Malta',
        '237': 'Malta', '238': 'Ukraine', '239': 'Cyprus',
        '240': 'Russia', '241': 'Georgia', '242': 'Moldova',
        '243': 'Ukraine', '244': 'Russia', '245': 'Russia',
        '246': 'Russia', '247': 'Russia', '248': 'Russia',
        '250': 'Russia', '255': 'Ukraine', '257': 'Russia',
        '258': 'Russia', '261': 'Lithuania', '262': 'Latvia',
        '263': 'Estonia', '264': 'Russia', '265': 'Russia',
        '266': 'Belarus', '267': 'Russia', '268': 'Russia',
        '301': 'Anguilla', '303': 'USA', '304': 'Antigua & Barbuda',
        '305': 'Antigua & Barbuda', '306': 'Curacao', '307': 'Aruba',
        '308': 'Bahamas', '309': 'Bahamas', '310': 'Bermuda',
        '311': 'Bahamas', '312': 'Belize', '314': 'Barbados',
        '316': 'Canada', '319': 'Cayman Islands', '321': 'Costa Rica',
        '323': 'Cuba', '325': 'Dominica', '327': 'Dominican Republic',
        '329': 'Guadeloupe', '330': 'Grenada', '331': 'Greenland',
        '332': 'Guatemala', '334': 'Honduras', '336': 'Haiti',
        '338': 'USA', '339': 'Jamaica', '341': 'Saint Kitts & Nevis',
        '343': 'Saint Lucia', '345': 'Mexico', '347': 'Martinique',
        '348': 'Montserrat', '350': 'Nicaragua', '351': 'Panama',
        '352': 'Panama', '353': 'Saint Pierre & Miquelon',
        '354': 'Puerto Rico', '355': 'Saint Barthelemy',
        '356': 'Saint Martin', '357': 'Saint Vincent & Grenadines',
        '358': 'Trinidad & Tobago', '359': 'Turks & Caicos Islands',
        '401': 'Afghanistan', '403': 'Saudi Arabia', '405': 'Bangladesh',
        '408': 'Bahrain', '410': 'Bhutan', '412': 'China',
        '413': 'China', '416': 'Taiwan', '417': 'Sri Lanka',
        '419': 'India', '422': 'Iran', '423': 'Azerbaijan',
        '424': 'Pakistan', '425': 'Lebanon', '428': 'Oman',
        '431': 'Qatar', '432': 'United Arab Emirates', '434': 'Turkmenistan',
        '436': 'Kazakhstan', '440': 'Yemen', '441': 'Yemen',
        '443': 'Palestine', '450': 'North Korea', '451': 'South Korea',
        '453': 'Macau', '455': 'Maldives', '457': 'Mongolia',
        '461': 'Malaysia', '463': 'Cambodia', '466': 'Myanmar',
        '470': 'Bangladesh', '471': 'Nepal', '473': 'Sri Lanka',
        '475': 'Maldives', '477': 'Thailand', '478': 'East Timor',
        '501': 'Fiji', '529': 'Kiribati', '533': 'Nauru',
        '536': 'New Zealand', '538': 'Micronesia', '553': 'Papua New Guinea',
        '555': 'Palau', '561': 'Solomon Islands', '570': 'Samoa',
        '578': 'Vanuatu', '601': 'South Africa', '603': 'Angola',
        '605': 'Algeria', '606': 'Saint Paul & Amsterdam Islands',
        '607': 'Ascension Island', '608': 'Mauritius', '609': 'Diego Garcia',
        '610': 'Tanzania', '611': 'Madagascar', '612': 'Reunion',
        '613': 'Comoros', '615': 'Mozambique', '616': 'Mauritius',
        '617': 'Mauritius', '618': 'Mozambique', '619': 'Eswatini',
        '620': 'Botswana', '621': 'Lesotho', '622': 'Namibia',
        '624': 'Zambia', '625': 'Zimbabwe', '626': 'Malawi',
        '627': 'Zambia', '630': 'South Africa', '631': 'South Africa',
        '632': 'South Africa', '633': 'South Africa', '634': 'South Africa',
        '635': 'South Africa', '636': 'South Africa', '642': 'Egypt',
        '644': 'Libya', '645': 'Mauritania', '647': 'Senegal',
        '649': 'Cape Verde', '650': 'Gambia', '654': 'Guinea-Bissau',
        '655': 'Guinea', '656': 'Benin', '657': 'Mauritania',
        '659': 'Ivory Coast', '660': 'Cameroon', '661': 'Central African Republic',
        '662': 'Congo', '663': 'Congo', '664': 'Gabon',
        '665': 'Equatorial Guinea', '666': 'Sao Tome & Principe',
        '667': 'Angola', '668': 'Angola', '669': 'Namibia',
        '670': 'Lesotho', '674': 'Sierra Leone', '678': 'Togo',
        '682': 'Ghana', '684': 'Burkina Faso', '686': 'Mali',
        '687': 'Mauritania', '694': 'Zambia', '695': 'Liberia',
        '701': 'Argentina', '710': 'Brazil', '725': 'Chile',
        '730': 'Colombia', '735': 'Ecuador', '740': 'Falkland Islands',
        '745': 'French Guiana', '750': 'Guyana', '755': 'Paraguay',
        '760': 'Peru', '765': 'Suriname', '770': 'Uruguay',
        '775': 'Venezuela',
    }
    
    def __init__(self):
        """Initialize MMSI analyzer."""
        self.analysis_results = {}
    
    def analyze(self, df: pd.DataFrame) -> Dict:
        """
        Perform comprehensive MMSI analysis.
        
        Args:
            df: Dataframe with MMSI column
            
        Returns:
            Dictionary with analysis results
        """
        logger.info("Starting MMSI analysis...")
        
        results = {
            'total_mmsi': len(df['MMSI'].unique()),
            'total_records': len(df),
            'mmsi_distribution': self._get_distribution(df),
            'country_distribution': self._get_country_distribution(df),
            'formatting_issues': self._check_formatting(df),
            'suspicious_patterns': self._detect_suspicious_patterns(df),
        }
        
        self.analysis_results = results
        return results
    
    def _get_distribution(self, df: pd.DataFrame) -> Dict:
        """Get MMSI distribution statistics."""
        mmsi_counts = df['MMSI'].value_counts()
        return {
            'unique_mmsi': len(mmsi_counts),
            'mean_records_per_mmsi': mmsi_counts.mean(),
            'median_records_per_mmsi': mmsi_counts.median(),
            'max_records_per_mmsi': mmsi_counts.max(),
            'min_records_per_mmsi': mmsi_counts.min(),
        }
    
    def _get_country_distribution(self, df: pd.DataFrame) -> Dict:
        """Extract country from MMSI MID and get distribution."""
        df_copy = df.copy()
        df_copy['MID'] = df_copy['MMSI'].astype(str).str[:3]
        df_copy['Country'] = df_copy['MID'].map(self.MID_COUNTRY_MAP).fillna('Unknown')
        
        country_dist = df_copy['Country'].value_counts().to_dict()
        return country_dist
    
    def _check_formatting(self, df: pd.DataFrame) -> Dict:
        """Check for MMSI formatting issues."""
        mmsi_str = df['MMSI'].astype(str)
        
        issues = {
            'non_numeric': (mmsi_str.str.contains(r'[^0-9]', regex=True)).sum(),
            'wrong_length': (mmsi_str.str.len() != 9).sum(),
            'leading_zeros': (mmsi_str.str.startswith('0')).sum(),
        }
        
        return issues
    
    def _detect_suspicious_patterns(self, df: pd.DataFrame) -> Dict:
        """Detect suspicious MMSI patterns."""
        mmsi_str = df['MMSI'].astype(str)
        
        patterns = {
            'sequential_digits': (mmsi_str.str.contains(r'123456789|987654321', regex=True)).sum(),
            'all_same_digit': (mmsi_str.str.contains(r'^(\d)\1{8}$', regex=True)).sum(),
            'repeating_pattern': (mmsi_str.str.contains(r'(\d{3})\1{2}', regex=True)).sum(),
        }
        
        return patterns
    
    def visualize_distribution(self, df: pd.DataFrame, top_n: int = 20):
        """Visualize MMSI distribution."""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Top MMSI by record count
        mmsi_counts = df['MMSI'].value_counts().head(top_n)
        axes[0, 0].barh(range(len(mmsi_counts)), mmsi_counts.values)
        axes[0, 0].set_yticks(range(len(mmsi_counts)))
        axes[0, 0].set_yticklabels(mmsi_counts.index)
        axes[0, 0].set_xlabel('Record Count')
        axes[0, 0].set_title(f'Top {top_n} MMSI by Record Count')
        
        # Distribution histogram
        axes[0, 1].hist(mmsi_counts.values, bins=50, edgecolor='black')
        axes[0, 1].set_xlabel('Records per MMSI')
        axes[0, 1].set_ylabel('Frequency')
        axes[0, 1].set_title('Distribution of Records per MMSI')
        
        # Country distribution
        df_copy = df.copy()
        df_copy['MID'] = df_copy['MMSI'].astype(str).str[:3]
        df_copy['Country'] = df_copy['MID'].map(self.MID_COUNTRY_MAP).fillna('Unknown')
        country_counts = df_copy['Country'].value_counts().head(top_n)
        axes[1, 0].barh(range(len(country_counts)), country_counts.values)
        axes[1, 0].set_yticks(range(len(country_counts)))
        axes[1, 0].set_yticklabels(country_counts.index)
        axes[1, 0].set_xlabel('Record Count')
        axes[1, 0].set_title(f'Top {top_n} Countries by MMSI')
        
        # Formatting issues
        issues = self._check_formatting(df)
        axes[1, 1].bar(issues.keys(), issues.values(), color=['red', 'orange', 'yellow'])
        axes[1, 1].set_ylabel('Count')
        axes[1, 1].set_title('MMSI Formatting Issues')
        
        plt.tight_layout()
        return fig

