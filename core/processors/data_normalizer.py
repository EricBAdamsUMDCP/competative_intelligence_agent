# core/processors/data_normalizer.py
import re
import logging
from datetime import datetime
from typing import Dict, List, Any, Union, Optional
import uuid

class DataNormalizer:
    """Normalizes data from different sources to a consistent format.
    
    This class standardizes data from multiple sources (SAM.gov, USASpending.gov, etc.)
    into a consistent format for storage in the knowledge graph.
    
    Attributes:
        logger: Logger instance for this class
    """
    
    def __init__(self):
        """Initialize the data normalizer."""
        self.logger = logging.getLogger("data_normalizer")
    
    def normalize(self, data: Dict[str, Any], source: str) -> Dict[str, Any]:
        """Normalize data from a specific source.
        
        Args:
            data: Raw data dictionary to normalize
            source: Source of the data (e.g., "sam.gov", "usaspending.gov")
            
        Returns:
            Normalized data dictionary
        """
        normalized = data.copy()
        
        # Apply common normalization
        normalized = self._normalize_common(normalized)
        
        # Apply source-specific normalization
        if source.lower() == "sam.gov":
            normalized = self._normalize_sam_gov(normalized)
        elif source.lower() == "usaspending.gov":
            normalized = self._normalize_usaspending(normalized)
        # Add other sources as needed
        
        # Ensure all standard fields exist
        normalized = self._ensure_standard_fields(normalized)
        
        return normalized
    
    def _normalize_common(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Apply common normalization steps across all sources.
        
        Args:
            data: Data dictionary to normalize
            
        Returns:
            Normalized data dictionary
        """
        normalized = data.copy()
        
        # Normalize agency names
        if 'agency_data' in normalized and 'name' in normalized['agency_data']:
            normalized['agency_data']['name'] = self._normalize_agency_name(
                normalized['agency_data']['name']
            )
        
        # Normalize dates
        for date_field in ['date', 'published_date', 'award_date', 'response_date', 'updated_date']:
            if date_field in normalized and normalized[date_field]:
                normalized[date_field] = self._normalize_date(normalized[date_field])
        
        # Normalize IDs
        for id_field in ['id', 'opportunity_id', 'contract_id', 'agency_id', 'contractor_id']:
            if id_field in normalized and normalized[id_field]:
                normalized[id_field] = str(normalized[id_field])
        
        # Normalize monetary values
        for value_field in ['value', 'amount', 'total_value', 'obligated_amount']:
            if value_field in normalized and normalized[value_field] is not None:
                normalized[value_field] = self._normalize_monetary_value(normalized[value_field])
        
        return normalized
    
    def _normalize_sam_gov(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Apply SAM.gov-specific normalization.
        
        Args:
            data: SAM.gov data dictionary
            
        Returns:
            Normalized SAM.gov data dictionary
        """
        normalized = data.copy()
        
        # Extract contract types
        if 'type' in normalized:
            contract_type = normalized['type']
            contract_type_description = self._get_contract_type_description(contract_type)
            
            normalized['contract_type'] = {
                'code': contract_type,
                'description': contract_type_description
            }
        
        # Normalize opportunity type
        if 'noticeType' in normalized:
            notice_type = normalized['noticeType']
            normalized['opportunity_type'] = self._normalize_notice_type(notice_type)
        
        # Extract set-aside information
        if 'setAside' in normalized and normalized['setAside']:
            normalized['set_aside'] = self._normalize_set_aside(normalized['setAside'])
        
        # Ensure award_data is present
        if 'award' in normalized and normalized['award'] and 'award_data' not in normalized:
            # Format award data for our knowledge graph
            award = normalized['award']
            normalized['award_data'] = {
                'agency_id': normalized.get('agencyCode', normalized.get('agencyId', '')),
                'agency_name': normalized.get('agency', ''),
                'contractor_id': award.get('awardeeId', award.get('awardeeCode', award.get('awardeeUEI', ''))),
                'contractor_name': award.get('awardee', ''),
                'opportunity_id': normalized.get('opportunityId', ''),
                'title': normalized.get('title', ''),
                'description': normalized.get('description', ''),
                'value': self._normalize_monetary_value(award.get('amount', 0)),
                'award_date': self._normalize_date(award.get('date', award.get('awardDate', ''))),
                'naics_code': normalized.get('naicsCode', ''),
                'contract_number': award.get('contractNumber', '')
            }
        
        return normalized
    
    def _normalize_usaspending(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Apply USASpending.gov-specific normalization.
        
        Args:
            data: USASpending.gov data dictionary
            
        Returns:
            Normalized USASpending.gov data dictionary
        """
        normalized = data.copy()
        
        # If contract_data is present, it's already been normalized by the collector
        if 'contract_data' in normalized:
            # Make sure award_data is present and in standard format
            if 'award_data' not in normalized:
                contract_data = normalized['contract_data']
                normalized['award_data'] = {
                    'agency_id': self._extract_nested(contract_data, ['awarding_agency', 'id'], ''),
                    'agency_name': self._extract_nested(contract_data, ['awarding_agency', 'name'], ''),
                    'contractor_id': self._extract_nested(contract_data, ['recipient', 'uei'], 
                                                         self._extract_nested(contract_data, ['recipient', 'duns'], '')),
                    'contractor_name': self._extract_nested(contract_data, ['recipient', 'name'], ''),
                    'opportunity_id': contract_data.get('id', ''),
                    'title': contract_data.get('title', ''),
                    'description': contract_data.get('title', ''),
                    'value': contract_data.get('value', 0),
                    'award_date': contract_data.get('award_date', ''),
                    'naics_code': self._extract_nested(contract_data, ['naics', 'code'], ''),
                    'contract_number': contract_data.get('id', ''),
                    'period_of_performance': (
                        f"{self._extract_nested(contract_data, ['period_of_performance', 'start_date'], '')} to "
                        f"{self._extract_nested(contract_data, ['period_of_performance', 'end_date'], '')}"
                    ),
                    'place_of_performance': self._format_place_of_performance(
                        self._extract_nested(contract_data, ['place_of_performance'], {})
                    ),
                    'contract_type': contract_data.get('type_description', '')
                }
        else:
            # Handle raw USASpending data if needed
            # This would be for cases where the USASpendingCollector didn't already normalize
            pass
        
        return normalized
    
    def _ensure_standard_fields(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Ensure all standard fields exist in the normalized data.
        
        Args:
            data: Data dictionary to check
            
        Returns:
            Data dictionary with all standard fields
        """
        # Define standard fields and their default values
        standard_fields = {
            'id': '',
            'source': '',
            'collection_time': '',
            'agency_data': {
                'id': '',
                'name': ''
            },
            'award_data': {
                'agency_id': '',
                'agency_name': '',
                'contractor_id': '',
                'contractor_name': '',
                'opportunity_id': '',
                'title': '',
                'description': '',
                'value': 0,
                'award_date': '',
                'naics_code': '',
                'contract_number': ''
            }
        }
        
        result = data.copy()
        
        # Ensure top-level fields exist
        for field, default in standard_fields.items():
            if field not in result:
                result[field] = default
            elif isinstance(default, dict) and isinstance(result[field], dict):
                # For nested dictionaries, ensure all sub-fields exist
                for sub_field, sub_default in default.items():
                    if sub_field not in result[field]:
                        result[field][sub_field] = sub_default
        
        # If id is empty but opportunity_id exists in award_data, use that
        if not result['id'] and result['award_data']['opportunity_id']:
            result['id'] = result['award_data']['opportunity_id']
        
        # If id is still empty, generate a UUID
        if not result['id']:
            result['id'] = str(uuid.uuid4())
        
        return result
    
    def _normalize_agency_name(self, name: str) -> str:
        """Normalize agency name to a standard format.
        
        Args:
            name: Agency name to normalize
            
        Returns:
            Normalized agency name
        """
        if not name:
            return ""
            
        # Standard agency name replacements
        agency_mappings = {
            "DEPT OF DEFENSE": "Department of Defense",
            "DOD": "Department of Defense",
            "DEPARTMENT OF DEFENSE": "Department of Defense",
            "DEPT OF HEALTH AND HUMAN SERVICES": "Department of Health and Human Services",
            "HHS": "Department of Health and Human Services",
            "DEPT OF HOMELAND SECURITY": "Department of Homeland Security",
            "DHS": "Department of Homeland Security",
            "GENERAL SERVICES ADMINISTRATION": "General Services Administration",
            "GSA": "General Services Administration",
            "DEPT OF VETERANS AFFAIRS": "Department of Veterans Affairs",
            "VA": "Department of Veterans Affairs",
            "DEPT OF COMMERCE": "Department of Commerce",
            "DOC": "Department of Commerce",
            "DEPT OF ENERGY": "Department of Energy",
            "DOE": "Department of Energy",
            "DEPT OF JUSTICE": "Department of Justice",
            "DOJ": "Department of Justice",
            "DEPT OF STATE": "Department of State",
            "DOS": "Department of State",
            "DEPT OF TREASURY": "Department of the Treasury",
            "TREASURY": "Department of the Treasury",
            "DEPT OF TRANSPORTATION": "Department of Transportation",
            "DOT": "Department of Transportation",
            "DEPT OF AGRICULTURE": "Department of Agriculture",
            "USDA": "Department of Agriculture",
            "DEPT OF LABOR": "Department of Labor",
            "DOL": "Department of Labor",
            "DEPT OF EDUCATION": "Department of Education",
            "ED": "Department of Education",
            "DEPT OF HOUSING AND URBAN DEVELOPMENT": "Department of Housing and Urban Development",
            "HUD": "Department of Housing and Urban Development",
            "DEPT OF THE INTERIOR": "Department of the Interior",
            "DOI": "Department of the Interior",
            "ENVIRONMENTAL PROTECTION AGENCY": "Environmental Protection Agency",
            "EPA": "Environmental Protection Agency",
            "NATIONAL AERONAUTICS AND SPACE ADMINISTRATION": "National Aeronautics and Space Administration",
            "NASA": "National Aeronautics and Space Administration",
            "SMALL BUSINESS ADMINISTRATION": "Small Business Administration",
            "SBA": "Small Business Administration"
        }
        
        # Check for exact matches
        normalized_name = agency_mappings.get(name.strip().upper(), name)
        
        # Check for partial matches
        if normalized_name == name:
            for key, value in agency_mappings.items():
                if key in name.upper():
                    normalized_name = value
                    break
        
        return normalized_name
    
    def _normalize_date(self, date_str: Optional[str]) -> str:
        """Normalize date to ISO format.
        
        Args:
            date_str: Date string to normalize
            
        Returns:
            ISO formatted date string
        """
        if not date_str:
            return ""
            
        # Return if already in ISO format
        if re.match(r'^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}', date_str):
            return date_str
            
        # Try different date formats
        date_formats = [
            '%Y-%m-%d',            # 2023-01-15
            '%m/%d/%Y',            # 01/15/2023
            '%Y-%m-%dT%H:%M:%S',   # 2023-01-15T00:00:00
            '%Y-%m-%dT%H:%M:%S.%f', # 2023-01-15T00:00:00.000
            '%Y/%m/%d',            # 2023/01/15
            '%b %d, %Y',           # Jan 15, 2023
            '%B %d, %Y',           # January 15, 2023
            '%d %b %Y',            # 15 Jan 2023
            '%d %B %Y',            # 15 January 2023
            '%Y%m%d'               # 20230115
        ]
        
        for fmt in date_formats:
            try:
                dt = datetime.strptime(date_str, fmt)
                return dt.isoformat()
            except ValueError:
                continue
        
        # Return as-is if we couldn't parse it
        return date_str
    
    def _normalize_monetary_value(self, value: Union[str, int, float]) -> float:
        """Normalize monetary value to float.
        
        Args:
            value: Monetary value to normalize
            
        Returns:
            Normalized float value
        """
        if value is None:
            return 0.0
            
        # Handle string values
        if isinstance(value, str):
            # Remove currency symbols, commas, and spaces
            clean_value = value.replace('$', '').replace(',', '').replace(' ', '')
            try:
                return float(clean_value)
            except ValueError:
                self.logger.warning(f"Failed to parse monetary value: {value}")
                return 0.0
        
        # Handle numeric values
        try:
            return float(value)
        except (ValueError, TypeError):
            self.logger.warning(f"Failed to parse monetary value: {value}")
            return 0.0
    
    def _get_contract_type_description(self, code: str) -> str:
        """Get contract type description from code.
        
        Args:
            code: Contract type code
            
        Returns:
            Contract type description
        """
        contract_types = {
            "A": "Fixed Price Redetermination",
            "B": "Fixed Price Level of Effort",
            "C": "No Cost",
            "D": "Cost Plus Award Fee",
            "E": "Cost Plus Fixed Fee",
            "F": "Cost No Fee",
            "J": "Fixed Price",
            "K": "Fixed Price with Economic Price Adjustment",
            "L": "Fixed Price Incentive",
            "M": "Time and Materials",
            "R": "Cost Plus",
            "S": "Cost Sharing",
            "T": "Time and Materials",
            "U": "Cost Plus Incentive Fee",
            "V": "Cost Plus Incentive Fee",
            "Y": "Time and Materials",
            "Z": "Labor Hours"
        }
        
        return contract_types.get(code, "Other")
    
    def _normalize_notice_type(self, notice_type: str) -> Dict[str, str]:
        """Normalize SAM.gov notice type.
        
        Args:
            notice_type: Notice type code or description
            
        Returns:
            Normalized notice type dictionary
        """
        notice_types = {
            "PRESOL": "Pre-solicitation",
            "COMBINE": "Combined Synopsis/Solicitation",
            "AWARD": "Award Notice",
            "ARCHIVE": "Archived",
            "SRCSGT": "Sources Sought",
            "SSALE": "Sale of Surplus Property",
            "SNOTE": "Special Notice",
            "ITB": "Intent to Bundle",
            "JA": "Justification and Approval",
            "FAIROPP": "Fair Opportunity / Limited Sources Justification"
        }
        
        # Try to match by code
        if notice_type in notice_types:
            return {
                "code": notice_type,
                "description": notice_types[notice_type]
            }
        
        # Try to match by description
        for code, desc in notice_types.items():
            if desc.lower() == notice_type.lower():
                return {
                    "code": code,
                    "description": desc
                }
        
        # Default to original value
        return {
            "code": notice_type,
            "description": notice_type
        }
    
    def _normalize_set_aside(self, set_aside: str) -> Dict[str, str]:
        """Normalize SAM.gov set-aside code.
        
        Args:
            set_aside: Set-aside code
            
        Returns:
            Normalized set-aside dictionary
        """
        set_aside_types = {
            "SBA": "Small Business Set-Aside",
            "8A": "8(a) Set-Aside",
            "SDVOSBC": "Service-Disabled Veteran-Owned Small Business",
            "WOSB": "Women-Owned Small Business",
            "EDWOSB": "Economically Disadvantaged Women-Owned Small Business",
            "HUBZone": "HUBZone Set-Aside",
            "TotalSB": "Total Small Business Set-Aside",
            "PartialSB": "Partial Small Business Set-Aside",
            "HBCU/MI": "Historically Black College or University / Minority Institution",
            "CompSBSA": "Competitive Small Business Set-Aside"
        }
        
        # Try to match by code
        if set_aside in set_aside_types:
            return {
                "code": set_aside,
                "description": set_aside_types[set_aside]
            }
        
        # Try to match by description
        for code, desc in set_aside_types.items():
            if desc.lower() == set_aside.lower():
                return {
                    "code": code,
                    "description": desc
                }
        
        # Default to original value
        return {
            "code": set_aside,
            "description": set_aside
        }
    
    def _extract_nested(self, data: Dict[str, Any], path: List[str], default: Any = None) -> Any:
        """Extract a field from a nested dictionary.
        
        Args:
            data: Dictionary to extract from
            path: List of keys to traverse
            default: Default value if not found
            
        Returns:
            Extracted value or default
        """
        current = data
        
        for key in path:
            if not isinstance(current, dict) or key not in current:
                return default
            current = current[key]
        
        return current
    
    def _format_place_of_performance(self, pop: Dict[str, Any]) -> str:
        """Format place of performance as a string.
        
        Args:
            pop: Place of performance dictionary
            
        Returns:
            Formatted place of performance string
        """
        if not pop:
            return ""
            
        parts = []
        
        if pop.get('city'):
            parts.append(pop['city'])
        
        if pop.get('state'):
            parts.append(pop['state'])
        
        if pop.get('zip'):
            parts.append(pop['zip'])
        
        if pop.get('country') and pop.get('country').upper() != 'UNITED STATES':
            parts.append(pop['country'])
        
        return ', '.join(parts)
    
    def batch_normalize(self, data_list: List[Dict[str, Any]], source: str) -> List[Dict[str, Any]]:
        """Normalize a list of data dictionaries.
        
        Args:
            data_list: List of data dictionaries to normalize
            source: Source of the data
            
        Returns:
            List of normalized data dictionaries
        """
        return [self.normalize(item, source) for item in data_list]