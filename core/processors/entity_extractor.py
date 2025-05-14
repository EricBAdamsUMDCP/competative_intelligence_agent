# core/processors/entity_extractor.py
import spacy
from spacy.tokens import Span
from spacy.language import Language
import logging
from typing import Dict, List, Any, Set, Tuple
import os
import re

class EntityExtractor:
    """Extract entities from government contract descriptions using NLP"""
    
    def __init__(self, model_name: str = "en_core_web_lg"):
        self.logger = logging.getLogger("entity_extractor")
        
        # Load spaCy model
        try:
            self.nlp = spacy.load(model_name)
            self.logger.info(f"Loaded spaCy model: {model_name}")
        except OSError:
            self.logger.info(f"Downloading spaCy model: {model_name}")
            os.system(f"python -m spacy download {model_name}")
            self.nlp = spacy.load(model_name)
        
        # Add custom components
        self._add_custom_components()
        
        # Government contracting specific entities
        self.govcon_entities = {
            "CONTRACT_VEHICLE": ["IDIQ", "BPA", "GWAC", "GSA Schedule", "GWACs", "IDIQs", "BPAs", "OASIS", "Alliant", "SEWP", "CIO-SP3"],
            "NAICS_CODE": [],  # Will be populated from patterns
            "AGENCY": ["DoD", "Department of Defense", "DHS", "Department of Homeland Security", "HHS", "GSA"],
            "TECHNOLOGY": ["cloud", "cybersecurity", "artificial intelligence", "machine learning", "blockchain", 
                          "IoT", "5G", "quantum", "zero trust", "DevSecOps", "microservices"],
            "CLEARANCE": ["Top Secret", "TS/SCI", "Secret", "Confidential", "Public Trust", "TS", "SCI"],
            "REGULATION": ["FAR", "DFAR", "CMMC", "NIST", "FedRAMP", "FISMA", "HIPAA", "ATO"]
        }
        
        # Generate NAICS code patterns
        for i in range(100000, 1000000):
            if i % 100000 == 0:
                self.govcon_entities["NAICS_CODE"].append(str(i))

    def __init__(self, model_name: str = "en_core_web_lg"):
        self.logger = logging.getLogger("entity_extractor")
        
        # Government contracting specific entities - INITIALIZE THIS FIRST
        self.govcon_entities = {
            "CONTRACT_VEHICLE": ["IDIQ", "BPA", "GWAC", "GSA Schedule", "GWACs", "IDIQs", "BPAs", "OASIS", "Alliant", "SEWP", "CIO-SP3"],
            "NAICS_CODE": [],  # Will be populated from patterns
            "AGENCY": ["DoD", "Department of Defense", "DHS", "Department of Homeland Security", "HHS", "GSA"],
            "TECHNOLOGY": ["cloud", "cybersecurity", "artificial intelligence", "machine learning", "blockchain", 
                        "IoT", "5G", "quantum", "zero trust", "DevSecOps", "microservices"],
            "CLEARANCE": ["Top Secret", "TS/SCI", "Secret", "Confidential", "Public Trust", "TS", "SCI"],
            "REGULATION": ["FAR", "DFAR", "CMMC", "NIST", "FedRAMP", "FISMA", "HIPAA", "ATO"]
        }
        
        # Generate NAICS code patterns
        for i in range(100000, 1000000):
            if i % 100000 == 0:
                self.govcon_entities["NAICS_CODE"].append(str(i))
        
        # Load spaCy model
        try:
            self.nlp = spacy.load(model_name)
            self.logger.info(f"Loaded spaCy model: {model_name}")
        except OSError:
            self.logger.info(f"Downloading spaCy model: {model_name}")
            os.system(f"python -m spacy download {model_name}")
            self.nlp = spacy.load(model_name)
        
        # Add custom components AFTER initializing govcon_entities
        self._add_custom_components()
    
    def _add_custom_components(self):
        """Add custom NLP components to the pipeline"""
        # Add entity ruler for pattern-based entity recognition
        if "entity_ruler" not in self.nlp.pipe_names:
            ruler = self.nlp.add_pipe("entity_ruler", before="ner")
            ruler.add_patterns(self._get_entity_patterns())
        
        # Register custom component for post-processing
        @Language.component("govcon_entities")
        def govcon_entities_component(doc):
            """Custom component to identify government contracting specific entities"""
            new_ents = []
            for ent in doc.ents:
                # Extend entity spans for better entity boundaries
                new_ents.append(ent)
            
            # Look for NAICS codes with regex
            naics_pattern = re.compile(r'\b\d{6}\b')
            for match in naics_pattern.finditer(doc.text):
                start, end = match.span()
                start_char = match.start()
                end_char = match.end()
                
                # Find token span
                start_token = None
                end_token = None
                for i, token in enumerate(doc):
                    if token.idx <= start_char < token.idx + len(token.text):
                        start_token = i
                    if token.idx <= end_char <= token.idx + len(token.text) and start_token is not None:
                        end_token = i + 1
                        break
                
                if start_token is not None and end_token is not None:
                    naics_ent = Span(doc, start_token, end_token, label="NAICS_CODE")
                    new_ents.append(naics_ent)
            
            doc.ents = new_ents
            return doc
        
        # Add custom component to pipeline if not already added
        if "govcon_entities" not in self.nlp.pipe_names:
            self.nlp.add_pipe("govcon_entities", after="ner")
    
    def _get_entity_patterns(self) -> List[Dict[str, Any]]:
        """Create patterns for entity recognition"""
        patterns = []
        
        # Add patterns from govcon_entities dictionary
        for ent_type, terms in self.govcon_entities.items():
            for term in terms:
                if term:  # Skip empty strings
                    # Single token pattern
                    patterns.append({"label": ent_type, "pattern": term})
                    
                    # Also add lowercase version if capitalized
                    if term[0].isupper():
                        patterns.append({"label": ent_type, "pattern": term.lower()})
        
        # Special patterns for NAICS codes
        patterns.append({"label": "NAICS_CODE", "pattern": [{"SHAPE": "dddddd"}]})
        
        return patterns
    
    def extract_entities(self, text: str) -> Dict[str, List[Dict[str, Any]]]:
        """Extract structured entities from text"""
        doc = self.nlp(text)
        
        # Organize entities by type
        entities = {}
        for ent in doc.ents:
            ent_type = ent.label_
            
            # Skip non-relevant entity types
            if ent_type not in ["PERSON", "ORG", "GPE", "LOC", "DATE", "MONEY", "PERCENT", "CARDINAL", 
                               "CONTRACT_VEHICLE", "NAICS_CODE", "AGENCY", "TECHNOLOGY", "CLEARANCE", "REGULATION"]:
                continue
                
            if ent_type not in entities:
                entities[ent_type] = []
            
            # Check if this entity is already added (avoid duplicates)
            entity_text = ent.text.strip()
            if not any(e["text"] == entity_text for e in entities[ent_type]):
                entities[ent_type].append({
                    "text": entity_text,
                    "start_char": ent.start_char,
                    "end_char": ent.end_char,
                    "confidence": 0.9  # Would be replaced with actual confidence score in a production system
                })
        
        return entities
    
    def process_document(self, document: Dict[str, Any]) -> Dict[str, Any]:
        """Process a document and enhance with extracted entities"""
        # Combine relevant text fields
        text = " ".join([
            document.get('title', ''),
            document.get('description', ''),
            document.get('additional_info', '')
        ])
        
        # Extract entities
        extracted_entities = self.extract_entities(text)
        
        # Add to document
        document['extracted_entities'] = extracted_entities
        
        # Add entity summary
        document['entity_summary'] = self._generate_entity_summary(extracted_entities)
        
        return document
    
    def _generate_entity_summary(self, entities: Dict[str, List[Dict[str, Any]]]) -> Dict[str, Any]:
        """Generate a summary of extracted entities"""
        summary = {}
        
        # Map entities to relevant categories for business intelligence
        tech_stack = []
        if "TECHNOLOGY" in entities:
            tech_stack = [ent["text"] for ent in entities["TECHNOLOGY"]]
        
        regulatory_requirements = []
        if "REGULATION" in entities:
            regulatory_requirements = [ent["text"] for ent in entities["REGULATION"]]
        
        clearance_requirements = []
        if "CLEARANCE" in entities:
            clearance_requirements = [ent["text"] for ent in entities["CLEARANCE"]]
        
        agencies_involved = []
        if "AGENCY" in entities:
            agencies_involved = [ent["text"] for ent in entities["AGENCY"]]
        elif "ORG" in entities:
            # Try to find government agencies in ORG entities
            gov_keywords = ["department", "agency", "administration", "bureau", "office"]
            agencies_involved = [
                ent["text"] for ent in entities["ORG"] 
                if any(keyword in ent["text"].lower() for keyword in gov_keywords)
            ]
        
        # Build summary
        summary = {
            "tech_stack": tech_stack,
            "regulatory_requirements": regulatory_requirements,
            "clearance_requirements": clearance_requirements,
            "agencies_involved": agencies_involved
        }
        
        return summary