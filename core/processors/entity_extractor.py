import spacy
from spacy.tokens import Span, Doc
from spacy.language import Language
import logging
from typing import Dict, List, Any, Set, Tuple, Optional
import os
import re

class EntityExtractor:
    """Extract entities from government contract descriptions using NLP"""
    
    def __init__(self, model_name: str = "en_core_web_lg"):
        self.logger = logging.getLogger("entity_extractor")
        
        # Government contracting specific entities - INITIALIZE THIS FIRST
        self.govcon_entities = {
            "CONTRACT_VEHICLE": ["IDIQ", "BPA", "GWAC", "GSA Schedule", "GWACs", "IDIQs", "BPAs", "OASIS", "Alliant", 
                                "SEWP", "CIO-SP3", "STARS III", "8(a) STARS III", "VETS 2", "ASTRO", "POLARIS", "GSA MAS"],
            "NAICS_CODE": ["541511", "541512", "541513", "541519", "518210", "541330", "541715", 
                          "334220", "334290", "517311", "517312", "517410", "541714", "541713"],
            "AGENCY": ["DoD", "Department of Defense", "DHS", "Department of Homeland Security", "HHS", "GSA", 
                       "Department of Health and Human Services", "Department of Veterans Affairs", "VA", 
                       "Department of Justice", "DOJ", "Department of State", "DOS", "Department of Energy", "DOE",
                       "Department of Labor", "DOL", "Department of Treasury", "Department of Commerce", "DOC",
                       "Department of Transportation", "DOT", "Department of Agriculture", "USDA",
                       "NASA", "EPA", "IRS", "FBI", "CIA", "DARPA", "DISA", "USCIS", "FAA", "FEMA", "SEC"],
            "TECHNOLOGY": ["cloud", "cybersecurity", "artificial intelligence", "machine learning", "blockchain", 
                          "IoT", "5G", "quantum", "zero trust", "DevSecOps", "microservices", "containerization",
                          "kubernetes", "serverless", "data analytics", "big data", "edge computing", "digital twin",
                          "RPA", "robotic process automation", "low-code", "no-code", "virtual reality", "augmented reality",
                          "mixed reality", "biometrics", "cloud native", "API", "APIs", "SaaS", "PaaS", "IaaS", "FedRAMP",
                          "AWS", "Azure", "GCP", "Google Cloud", "Amazon Web Services", "Microsoft Azure"],
            "CLEARANCE": ["Top Secret", "TS/SCI", "Secret", "Confidential", "Public Trust", "TS", "SCI", 
                          "Q Clearance", "L Clearance"],
            "REGULATION": ["FAR", "DFAR", "CMMC", "NIST", "FedRAMP", "FISMA", "HIPAA", "ATO", "FIPS", 
                          "NIST 800-53", "NIST 800-171", "NIST 800-53r4", "NIST 800-53r5", "NIST 800-171r2", 
                          "CMMC 2.0", "CMMC Level 1", "CMMC Level 2", "CMMC Level 3", "FIPS 140-2", "FIPS 140-3", 
                          "FIPS 199", "FIPS 200", "Section 508", "GDPR", "CCPA", "CPRA"],
            "CONTRACT_TYPE": ["FFP", "T&M", "Firm Fixed Price", "Time and Materials", "Cost Plus", "CPFF", "CPIF", 
                             "Cost Plus Fixed Fee", "Cost Plus Incentive Fee", "IDIQ", "BPA", "BOA", "GWAC"]
        }
        
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
        
        # Keywords for sentiment analysis
        self.positive_words = [
            "excellent", "outstanding", "innovative", "successful", "efficient",
            "effective", "improved", "enhance", "benefit", "advantage", "superior",
            "best", "leading", "advanced", "cutting-edge", "state-of-the-art",
            "streamlined", "optimized", "collaborative", "partnership", "secure",
            "reliable", "robust", "seamless", "high-quality", "cost-effective",
            "savings", "increase", "improve", "modernize", "transform"
        ]
        
        self.negative_words = [
            "problem", "issue", "concern", "risk", "challenge", "difficult",
            "complex", "failure", "failed", "poor", "inadequate", "deficient",
            "outdated", "obsolete", "vulnerability", "breach", "delay", "over-budget",
            "cost overrun", "behind schedule", "insecure", "unreliable", "unstable",
            "inefficient", "expensive", "critical", "severe", "threat", "weakness"
        ]
        
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
            
            # Filter out overlapping entities by keeping the longest one
            filtered_ents = []
            sorted_ents = sorted(new_ents, key=lambda e: e.end - e.start, reverse=True)
            
            # Keep track of token indices that are already part of an entity
            covered_tokens = set()
            for ent in sorted_ents:
                token_indices = set(range(ent.start, ent.end))
                if not any(idx in covered_tokens for idx in token_indices):
                    filtered_ents.append(ent)
                    covered_tokens.update(token_indices)
            
            # Sort entities by their position in the text
            doc.ents = sorted(filtered_ents, key=lambda e: e.start)
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
        
        # Add special patterns
        # NAICS code pattern
        patterns.append({"label": "NAICS_CODE", "pattern": [{"SHAPE": "dddddd"}]})
        
        # Contract number patterns - e.g., "W91QF5-09-D-0022"
        patterns.append({"label": "CONTRACT_NUMBER", "pattern": [
            {"SHAPE": "ddddd?d?-dd-d-dddd"}
        ]})
        
        # Contract number patterns - e.g., "N00178-15-D-8044"
        patterns.append({"label": "CONTRACT_NUMBER", "pattern": [
            {"SHAPE": "d?d?@@@dd-dd-d-dddd"}
        ]})
        
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
                               "CONTRACT_VEHICLE", "NAICS_CODE", "AGENCY", "TECHNOLOGY", "CLEARANCE", 
                               "REGULATION", "CONTRACT_TYPE", "CONTRACT_NUMBER"]:
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
                    "confidence": 0.9  # Placeholder confidence score
                })
        
        return entities
    
    def extract_relationships(self, doc: Doc, entities: Dict[str, List[Dict[str, Any]]]) -> List[Dict[str, Any]]:
        """Extract relationships between entities using dependency parsing"""
        relationships = []
        
        # Get entities by their character spans for easy lookup
        entity_spans = {}
        for entity_type, entity_list in entities.items():
            for entity in entity_list:
                span = (entity["start_char"], entity["end_char"])
                entity_spans[span] = {
                    "text": entity["text"],
                    "type": entity_type
                }
        
        # Analyze sentences
        for sent in doc.sents:
            # Look for verb dependency patterns between entities
            for token in sent:
                if token.dep_ == "ROOT" and token.pos_ == "VERB":
                    # Find subject and object connected to this verb
                    subject = None
                    obj = None
                    
                    # Find subject
                    for child in token.children:
                        if child.dep_ in ["nsubj", "nsubjpass"]:
                            subject_span = self._find_entity_containing_token(child, entity_spans)
                            if subject_span:
                                subject = entity_spans[subject_span]
                    
                    # Find object
                    for child in token.children:
                        if child.dep_ in ["dobj", "pobj"]:
                            object_span = self._find_entity_containing_token(child, entity_spans)
                            if object_span:
                                obj = entity_spans[object_span]
                    
                    # If we found both subject and object, create relationship
                    if subject and obj:
                        relationship = {
                            "source": subject["text"],
                            "source_type": subject["type"],
                            "target": obj["text"],
                            "target_type": obj["type"],
                            "relation": token.lemma_.upper(),
                            "confidence": 0.8
                        }
                        relationships.append(relationship)
        
        # Add known entity type relationships
        relationships.extend(self._extract_type_based_relationships(entities))
        
        return relationships
    
    def _find_entity_containing_token(self, token, entity_spans):
        """Find if a token is within any entity span"""
        token_start = token.idx
        token_end = token.idx + len(token.text)
        
        for (start, end), entity in entity_spans.items():
            if start <= token_start and token_end <= end:
                return (start, end)
        
        return None
    
    def _extract_type_based_relationships(self, entities: Dict[str, List[Dict[str, Any]]]) -> List[Dict[str, Any]]:
        """Extract relationships based on entity types"""
        relationships = []
        
        # Agency-Technology relationships
        if "AGENCY" in entities and "TECHNOLOGY" in entities:
            for agency in entities["AGENCY"]:
                for tech in entities["TECHNOLOGY"]:
                    relationships.append({
                        "source": agency["text"],
                        "source_type": "AGENCY",
                        "target": tech["text"],
                        "target_type": "TECHNOLOGY",
                        "relation": "INTERESTED_IN",
                        "confidence": 0.7
                    })
        
        # Agency-Regulation relationships
        if "AGENCY" in entities and "REGULATION" in entities:
            for agency in entities["AGENCY"]:
                for reg in entities["REGULATION"]:
                    relationships.append({
                        "source": agency["text"],
                        "source_type": "AGENCY",
                        "target": reg["text"],
                        "target_type": "REGULATION",
                        "relation": "COMPLIES_WITH",
                        "confidence": 0.8
                    })
        
        # Clearance requirements
        if "AGENCY" in entities and "CLEARANCE" in entities:
            for agency in entities["AGENCY"]:
                for clearance in entities["CLEARANCE"]:
                    relationships.append({
                        "source": agency["text"],
                        "source_type": "AGENCY",
                        "target": clearance["text"],
                        "target_type": "CLEARANCE",
                        "relation": "REQUIRES",
                        "confidence": 0.9
                    })
        
        # Organization-Contract vehicle relationships
        if "ORG" in entities and "CONTRACT_VEHICLE" in entities:
            for org in entities["ORG"]:
                for vehicle in entities["CONTRACT_VEHICLE"]:
                    relationships.append({
                        "source": org["text"],
                        "source_type": "ORG",
                        "target": vehicle["text"],
                        "target_type": "CONTRACT_VEHICLE",
                        "relation": "USES",
                        "confidence": 0.6
                    })
        
        # Organization-NAICS code relationships
        if "ORG" in entities and "NAICS_CODE" in entities:
            for org in entities["ORG"]:
                for naics in entities["NAICS_CODE"]:
                    relationships.append({
                        "source": org["text"],
                        "source_type": "ORG",
                        "target": naics["text"],
                        "target_type": "NAICS_CODE",
                        "relation": "OPERATES_IN",
                        "confidence": 0.7
                    })
        
        return relationships
    
    def analyze_sentiment(self, text: str) -> Dict[str, Any]:
        """Analyze sentiment in text"""
        text = text.lower()
        
        try:
            # Try using spaCy's built-in sentiment analysis if the model supports it
            doc = self.nlp(text)
            if hasattr(doc, "sentiment"):
                # spaCy's sentiment is between -1 and 1
                sentiment_score = doc.sentiment
                if sentiment_score > 0.1:
                    sentiment = "positive"
                    score = 0.5 + (sentiment_score / 2)
                elif sentiment_score < -0.1:
                    sentiment = "negative"
                    score = 0.5 - (abs(sentiment_score) / 2)
                else:
                    sentiment = "neutral"
                    score = 0.5
                
                return {
                    "sentiment": sentiment,
                    "score": score,
                    "method": "spacy_native"
                }
        except Exception:
            # Fall back to keyword-based approach if spaCy's sentiment fails
            pass
        
        # Keyword-based approach
        # Count positive and negative words
        pos_count = sum(1 for word in self.positive_words if word in text)
        neg_count = sum(1 for word in self.negative_words if word in text)
        
        # Calculate sentiment
        if pos_count > neg_count:
            sentiment = "positive"
            score = min(0.9, 0.5 + (pos_count - neg_count) / 10)
        elif neg_count > pos_count:
            sentiment = "negative"
            score = min(0.9, 0.5 + (neg_count - pos_count) / 10)
        else:
            sentiment = "neutral"
            score = 0.5
        
        return {
            "sentiment": sentiment,
            "score": score,
            "positive_count": pos_count,
            "negative_count": neg_count,
            "method": "keyword_based"
        }
    
    def process_document(self, document: Dict[str, Any]) -> Dict[str, Any]:
        """Process a document and enhance with extracted entities, relationships, and sentiment"""
        # Combine relevant text fields
        text = " ".join([
            document.get('title', ''),
            document.get('description', ''),
            document.get('additional_info', '')
        ])
        
        # Process with spaCy
        doc = self.nlp(text)
        
        # Extract entities
        extracted_entities = self.extract_entities(text)
        
        # Extract relationships
        entity_relationships = self.extract_relationships(doc, extracted_entities)
        
        # Add sentiment analysis
        sentiment = self.analyze_sentiment(text)
        
        # Add to document
        document['extracted_entities'] = extracted_entities
        document['entity_relationships'] = entity_relationships
        document['sentiment_analysis'] = sentiment
        
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
        
        # Extract contract vehicles
        contract_vehicles = []
        if "CONTRACT_VEHICLE" in entities:
            contract_vehicles = [ent["text"] for ent in entities["CONTRACT_VEHICLE"]]
        
        # Extract contract types
        contract_types = []
        if "CONTRACT_TYPE" in entities:
            contract_types = [ent["text"] for ent in entities["CONTRACT_TYPE"]]
        
        # Extract NAICS codes
        naics_codes = []
        if "NAICS_CODE" in entities:
            naics_codes = [ent["text"] for ent in entities["NAICS_CODE"]]
        
        # Build summary
        summary = {
            "tech_stack": tech_stack,
            "regulatory_requirements": regulatory_requirements,
            "clearance_requirements": clearance_requirements,
            "agencies_involved": agencies_involved,
            "contract_vehicles": contract_vehicles,
            "contract_types": contract_types,
            "naics_codes": naics_codes
        }
        
        return summary
    
    def batch_process_documents(self, documents: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Process a batch of documents in parallel"""
        processed_data = []
        for doc in documents:
            processed_data.append(self.process_document(doc))
        return processed_data


# Simple test if run directly
if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(level=logging.INFO)
    
    # Force flush on every print
    import sys
    print("Starting entity extractor test...", flush=True)
    
    try:
        # Initialize extractor
        extractor = EntityExtractor()
        print("Entity extractor initialized", flush=True)
        
        # Test text
        text = "The Department of Defense awarded a $10M cybersecurity contract to TechDefense Solutions for CMMC compliance."
        print(f"Test text: {text}", flush=True)
        
        # Process with spaCy
        print("Processing text with spaCy...", flush=True)
        doc = extractor.nlp(text)
        print("Text processed successfully", flush=True)
        
        # Print entities
        print("\nEntities found:", flush=True)
        if len(doc.ents) == 0:
            print("No entities found", flush=True)
        else:
            for ent in doc.ents:
                print(f"  {ent.text} ({ent.label_})", flush=True)
        
        # Extract relationships
        print("\nExtracting relationships...", flush=True)
        entities = extractor.extract_entities(text)
        relationships = extractor.extract_relationships(doc, entities)
        
        # Print relationships
        print("\nRelationships found:", flush=True)
        if len(relationships) == 0:
            print("No relationships found", flush=True)
        else:
            for rel in relationships:
                print(f"  {rel['source']} ({rel['source_type']}) -{rel['relation']}-> {rel['target']} ({rel['target_type']})", flush=True)
        
        # Sentiment analysis
        print("\nAnalyzing sentiment...", flush=True)
        sentiment = extractor.analyze_sentiment(text)
        print(f"Sentiment: {sentiment['sentiment']} (score: {sentiment['score']:.2f})", flush=True)
        
        print("\nspaCy is working correctly!", flush=True)
    except Exception as e:
        print(f"Error: {str(e)}", flush=True)
        import traceback
        traceback.print_exc()