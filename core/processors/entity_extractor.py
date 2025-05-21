# core/processors/entity_extractor.py
import spacy
from spacy.tokens import Span, DocBin
from spacy.training import Example
from spacy.language import Language
from spacy.util import minibatch, compounding
import logging
from typing import Dict, List, Any, Set, Tuple, Optional, Union, Iterator
import os
import re
import random
import json
from pathlib import Path
import numpy as np
from tqdm import tqdm

class EntityExtractor:
    """Extract entities from government contract descriptions using NLP"""
    
    def __init__(self, model_name: str = "en_core_web_lg", custom_model_path: str = None):
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
        
        # Check if custom model exists and load if specified
        if custom_model_path and os.path.exists(custom_model_path):
            try:
                self.logger.info(f"Loading custom NER model from {custom_model_path}")
                self.nlp = spacy.load(custom_model_path)
            except Exception as e:
                self.logger.error(f"Failed to load custom model: {str(e)}. Falling back to standard model.")
                self._load_standard_model(model_name)
        else:
            self._load_standard_model(model_name)
        
        # Add custom components AFTER initializing govcon_entities
        self._add_custom_components()
    
    def _load_standard_model(self, model_name: str):
        """Load standard spaCy model"""
        try:
            self.nlp = spacy.load(model_name)
            self.logger.info(f"Loaded spaCy model: {model_name}")
        except OSError:
            self.logger.info(f"Downloading spaCy model: {model_name}")
            os.system(f"python -m spacy download {model_name}")
            self.nlp = spacy.load(model_name)
    
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
            # Store existing entities
            original_ents = list(doc.ents)
            
            # Track entity spans to detect overlaps
            entity_spans = [(ent.start, ent.end) for ent in original_ents]
            
            # Collect new entities
            new_ents = []
            
            # Look for NAICS codes with regex
            naics_pattern = re.compile(r'\b\d{6}\b')
            for match in naics_pattern.finditer(doc.text):
                start_char = match.start()
                end_char = match.end()
                
                # Find token span for character offsets
                start_token = None
                end_token = None
                
                for i, token in enumerate(doc):
                    if token.idx <= start_char < token.idx + len(token.text):
                        start_token = i
                    if token.idx <= end_char <= token.idx + len(token.text) and start_token is not None:
                        end_token = i + 1
                        break
                
                if start_token is not None and end_token is not None:
                    # Check for overlaps with existing entities
                    if not any(start_token < e[1] and end_token > e[0] for e in entity_spans):
                        naics_ent = Span(doc, start_token, end_token, label="NAICS_CODE")
                        new_ents.append(naics_ent)
                        # Add this span to our tracking list
                        entity_spans.append((start_token, end_token))
            
            # Combine original entities with new entities that don't overlap
            doc.ents = original_ents + new_ents
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
                    "confidence": self._get_entity_confidence(ent)
                })
        
        return entities
    
    def _get_entity_confidence(self, ent: Span) -> float:
        """Calculate confidence score for an entity.
        
        In a production model with a properly trained NER component,
        we would extract the actual confidence. For now, we use a 
        simple heuristic based on entity type.
        
        Args:
            ent: The entity span
            
        Returns:
            Confidence score between 0.0 and 1.0
        """
        # Custom entity types get higher base confidence
        if ent.label_ in self.govcon_entities:
            base_confidence = 0.9
        else:
            base_confidence = 0.75
        
        # Minor adjustment for entity length (longer entities are slightly less confident)
        length_factor = max(0.9, 1.0 - 0.01 * (len(ent.text.split()) - 1))
        
        return base_confidence * length_factor
    
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
    
    # CUSTOM MODEL TRAINING IMPLEMENTATION
    
    def train_custom_model(
        self, 
        training_data: List[Dict[str, Any]],
        output_dir: str,
        n_iter: int = 25,
        batch_size: int = 16,
        dropout: float = 0.2,
        eval_split: float = 0.2,
        base_model: str = None
    ) -> Dict[str, Any]:
        """Train a custom NER model for government contracting.
        
        Args:
            training_data: List of training examples, each with 'text' and 'entities' fields
            output_dir: Directory to save the trained model
            n_iter: Number of training iterations
            batch_size: Batch size for training
            dropout: Dropout rate for training
            eval_split: Fraction of data to use for evaluation
            base_model: Base model to start from. If None, uses the current model.
            
        Returns:
            Dictionary with training statistics
            
        Example training_data format:
            [
                {
                    "text": "The Department of Defense requires cloud services with NIST compliance.",
                    "entities": [
                        {"start": 4, "end": 26, "label": "AGENCY"},
                        {"start": 36, "end": 41, "label": "TECHNOLOGY"},
                        {"start": 53, "end": 57, "label": "REGULATION"}
                    ]
                },
                ...
            ]
        """
        self.logger.info(f"Starting custom NER model training with {len(training_data)} examples")
        
        # Validate training data format
        self._validate_training_data(training_data)
        
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # Initialize new model or start with existing
        if base_model:
            try:
                nlp = spacy.load(base_model)
                self.logger.info(f"Using base model: {base_model}")
            except:
                self.logger.warning(f"Could not load base model {base_model}, creating new model")
                nlp = spacy.blank("en")
        else:
            # Use current model as base
            nlp = spacy.load(self.nlp.meta["name"])
            self.logger.info(f"Using current model as base: {self.nlp.meta['name']}")
        
        # Prepare pipeline
        if "ner" not in nlp.pipe_names:
            ner = nlp.add_pipe("ner", last=True)
        else:
            ner = nlp.get_pipe("ner")
        
        # Add entity labels from training data
        for example in training_data:
            for entity in example.get("entities", []):
                ner.add_label(entity["label"])
        
        # Split training data into train/eval sets
        random.shuffle(training_data)
        split_point = int(len(training_data) * (1 - eval_split))
        train_data = training_data[:split_point]
        eval_data = training_data[split_point:]
        
        self.logger.info(f"Training on {len(train_data)} examples, evaluating on {len(eval_data)} examples")
        
        # Convert to spaCy format
        train_examples = self._create_examples(nlp, train_data)
        eval_examples = self._create_examples(nlp, eval_data)
        
        # Get names of other pipes to disable during training
        pipe_exceptions = ["ner", "trf_wordpiecer", "trf_tok2vec"]
        other_pipes = [pipe for pipe in nlp.pipe_names if pipe not in pipe_exceptions]
        
        # Train the model
        with nlp.disable_pipes(*other_pipes):
            # Set up the optimizer
            optimizer = nlp.create_optimizer()
            
            # Begin training
            self.logger.info("Beginning training...")
            batch_sizes = compounding(4.0, batch_size, 1.001)
            
            # Track metrics
            train_losses = []
            eval_metrics = []
            
            # Training loop
            for i in range(n_iter):
                # Shuffle training data
                random.shuffle(train_examples)
                losses = {}
                
                # Batch training
                batches = minibatch(train_examples, size=batch_sizes)
                for batch in tqdm(batches, desc=f"Iteration {i+1}/{n_iter}"):
                    nlp.update(
                        batch,
                        drop=dropout,
                        losses=losses,
                        sgd=optimizer
                    )
                
                # Track training loss
                iteration_loss = losses.get("ner", 0)
                train_losses.append(iteration_loss)
                
                # Evaluate on validation set every few iterations
                if (i + 1) % 5 == 0 or i == n_iter - 1:
                    eval_result = self._evaluate_model(nlp, eval_examples)
                    eval_metrics.append(eval_result)
                    
                    self.logger.info(
                        f"Iteration {i+1}: Loss: {iteration_loss:.4f}, "
                        f"Precision: {eval_result['precision']:.4f}, "
                        f"Recall: {eval_result['recall']:.4f}, "
                        f"F1: {eval_result['f1']:.4f}"
                    )
        
        # Save the trained model
        model_path = os.path.join(output_dir, "model")
        nlp.to_disk(model_path)
        self.logger.info(f"Model saved to {model_path}")
        
        # Save training metrics
        metrics_path = os.path.join(output_dir, "training_metrics.json")
        metrics = {
            "train_losses": train_losses,
            "eval_metrics": eval_metrics,
            "final_metrics": eval_metrics[-1] if eval_metrics else None,
            "training_config": {
                "n_iter": n_iter,
                "batch_size": batch_size,
                "dropout": dropout,
                "eval_split": eval_split,
                "base_model": base_model or self.nlp.meta["name"],
                "training_examples": len(train_data),
                "evaluation_examples": len(eval_data)
            }
        }
        
        with open(metrics_path, "w") as f:
            json.dump(metrics, f, indent=2)
        
        self.logger.info(f"Training metrics saved to {metrics_path}")
        
        # Update the current model
        self.nlp = nlp
        self.logger.info("Updated current model to trained model")
        
        return metrics
    
    def _validate_training_data(self, training_data: List[Dict[str, Any]]):
        """Validate the format of training data.
        
        Args:
            training_data: List of training examples
            
        Raises:
            ValueError: If training data format is invalid
        """
        if not training_data:
            raise ValueError("Training data is empty")
        
        for i, example in enumerate(training_data):
            if "text" not in example:
                raise ValueError(f"Example {i} missing 'text' field")
            
            if "entities" not in example:
                raise ValueError(f"Example {i} missing 'entities' field")
            
            if not isinstance(example["entities"], list):
                raise ValueError(f"Example {i} 'entities' must be a list")
            
            text = example["text"]
            
            for j, entity in enumerate(example["entities"]):
                if "start" not in entity or "end" not in entity or "label" not in entity:
                    raise ValueError(f"Entity {j} in example {i} missing required fields")
                
                start, end = entity["start"], entity["end"]
                
                if not (0 <= start < end <= len(text)):
                    raise ValueError(
                        f"Entity {j} in example {i} has invalid character spans: "
                        f"start={start}, end={end}, text length={len(text)}"
                    )
    
    def _create_examples(self, nlp: Language, data: List[Dict[str, Any]]) -> List[Example]:
        """Convert training data to spaCy Example objects.
        
        Args:
            nlp: spaCy Language object
            data: List of training examples
            
        Returns:
            List of spaCy Example objects
        """
        examples = []
        
        for example in data:
            text = example["text"]
            entities = example.get("entities", [])
            
            # Create doc
            doc = nlp.make_doc(text)
            
            # Sort entities by start position and then by length (longer entities first)
            # This prioritizes longer entities when there are overlaps
            sorted_entities = sorted(entities, key=lambda e: (e["start"], -1 * (e["end"] - e["start"])))
            
            # Track which tokens have been assigned to entities
            token_to_entity = {}
            ents = []
            
            for entity in sorted_entities:
                start_char = entity["start"]
                end_char = entity["end"]
                label = entity["label"]
                
                # Find token span for character offsets
                start_token = None
                end_token = None
                
                for i, token in enumerate(doc):
                    if token.idx <= start_char < token.idx + len(token.text) and start_token is None:
                        start_token = i
                    if token.idx <= end_char <= token.idx + len(token.text) and start_token is not None:
                        end_token = i + 1
                        break
                
                if start_token is not None and end_token is not None:
                    # Check if any token in this span is already part of another entity
                    overlap = False
                    for i in range(start_token, end_token):
                        if i in token_to_entity:
                            overlap = True
                            break
                    
                    # Skip this entity if there's an overlap
                    if overlap:
                        continue
                    
                    # Mark these tokens as used
                    for i in range(start_token, end_token):
                        token_to_entity[i] = label
                    
                    # Create entity span
                    span = Span(doc, start_token, end_token, label=label)
                    ents.append(span)
            
            # Set the entities on the doc
            if ents:
                doc.ents = ents
            
            # Create training example
            reference = doc.copy()
            reference.ents = ents
            examples.append(Example(doc, reference))
        
        return examples
    
    def _evaluate_model(self, nlp: Language, examples: List[Example]) -> Dict[str, float]:
        """Evaluate NER model performance on evaluation examples.
        
        Args:
            nlp: spaCy Language object
            examples: List of spaCy Example objects
            
        Returns:
            Dictionary with precision, recall, and F1 score
        """
        tp = 0  # True positives
        fp = 0  # False positives
        fn = 0  # False negatives
        
        for example in examples:
            # Get gold entities
            gold_entities = [(e.start, e.end, e.label_) for e in example.reference.ents]
            
            # Get predicted entities
            pred_doc = nlp(example.reference.text)
            pred_entities = [(e.start, e.end, e.label_) for e in pred_doc.ents]
            
            # Count true positives, false positives, and false negatives
            for entity in pred_entities:
                if entity in gold_entities:
                    tp += 1
                else:
                    fp += 1
            
            for entity in gold_entities:
                if entity not in pred_entities:
                    fn += 1
        
        # Calculate precision, recall, and F1
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        
        return {"precision": precision, "recall": recall, "f1": f1}
    
    def augment_training_data(
        self, 
        training_data: List[Dict[str, Any]],
        augmentation_factor: int = 2,
        synonym_replacement: bool = True,
        word_insertion: bool = False,  # Disabled by default to avoid entity overlap issues
        word_deletion: bool = False    # Disabled by default to avoid entity overlap issues
    ) -> List[Dict[str, Any]]:
        """Augment training data with various techniques.
        
        Args:
            training_data: Original training data
            augmentation_factor: Number of augmented examples to generate per original
            synonym_replacement: Whether to use synonym replacement
            word_insertion: Whether to use random word insertion (disabled by default)
            word_deletion: Whether to use random word deletion (disabled by default)
            
        Returns:
            Augmented training data
        """
        augmented_data = list(training_data)  # Start with original data
        
        # Load WordNet for synonyms if synonym replacement is enabled
        if synonym_replacement:
            try:
                from nltk.corpus import wordnet
                import nltk
                nltk.download('wordnet', quiet=True)
            except ImportError:
                self.logger.warning("NLTK WordNet not available, disabling synonym replacement")
                synonym_replacement = False
        
        # Process each training example
        for example in training_data:
            text = example["text"]
            entities = example["entities"]
            
            # Skip examples with no entities
            if not entities:
                continue
            
            # Sort entities by start position to handle overlaps correctly
            sorted_entities = sorted(entities, key=lambda e: e["start"])
            
            # Create entity spans
            entity_spans = [(e["start"], e["end"]) for e in sorted_entities]
            
            # Generate augmented examples
            for _ in range(augmentation_factor):
                augmented_text = text
                adjusted_entities = []
                
                # Track character offsets for entity adjustment
                offset = 0
                
                # Process the text directly instead of using spaCy tokenization
                # This avoids potential overlapping entity issues
                words = text.split()
                
                # Create a mask of word positions that overlap with entities
                entity_words = set()
                current_pos = 0
                
                for i, word in enumerate(words):
                    word_start = current_pos
                    word_end = word_start + len(word)
                    
                    # Check if this word overlaps with any entity
                    for start, end in entity_spans:
                        if max(word_start, start) < min(word_end, end):
                            entity_words.add(i)
                            break
                    
                    current_pos = word_end + 1  # +1 for the space
                
                # Copy entities to adjusted_entities with initial offsets of 0
                for entity in sorted_entities:
                    adjusted_entities.append(entity.copy())
                
                # Perform synonym replacement
                if synonym_replacement and random.random() < 0.7:
                    modified_words = list(words)
                    
                    for i, word in enumerate(words):
                        # Skip entity words, stopwords, short words
                        if (i in entity_words or len(word) < 4 or 
                            word.lower() in ['the', 'and', 'for', 'with', 'this', 'that']):
                            continue
                        
                        # Random chance to replace with synonym
                        if random.random() < 0.2:
                            synonyms = []
                            
                            if synonym_replacement:
                                # Get synonyms from WordNet
                                for syn in wordnet.synsets(word.lower()):
                                    for lemma in syn.lemmas():
                                        synonym = lemma.name().replace('_', ' ')
                                        if synonym != word.lower() and len(synonym) > 2:
                                            synonyms.append(synonym)
                            
                            if synonyms:
                                # Choose a random synonym
                                synonym = random.choice(synonyms)
                                
                                # Replace word
                                modified_words[i] = synonym
                    
                    # Reconstruct the text with synonyms
                    # We'll rebuild character by character to correctly track offsets
                    new_text = ""
                    for i, word in enumerate(modified_words):
                        if i > 0:
                            new_text += " "
                        
                        old_word = words[i]
                        new_word = word
                        
                        # Calculate the difference in length
                        delta = len(new_word) - len(old_word)
                        
                        # Add the word to the new text
                        new_text += new_word
                        
                        # Adjust the entities that come after this word
                        if delta != 0:
                            # Calculate the start position of this word
                            word_pos = len(" ".join(words[:i]))
                            if i > 0:
                                word_pos += 1  # Add space
                            
                            # Word end position
                            word_end = word_pos + len(old_word)
                            
                            # Update all entities that start after this word
                            for j, entity in enumerate(adjusted_entities):
                                if entity["start"] > word_end:
                                    adjusted_entities[j]["start"] += delta
                                    adjusted_entities[j]["end"] += delta
                    
                    # Use the modified text
                    augmented_text = new_text
                
                # Add augmented example if it's different from the original
                if augmented_text != text:
                    augmented_data.append({
                        "text": augmented_text,
                        "entities": adjusted_entities
                    })
        
        self.logger.info(f"Augmented {len(training_data)} examples to {len(augmented_data)} examples")
        return augmented_data
    
    def save_model(self, output_dir: str):
        """Save the current model to disk.
        
        Args:
            output_dir: Directory to save the model
        """
        os.makedirs(output_dir, exist_ok=True)
        self.nlp.to_disk(output_dir)
        self.logger.info(f"Model saved to {output_dir}")
    
    def load_model(self, model_path: str):
        """Load a model from disk.
        
        Args:
            model_path: Path to the saved model
            
        Returns:
            True if model loaded successfully, False otherwise
        """
        try:
            self.nlp = spacy.load(model_path)
            self.logger.info(f"Loaded model from {model_path}")
            
            # Re-add custom components
            self._add_custom_components()
            
            return True
        except Exception as e:
            self.logger.error(f"Failed to load model from {model_path}: {str(e)}")
            return False
    
    def export_training_data(self, documents: List[Dict[str, Any]], output_file: str):
        """Export training data from processed documents.
        
        This is useful for creating a training dataset from documents
        that have already been processed with pattern matching or
        rule-based entity extraction.
        
        Args:
            documents: List of processed documents with extracted entities
            output_file: Path to save the training data
        """
        training_data = []
        
        for doc in documents:
            # Skip documents without extracted entities
            if "extracted_entities" not in doc:
                continue
            
            # Get text from document
            text = " ".join([
                doc.get('title', ''),
                doc.get('description', ''),
                doc.get('additional_info', '')
            ])
            
            # Skip empty text
            if not text.strip():
                continue
            
            # Extract entities in the format needed for training
            entities = []
            
            for entity_type, entity_list in doc["extracted_entities"].items():
                for entity in entity_list:
                    # Check for required fields
                    if "text" not in entity or "start_char" not in entity or "end_char" not in entity:
                        continue
                    
                    # Add entity to training data
                    entities.append({
                        "start": entity["start_char"],
                        "end": entity["end_char"],
                        "label": entity_type
                    })
            
            # Add example to training data
            if entities:
                training_data.append({
                    "text": text,
                    "entities": entities
                })
        
        # Save training data to file
        with open(output_file, "w") as f:
            json.dump(training_data, f, indent=2)
        
        self.logger.info(f"Exported {len(training_data)} training examples to {output_file}")
        
        return len(training_data)
    
    def generate_training_data_from_text(
        self, 
        texts: List[str],
        output_file: str = None,
        min_confidence: float = 0.8
    ) -> List[Dict[str, Any]]:
        """Generate training data from a list of texts.
        
        Uses the current model to extract entities and convert
        them to training data format.
        
        Args:
            texts: List of text strings
            output_file: Optional path to save the training data
            min_confidence: Minimum confidence threshold for entities
            
        Returns:
            List of training examples
        """
        training_data = []
        
        for text in texts:
            # Extract entities
            doc = self.nlp(text)
            
            # Convert to training data format
            entities = []
            
            for ent in doc.ents:
                confidence = self._get_entity_confidence(ent)
                
                # Only include entities with high confidence
                if confidence >= min_confidence:
                    entities.append({
                        "start": ent.start_char,
                        "end": ent.end_char,
                        "label": ent.label_
                    })
            
            # Add example to training data
            if entities:
                training_data.append({
                    "text": text,
                    "entities": entities
                })
        
        # Save training data to file if specified
        if output_file:
            with open(output_file, "w") as f:
                json.dump(training_data, f, indent=2)
            
            self.logger.info(f"Generated {len(training_data)} training examples and saved to {output_file}")
        else:
            self.logger.info(f"Generated {len(training_data)} training examples")
        
        return training_data
    
    def create_training_doc_bin(self, training_data: List[Dict[str, Any]], output_file: str):
        """Create a spaCy DocBin file from training data.
        
        This is useful for training with spaCy's CLI.
        
        Args:
            training_data: Training data
            output_file: Path to save the DocBin file
        """
        nlp = spacy.blank("en")
        db = DocBin()
        
        for example in training_data:
            text = example["text"]
            entities = example.get("entities", [])
            
            # Create doc
            doc = nlp.make_doc(text)
            ents = []
            
            # Add entities
            for entity in entities:
                start_char = entity["start"]
                end_char = entity["end"]
                label = entity["label"]
                
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
                    span = Span(doc, start_token, end_token, label=label)
                    ents.append(span)
            
            # Set entities
            doc.ents = ents
            
            # Add to DocBin
            db.add(doc)
        
        # Save DocBin
        db.to_disk(output_file)
        self.logger.info(f"Created DocBin with {len(training_data)} documents and saved to {output_file}")