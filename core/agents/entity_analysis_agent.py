# core/agents/entity_analysis_agent.py
from typing import Dict, List, Any, Optional
import logging
import asyncio
import uuid

from core.agents.base_agent import BaseAgent
from core.processors.entity_extractor import EntityExtractor

class EntityAnalysisAgent(BaseAgent):
    """Agent for analyzing entities in contract data."""
    
    def __init__(self, agent_id: str = None, name: str = None, 
                 model_name: str = "en_core_web_lg", config: Dict[str, Any] = None):
        """Initialize a new entity analysis agent.
        
        Args:
            agent_id: Unique identifier for the agent
            name: Human-readable name for the agent
            model_name: Name of the spaCy model to use
            config: Additional configuration for the agent
        """
        agent_config = config or {}
        agent_config["model_name"] = model_name
        
        super().__init__(agent_id, name or "entity_analysis_agent", agent_config)
        
        self.entity_extractor = EntityExtractor(model_name=model_name)
        self.logger.info(f"Initialized entity analysis agent with model {model_name}")
    
    async def execute_task(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Execute entity analysis task.
        
        Args:
            params: Parameters for the analysis task
                - documents: List of documents to analyze
                - text: Text to analyze (alternative to documents)
                - analysis_type: Type of analysis to perform (default: 'full')
                
        Returns:
            Results of the analysis task
        """
        documents = params.get("documents", [])
        text = params.get("text")
        analysis_type = params.get("analysis_type", "full")
        
        if not documents and not text:
            raise ValueError("No documents or text provided for analysis")
        
        self.logger.info(f"Starting entity analysis of {len(documents)} documents")
        
        results = {
            "document_count": len(documents),
            "processed_documents": [],
            "entity_statistics": {},
            "entity_relationships": []
        }
        
        # Process individual text if provided
        if text:
            text_entities = self.entity_extractor.extract_entities(text)
            text_summary = self.entity_extractor._generate_entity_summary(text_entities)
            
            results["text_analysis"] = {
                "entities": text_entities,
                "summary": text_summary
            }
        
        # Process documents if provided
        entity_counts = {}
        for doc in documents:
            processed_doc = self.entity_extractor.process_document(doc)
            results["processed_documents"].append(processed_doc)
            
            # Count entity occurrences
            for entity_type, entities in processed_doc.get("extracted_entities", {}).items():
                if entity_type not in entity_counts:
                    entity_counts[entity_type] = {}
                
                for entity in entities:
                    entity_text = entity["text"]
                    if entity_text not in entity_counts[entity_type]:
                        entity_counts[entity_type][entity_text] = 0
                    entity_counts[entity_type][entity_text] += 1
        
        # Convert entity counts to a more structured format
        for entity_type, entities in entity_counts.items():
            entity_list = [{"text": k, "count": v} for k, v in entities.items()]
            entity_list.sort(key=lambda x: x["count"], reverse=True)
            results["entity_statistics"][entity_type] = entity_list
        
        # If full analysis is requested, identify entity relationships
        if analysis_type == "full" and documents:
            # Identify relationships between entities based on co-occurrence
            entity_pairs = {}
            
            for doc in results["processed_documents"]:
                entities = doc.get("extracted_entities", {})
                entity_texts = []
                
                # Collect all entities in this document
                for entity_type, entity_list in entities.items():
                    for entity in entity_list:
                        entity_texts.append((entity_type, entity["text"]))
                
                # Create pairs of co-occurring entities
                for i, (type1, text1) in enumerate(entity_texts):
                    for j, (type2, text2) in enumerate(entity_texts[i+1:], i+1):
                        if type1 != type2:  # Only consider relationships between different types
                            pair_key = f"{type1}:{text1}|{type2}:{text2}"
                            if pair_key not in entity_pairs:
                                entity_pairs[pair_key] = {
                                    "entity1_type": type1,
                                    "entity1_text": text1,
                                    "entity2_type": type2,
                                    "entity2_text": text2,
                                    "co_occurrences": 0
                                }
                            entity_pairs[pair_key]["co_occurrences"] += 1
            
            # Add relationships to results
            relationship_list = list(entity_pairs.values())
            relationship_list.sort(key=lambda x: x["co_occurrences"], reverse=True)
            results["entity_relationships"] = relationship_list[:100]  # Limit to top 100 relationships
        
        return results