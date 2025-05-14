# core/knowledge/graph_store.py
from neo4j import GraphDatabase
from typing import Dict, List, Any
import logging
import os
from datetime import datetime

class KnowledgeGraph:
    """Neo4j-based knowledge graph for competitive intelligence"""
    
    def __init__(self, uri: str = None, user: str = None, password: str = None):
        # Use parameters or environment variables
        self.uri = uri or os.environ.get("NEO4J_URI", "bolt://neo4j:7687")
        self.user = user or os.environ.get("NEO4J_USER", "neo4j")
        self.password = password or os.environ.get("NEO4J_PASSWORD", "password")
        
        self.driver = GraphDatabase.driver(self.uri, auth=(self.user, self.password))
        self.logger = logging.getLogger("knowledge_graph")
        
        # Initialize schema
        self._initialize_schema()
    
    def _initialize_schema(self):
        """Set up initial schema constraints"""
        with self.driver.session() as session:
            # Create uniqueness constraints
            session.run("CREATE CONSTRAINT IF NOT EXISTS FOR (a:Agency) REQUIRE a.id IS UNIQUE")
            session.run("CREATE CONSTRAINT IF NOT EXISTS FOR (c:Contractor) REQUIRE c.id IS UNIQUE")
            session.run("CREATE CONSTRAINT IF NOT EXISTS FOR (o:Opportunity) REQUIRE o.id IS UNIQUE")
            session.run("CREATE CONSTRAINT IF NOT EXISTS FOR (p:Person) REQUIRE p.id IS UNIQUE")
            session.run("CREATE CONSTRAINT IF NOT EXISTS FOR (t:Technology) REQUIRE t.name IS UNIQUE")
            session.run("CREATE CONSTRAINT IF NOT EXISTS FOR (r:Regulation) REQUIRE r.name IS UNIQUE")
            session.run("CREATE CONSTRAINT IF NOT EXISTS FOR (cl:Clearance) REQUIRE cl.level IS UNIQUE")
            
            # Create indexes for full-text search
            session.run("CREATE FULLTEXT INDEX opportunity_text IF NOT EXISTS FOR (o:Opportunity) ON EACH [o.title, o.description]")
            
            self.logger.info("Knowledge graph schema initialized")
    
    def add_contract_award(self, award_data: Dict[str, Any]):
        """Add a contract award to the knowledge graph with extracted entities"""
        with self.driver.session() as session:
            # Basic contract award structure
            session.run("""
                MERGE (a:Agency {id: $agency_id})
                ON CREATE SET a.name = $agency_name
                
                MERGE (c:Contractor {id: $contractor_id})
                ON CREATE SET c.name = $contractor_name
                
                MERGE (o:Opportunity {id: $opportunity_id})
                ON CREATE SET o.title = $title,
                           o.description = $description,
                           o.value = $value,
                           o.award_date = $award_date
                
                MERGE (c)-[r:WON {award_date: $award_date}]->(o)
                MERGE (a)-[:AWARDED {award_date: $award_date}]->(o)
            """, award_data)
            
            # Add extracted entities if available
            if 'entity_summary' in award_data:
                summary = award_data['entity_summary']
                
                # Add technologies
                for tech in summary.get('tech_stack', []):
                    session.run("""
                        MATCH (o:Opportunity {id: $opportunity_id})
                        MERGE (t:Technology {name: $tech_name})
                        MERGE (o)-[:REQUIRES]->(t)
                    """, {
                        'opportunity_id': award_data['opportunity_id'],
                        'tech_name': tech
                    })
                
                # Add regulations
                for reg in summary.get('regulatory_requirements', []):
                    session.run("""
                        MATCH (o:Opportunity {id: $opportunity_id})
                        MERGE (r:Regulation {name: $reg_name})
                        MERGE (o)-[:COMPLIES_WITH]->(r)
                    """, {
                        'opportunity_id': award_data['opportunity_id'],
                        'reg_name': reg
                    })
                
                # Add clearance requirements
                for clearance in summary.get('clearance_requirements', []):
                    session.run("""
                        MATCH (o:Opportunity {id: $opportunity_id})
                        MERGE (cl:Clearance {level: $clearance_level})
                        MERGE (o)-[:REQUIRES_CLEARANCE]->(cl)
                    """, {
                        'opportunity_id': award_data['opportunity_id'],
                        'clearance_level': clearance
                    })
            
            self.logger.info(f"Added contract award with entities: {award_data.get('title')}")
    
    def search_opportunities(self, query: str, limit: int = 10):
        """Search for opportunities by keyword"""
        with self.driver.session() as session:
            result = session.run("""
                CALL db.index.fulltext.queryNodes("opportunity_text", $query) 
                YIELD node, score
                RETURN node.id as id, node.title as title, node.description as description, 
                       node.value as value, node.award_date as award_date, score
                LIMIT $limit
            """, {"query": query, "limit": limit})
            
            return [dict(record) for record in result]
    
    def get_competitor_insights(self, competitor_id: str):
        """Get insights about a competitor"""
        with self.driver.session() as session:
            result = session.run("""
                MATCH (c:Contractor {id: $competitor_id})-[r:WON]->(o:Opportunity)<-[:AWARDED]-(a:Agency)
                RETURN a.name as agency_name, count(o) as contract_count, sum(o.value) as total_value
                ORDER BY total_value DESC
            """, {"competitor_id": competitor_id})
            
            return [dict(record) for record in result]
    
    def get_technology_landscape(self):
        """Get the technology landscape across opportunities"""
        with self.driver.session() as session:
            result = session.run("""
                MATCH (t:Technology)<-[:REQUIRES]-(o:Opportunity)
                RETURN t.name as technology, count(o) as opportunity_count, 
                       sum(o.value) as total_value
                ORDER BY opportunity_count DESC
            """)
            
            return [dict(record) for record in result]
    
    def get_entity_statistics(self, entity_type: str = None):
        """Get statistics on entities in the knowledge graph"""
        with self.driver.session() as session:
            if entity_type and entity_type.lower() == "technology":
                result = session.run("""
                    MATCH (t:Technology)<-[:REQUIRES]-(o:Opportunity)
                    RETURN t.name as name, count(o) as count
                    ORDER BY count DESC
                    LIMIT 10
                """)
                return {"entity_type": "technology", "statistics": [dict(record) for record in result]}
            
            elif entity_type and entity_type.lower() == "regulation":
                result = session.run("""
                    MATCH (r:Regulation)<-[:COMPLIES_WITH]-(o:Opportunity)
                    RETURN r.name as name, count(o) as count
                    ORDER BY count DESC
                    LIMIT 10
                """)
                return {"entity_type": "regulation", "statistics": [dict(record) for record in result]}
            
            elif entity_type and entity_type.lower() == "clearance":
                result = session.run("""
                    MATCH (cl:Clearance)<-[:REQUIRES_CLEARANCE]-(o:Opportunity)
                    RETURN cl.level as name, count(o) as count
                    ORDER BY count DESC
                    LIMIT 10
                """)
                return {"entity_type": "clearance", "statistics": [dict(record) for record in result]}
            
            else:
                # Return stats for all entity types
                tech_result = session.run("""
                    MATCH (t:Technology)
                    RETURN count(t) as count
                """)
                reg_result = session.run("""
                    MATCH (r:Regulation)
                    RETURN count(r) as count
                """)
                clear_result = session.run("""
                    MATCH (cl:Clearance)
                    RETURN count(cl) as count
                """)
                
                return {
                    "technology_count": tech_result.single()["count"],
                    "regulation_count": reg_result.single()["count"],
                    "clearance_count": clear_result.single()["count"]
                }
    
    def get_opportunity_entities(self, opportunity_id: str):
        """Get all entities associated with an opportunity"""
        with self.driver.session() as session:
            # Get technologies
            tech_result = session.run("""
                MATCH (o:Opportunity {id: $opportunity_id})-[:REQUIRES]->(t:Technology)
                RETURN t.name as name
            """, {"opportunity_id": opportunity_id})
            
            # Get regulations
            reg_result = session.run("""
                MATCH (o:Opportunity {id: $opportunity_id})-[:COMPLIES_WITH]->(r:Regulation)
                RETURN r.name as name
            """, {"opportunity_id": opportunity_id})
            
            # Get clearances
            clear_result = session.run("""
                MATCH (o:Opportunity {id: $opportunity_id})-[:REQUIRES_CLEARANCE]->(cl:Clearance)
                RETURN cl.level as name
            """, {"opportunity_id": opportunity_id})
            
            return {
                "technologies": [record["name"] for record in tech_result],
                "regulations": [record["name"] for record in reg_result],
                "clearances": [record["name"] for record in clear_result]
            }
    
    def find_similar_opportunities(self, opportunity_id: str, limit: int = 5):
        """Find opportunities with similar entity requirements"""
        with self.driver.session() as session:
            result = session.run("""
                MATCH (o:Opportunity {id: $opportunity_id})-[:REQUIRES]->(t:Technology)<-[:REQUIRES]-(similar:Opportunity)
                WHERE o <> similar
                WITH similar, count(t) as common_tech
                ORDER BY common_tech DESC
                LIMIT $limit
                RETURN similar.id as id, similar.title as title, similar.description as description,
                       similar.value as value, similar.award_date as award_date, common_tech
            """, {"opportunity_id": opportunity_id, "limit": limit})
            
            return [dict(record) for record in result]
    
    def close(self):
        """Close the database connection"""
        self.driver.close()