from __future__ import annotations

from typing import Any, Dict, List

from retriever.embedder import Embedder
from retriever.vector_store import EndeeVectorStore
from utils.config import logger


class TravelTools:
    """
    Toolset used by the travel planning agent.
    
    This class provides the agent with two key capabilities:
    1. **Retrieval**: Semantic search over Endee vector database using embeddings
    2. **Budget estimation**: Deterministic cost estimates in INR
    
    The retrieval system is the core of our RAG implementation - it performs
    multiple targeted searches across different travel aspects to gather
    comprehensive context for plan generation.
    """

    def __init__(self, embedder: Embedder, vector_store: EndeeVectorStore) -> None:
        self.embedder = embedder
        self.vector_store = vector_store

    # ------------------------------------------------------------------ #
    # Retrieval                                                          #
    # ------------------------------------------------------------------ #
    def retrieve_travel_data(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """Retrieve relevant travel documents using Endee vector search."""
        if not query:
            return []
        try:
            query_vec = self.embedder.embed_text(query)
            results = self.vector_store.search(query_vec, top_k=top_k)
            logger.info("Retrieved %s documents from Endee", len(results))
            return results
        except Exception:
            logger.exception("Failed to retrieve travel data for query='%s'", query)
            return []

    @staticmethod
    def format_context(results: List[Dict[str, Any]]) -> str:
        """
        Format retrieved results into rich contextual information for RAG.
        Includes all relevant metadata for comprehensive context.
        """
        if not results:
            return "No matching travel data was found in the Endee vector database."

        lines: List[str] = []
        lines.append("=== RETRIEVED TRAVEL INFORMATION (from Endee Vector Database) ===\n")
        
        for i, item in enumerate(results, start=1):
            doc = item.get("document", {}) or {}
            score = float(item.get("score", 0.0))
            
            # Extract all available information
            title = doc.get("title") or doc.get("destination") or "Travel item"
            description = doc.get("description", "")
            destination = doc.get("destination", "")
            doc_type = doc.get("type", "")
            price_range = doc.get("price_range_inr") or doc.get("price_range", "")
            area = doc.get("area", "")
            activities = doc.get("activities", [])
            budget_category = doc.get("budget_category", "")
            
            lines.append(f"\n[{i}] {title} (Relevance: {score:.2f})")
            if destination:
                lines.append(f"    Destination: {destination}")
            if doc_type:
                lines.append(f"    Type: {doc_type}")
            if description:
                lines.append(f"    Description: {description}")
            if price_range:
                lines.append(f"    Price Range: {price_range}")
            if area:
                lines.append(f"    Area/Location: {area}")
            if activities:
                lines.append(f"    Activities: {', '.join(activities)}")
            if budget_category:
                lines.append(f"    Budget Category: {budget_category}")
        
        lines.append("\n=== END OF RETRIEVED CONTEXT ===")
        return "\n".join(lines)
    
    def retrieve_multiple_aspects(
        self, 
        intent: Dict[str, Any], 
        top_k_per_aspect: int = 3
    ) -> Dict[str, List[Dict[str, Any]]]:
        """
        Perform multiple RAG queries for different aspects of travel planning.
        
        This is a key enhancement over simple single-query RAG. Instead of one
        broad search, we perform targeted searches for:
        1. Destinations: General info about the place
        2. Accommodations: Hotels and stays with prices
        3. Food: Dining options and cuisine
        4. Activities: Things to do based on user preferences
        5. General: Overall travel guides and tips
        
        Each aspect gets top_k results, giving us comprehensive context while
        keeping token usage reasonable. The results are categorized so the
        plan generator can easily find relevant information for each section.
        
        This multi-aspect approach significantly improves plan quality because
        we're not relying on a single query to cover everything.
        """
        destination = intent.get("destination", "").lower()
        duration = intent.get("duration", 3)
        preferences = intent.get("preferences", [])
        
        aspects = {}
        
        # 1. Destination information - general overview and attractions
        if destination:
            dest_query = f"{destination} travel destination attractions"
            aspects["destinations"] = self.retrieve_travel_data(dest_query, top_k=top_k_per_aspect)
        
        # 2. Accommodation/hotels - where to stay with budget info
        hotel_query = f"{destination} hotels accommodation stays budget"
        aspects["accommodations"] = self.retrieve_travel_data(hotel_query, top_k=top_k_per_aspect)
        
        # 3. Food and dining - local cuisine and restaurants
        food_query = f"{destination} food restaurants cuisine dining"
        aspects["food"] = self.retrieve_travel_data(food_query, top_k=top_k_per_aspect)
        
        # 4. Activities based on preferences - personalized recommendations
        if preferences:
            activity_query = f"{destination} {' '.join(preferences)} activities things to do"
            aspects["activities"] = self.retrieve_travel_data(activity_query, top_k=top_k_per_aspect)
        
        # 5. General travel info - overall guides and tips
        general_query = f"{destination} {duration} day trip travel guide"
        aspects["general"] = self.retrieve_travel_data(general_query, top_k=top_k_per_aspect)
        
        logger.info("Retrieved %s aspects with RAG", len(aspects))
        return aspects

    # ------------------------------------------------------------------ #
    # Budget estimation                                                  #
    # ------------------------------------------------------------------ #
    @staticmethod
    def estimate_budget(
        destination: str,
        duration_days: int,
        preferences: List[str] | None = None,
    ) -> Dict[str, Any]:
        """
        Deterministic budget estimate in INR.
        
        This is intentionally simple and rule-based so the agent can rely
        on it even without an LLM call. The estimates are rough but reasonable
        for budget-conscious Indian travel.
        
        We use base costs per destination (based on typical mid-range travel)
        and apply multipliers based on preferences (luxury = more expensive,
        budget = less expensive).
        
        In a production system, you'd want to:
        - Pull real-time prices from APIs
        - Consider seasonal variations
        - Factor in group size
        - Use historical booking data
        But for this demo, deterministic estimates work well.
        """
        dest_key = (destination or "").lower()
        prefs = [p.lower() for p in (preferences or [])]

        # Base daily costs per destination (accommodation, food, transport)
        # These are rough estimates for mid-range travel in 2024
        base_costs = {
            "goa": {"accommodation": 1200, "food": 500, "transport": 300},
            "mumbai": {"accommodation": 1800, "food": 600, "transport": 400},
            "delhi": {"accommodation": 1500, "food": 500, "transport": 350},
            "bangalore": {"accommodation": 1400, "food": 500, "transport": 300},
            "jaipur": {"accommodation": 1000, "food": 400, "transport": 250},
            "agra": {"accommodation": 900, "food": 400, "transport": 250},
        }
        # Default to Goa-like costs if destination not in our list
        costs = base_costs.get(
            dest_key, {"accommodation": 1200, "food": 500, "transport": 300}
        )

        # Apply preference-based multipliers
        # Luxury travel costs more, budget travel costs less
        multiplier = 1.0
        if "luxury" in prefs:
            multiplier += 0.6  # 60% more expensive
        if "adventure" in prefs:
            multiplier += 0.2  # Adventure activities cost more
        if "budget" in prefs:
            multiplier -= 0.2  # Budget travel is cheaper

        # Calculate final costs
        adjusted = {k: int(v * multiplier) for k, v in costs.items()}
        daily_cost = sum(adjusted.values())
        total_cost = daily_cost * max(duration_days, 1)  # Ensure at least 1 day

        logger.info(
            "Estimated budget for destination=%s days=%d prefs=%s: total=%d",
            destination,
            duration_days,
            prefs,
            total_cost,
        )

        return {
            "currency": "INR",
            "daily_cost": daily_cost,
            "total_cost": total_cost,
            "breakdown": adjusted,  # Detailed breakdown for transparency
        }

