from __future__ import annotations

import json
import re
from typing import Any, Dict

import openai
from openai import OpenAI

from agent.memory import ConversationMemory
from agent.tools import TravelTools
from utils.config import OPENAI_API_KEY, OPENAI_MODEL, logger
from utils.prompts import INTENT_EXTRACTION_PROMPT, PLAN_GENERATION_PROMPT


class TravelPlannerAgent:
    """
    Main orchestration agent for travel planning.

    This is the core of our agentic AI system - it orchestrates the entire workflow:
      1. Extract user intent (using LLM or heuristics)
      2. Retrieve contextual data via Endee vector database (RAG)
      3. Generate a structured travel plan (using LLM with RAG context or fallback logic)
    
    The agent is designed to work even without OpenAI API keys, falling back to
    rule-based planning that still leverages RAG-retrieved context for better results.
    """

    def __init__(self, tools: TravelTools) -> None:
        """
        Initialize the travel planner agent.
        
        We try to set up OpenAI client if API key is available, but gracefully
        fall back to rule-based planning if it's not. This makes the system
        more robust and usable even without paid API access.
        """
        self.tools = tools
        self.memory = ConversationMemory()  # Simple conversation history
        self._client: OpenAI | None = None

        # Try to initialize OpenAI client - but don't fail if unavailable
        # This allows the system to work in fallback mode for testing/demos
        if OPENAI_API_KEY:
            try:
                # Initialize OpenAI client - handle version compatibility issues
                self._client = OpenAI(api_key=OPENAI_API_KEY)
                logger.info("TravelPlannerAgent initialised with OpenAI backend")
            except Exception as exc:
                # Some versions of openai library might have compatibility issues
                # Log the error but continue with fallback mode
                logger.warning(
                    "Failed to initialize OpenAI client: %s. Falling back to rule-based planner",
                    exc
                )
                self._client = None
        else:
            logger.warning("OPENAI_API_KEY not set – falling back to rule-based planner")

    # ------------------------------------------------------------------ #
    # Public entry point                                                 #
    # ------------------------------------------------------------------ #
    def plan_trip(self, query: str) -> Dict[str, Any]:
        """Run the full agentic workflow for a user query."""
        logger.info("Planning trip for query: %s", query)
        self.memory.add("user", query)

        intent = self._extract_intent(query)
        context = self._retrieve_context(intent)
        plan = self._generate_plan(intent, context)

        self.memory.add("assistant", json.dumps(plan, ensure_ascii=False))
        return plan

    # ------------------------------------------------------------------ #
    # Step 1 – Intent extraction                                         #
    # ------------------------------------------------------------------ #
    def _extract_intent(self, query: str) -> Dict[str, Any]:
        if self._client is None:
            logger.info("Using heuristic intent extraction")
            return self._heuristic_intent(query)

        prompt = INTENT_EXTRACTION_PROMPT.format(query=query)
        try:
            completion = self._client.chat.completions.create(
                model=OPENAI_MODEL,
                messages=[
                    {"role": "system", "content": "You extract structured travel intent."},
                    {"role": "user", "content": prompt},
                ],
                temperature=0.0,
            )
            raw = completion.choices[0].message.content or "{}"
            intent = json.loads(raw)
            logger.info("LLM intent extracted: %s", intent)
            return intent
        except Exception:
            logger.exception("LLM intent extraction failed, falling back to heuristics")
            return self._heuristic_intent(query)

    @staticmethod
    def _heuristic_intent(query: str) -> Dict[str, Any]:
        """
        Fallback intent extraction using simple regex and keyword matching.
        
        This is used when OpenAI API is not available. It's intentionally simple
        but covers common patterns like "3-day Goa trip under ₹10,000".
        In production, you'd want more sophisticated NLP here, but this works
        well enough for demos and testing.
        """
        q_lower = query.lower()

        # Simple destination detection - matches common Indian destinations
        # In a real system, you might use NER or a destination database
        known_destinations = ["goa", "delhi", "mumbai", "bangalore", "jaipur", "agra"]
        destination = next((d.capitalize() for d in known_destinations if d in q_lower), "Not specified")

        # Extract budget - look for ₹ symbol or "rupees" keyword
        # We preserve the original format to show it back to the user
        m = re.search(r"(₹\s*\d[\d,]*)|(\d[\d,]*\s*₹)", query)
        budget = m.group(0).strip() if m else "Not specified"

        # Extract duration - look for patterns like "3-day", "3 day", "3 days"
        m_days = re.search(r"(\d+)\s*[-]?\s*day", q_lower)
        duration = int(m_days.group(1)) if m_days else 3  # Default to 3 days if not specified

        # Extract preferences - match common travel keywords
        # These help us tailor the itinerary and budget estimates
        pref_keywords = ["beach", "mountain", "culture", "food", "adventure", "luxury", "budget", "nightlife"]
        prefs = [k for k in pref_keywords if k in q_lower] or ["general sightseeing"]

        return {
            "destination": destination,
            "budget": budget,
            "duration": duration,
            "preferences": prefs,
        }

    # ------------------------------------------------------------------ #
    # Step 2 – Retrieval via Endee (RAG)                                #
    # ------------------------------------------------------------------ #
    def _retrieve_context(self, intent: Dict[str, Any]) -> str:
        """
        Perform comprehensive RAG retrieval for travel planning.
        
        This is the heart of our RAG system - instead of a single query, we perform
        multiple targeted searches across different aspects of travel:
        - Destinations: general info about the place
        - Accommodations: hotels and stays
        - Food: dining options and cuisine
        - Activities: things to do based on preferences
        - General: overall travel guides
        
        This multi-aspect approach gives us richer context for generating detailed plans.
        The retrieved information is then formatted into a structured context string
        that the LLM (or fallback logic) can use to generate accurate, data-driven plans.
        """
        # Get categorized retrieval results - each aspect gets top 3 results
        # This balances comprehensiveness with token limits
        aspects = self.tools.retrieve_multiple_aspects(intent, top_k_per_aspect=3)
        
        # Format all aspects into a comprehensive context string
        # We structure it clearly so the LLM can easily parse different sections
        context_parts = []
        context_parts.append("=== RAG-RETRIEVED TRAVEL CONTEXT ===\n")
        
        # Add each aspect if we found relevant results
        if aspects.get("destinations"):
            context_parts.append("\n--- DESTINATION INFORMATION ---")
            context_parts.append(self.tools.format_context(aspects["destinations"]))
        
        if aspects.get("accommodations"):
            context_parts.append("\n--- ACCOMMODATION OPTIONS ---")
            context_parts.append(self.tools.format_context(aspects["accommodations"]))
        
        if aspects.get("food"):
            context_parts.append("\n--- FOOD & DINING ---")
            context_parts.append(self.tools.format_context(aspects["food"]))
        
        if aspects.get("activities"):
            context_parts.append("\n--- ACTIVITIES & ATTRACTIONS ---")
            context_parts.append(self.tools.format_context(aspects["activities"]))
        
        if aspects.get("general"):
            context_parts.append("\n--- GENERAL TRAVEL INFORMATION ---")
            context_parts.append(self.tools.format_context(aspects["general"]))
        
        context = "\n".join(context_parts)
        logger.info("Retrieved comprehensive RAG context (length=%d chars)", len(context))
        return context
    
    def _extract_from_rag_context(
        self, 
        context: str, 
        intent: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Extract structured information from RAG context for use in plan generation.
        This makes the fallback plan RAG-aware even without LLM.
        """
        extracted = {
            "hotels": [],
            "activities": [],
            "places": [],
            "food_options": [],
            "tips": []
        }
        
        # Simple extraction from context text
        # In a production system, you might use NER or structured parsing
        lines = context.split("\n")
        current_section = None
        
        for line in lines:
            line_lower = line.lower()
            
            # Detect sections
            if "accommodation" in line_lower or "hotel" in line_lower:
                current_section = "hotels"
            elif "activity" in line_lower or "attraction" in line_lower:
                current_section = "activities"
            elif "food" in line_lower or "dining" in line_lower:
                current_section = "food"
            
            # Extract hotel information
            if current_section == "hotels" and "price" in line_lower:
                price_match = line.split(":")[-1].strip() if ":" in line else ""
                if price_match:
                    extracted["hotels"].append({
                        "price_info": price_match,
                        "source": "RAG"
                    })
            
            # Extract activities
            if current_section == "activities" and any(word in line_lower for word in ["beach", "temple", "market", "palace", "museum"]):
                activity = line.split(":")[-1].strip() if ":" in line else line.strip()
                if activity and len(activity) > 5:
                    extracted["activities"].append(activity)
        
        logger.debug("Extracted %s items from RAG context", sum(len(v) for v in extracted.values()))
        return extracted

    # ------------------------------------------------------------------ #
    # Step 3 – Plan generation                                          #
    # ------------------------------------------------------------------ #
    def _generate_plan(self, intent: Dict[str, Any], context: str) -> Dict[str, Any]:
        """
        Generate the final travel plan using either LLM or fallback logic.
        
        We always compute a deterministic budget estimate first (using our
        rule-based logic) so we have reliable numbers even if LLM fails.
        Then we either:
        1. Use LLM with RAG context (preferred - more natural, context-aware)
        2. Use RAG-enhanced fallback logic (works without API keys, still uses RAG)
        
        The fallback is actually quite sophisticated - it extracts information
        from the RAG context and builds detailed plans, so results are still good.
        """
        duration = int(intent.get("duration") or 3)
        
        # Always compute budget deterministically - this ensures consistency
        # The LLM might adjust it, but we have a baseline
        budget_info = self.tools.estimate_budget(
            intent.get("destination", "Not specified"),
            duration_days=duration,
            preferences=intent.get("preferences") or [],
        )

        # If no OpenAI client, use our RAG-enhanced fallback
        if self._client is None:
            logger.info("Using RAG-enhanced rule-based plan generation")
            return self._fallback_plan(intent, budget_info, context)

        # Try LLM-based generation with RAG context
        intent_json = json.dumps(intent, ensure_ascii=False)
        prompt = PLAN_GENERATION_PROMPT.format(intent_json=intent_json, context=context)

        try:
            completion = self._client.chat.completions.create(
                model=OPENAI_MODEL,
                messages=[
                    {
                        "role": "system",
                        "content": "You are a meticulous Indian travel planner. "
                        "Always respond with strict JSON as specified.",
                    },
                    {"role": "user", "content": prompt},
                ],
                temperature=0.3,  # Lower temperature for more consistent, factual output
            )
            raw = completion.choices[0].message.content or "{}"
            plan = json.loads(raw)

            # Ensure required top-level fields exist - LLM might miss some
            plan.setdefault("destination", intent.get("destination", "Not specified"))
            plan.setdefault("itinerary", [])
            plan.setdefault("hotels", [])
            plan.setdefault("tips", [])

            # Merge our deterministic budget into the LLM's structure
            # This ensures we always have reliable budget numbers
            llm_budget = plan.get("budget") or {}
            llm_budget.update(budget_info)
            plan["budget"] = llm_budget

            return plan
        except Exception:
            # If LLM fails (network, API error, invalid JSON), fall back gracefully
            logger.exception("LLM plan generation failed, falling back to RAG-enhanced plan")
            return self._fallback_plan(intent, budget_info, context)

    def _fallback_plan(
        self, 
        intent: Dict[str, Any], 
        budget_info: Dict[str, Any],
        context: str = ""
    ) -> Dict[str, Any]:
        """
        RAG-enhanced fallback plan when LLM is unavailable.
        Uses retrieved context to generate comprehensive, detailed journey plans.
        """
        destination = intent.get("destination", "Destination")
        duration = int(intent.get("duration") or 3)
        preferences = intent.get("preferences", [])

        # Extract information from RAG context
        rag_data = self._extract_from_rag_context(context, intent) if context else {}
        
        # Build comprehensive itinerary with detailed journey planning
        itinerary: list[Dict[str, Any]] = []
        
        # Try to extract detailed activities from context
        detailed_activities = []
        if context:
            # Look for detailed_activities in context
            import re
            detailed_match = re.search(r'"detailed_activities":\s*\[(.*?)\]', context, re.DOTALL)
            if detailed_match:
                # Extract structured activity data
                try:
                    import json
                    # Try to parse if it's JSON-like
                    activities_text = detailed_match.group(1)
                    # Simple extraction of day-based activities
                    for day_num in range(1, duration + 1):
                        day_pattern = f'"day":\s*{day_num}'
                        if day_pattern in activities_text:
                            # Extract activities for this day
                            day_section = re.search(
                                f'{{"day":\s*{day_num}.*?}}',
                                activities_text,
                                re.DOTALL
                            )
                            if day_section:
                                detailed_activities.append({
                                    "day": day_num,
                                    "raw": day_section.group(0)
                                })
                except:
                    pass
        
        # Build detailed itinerary with comprehensive journey planning
        activities_list = rag_data.get("activities", [])
        
        for day in range(1, duration + 1):
            day_plan = {
                "day": day,
                "title": f"Day {day} in {destination}",
                "activities": [],
                "timing": {},
                "transport": "",
                "estimated_cost": 0,
                "source": "RAG-enhanced" if context else "default"
            }
            
            # Try to extract detailed activities from context
            if context and (f"Day {day}" in context or f'"day": {day}' in context):
                # Extract morning, afternoon, evening activities from context
                morning_acts = self._extract_time_specific_activities(context, day, "morning")
                afternoon_acts = self._extract_time_specific_activities(context, day, "afternoon")
                evening_acts = self._extract_time_specific_activities(context, day, "evening")
                early_morning_acts = self._extract_time_specific_activities(context, day, "early_morning")
                
                if morning_acts or afternoon_acts or evening_acts or early_morning_acts:
                    day_plan["timing"] = {}
                    if early_morning_acts:
                        day_plan["timing"]["early_morning"] = early_morning_acts
                    if morning_acts:
                        day_plan["timing"]["morning"] = morning_acts
                    if afternoon_acts:
                        day_plan["timing"]["afternoon"] = afternoon_acts
                    if evening_acts:
                        day_plan["timing"]["evening"] = evening_acts
                    
                    # Flatten for activities list
                    all_activities = early_morning_acts + morning_acts + afternoon_acts + evening_acts
                    day_plan["activities"] = all_activities if all_activities else self._get_default_day_activities(destination, day, duration, preferences)
                else:
                    # Use general activities
                    day_plan["activities"] = self._get_default_day_activities(destination, day, duration, preferences)
            else:
                # Use general activities
                day_plan["activities"] = self._get_default_day_activities(destination, day, duration, preferences)
            
            # Extract transport information from context
            transport_info = self._extract_transport_info(context, destination)
            if transport_info:
                day_plan["transport"] = transport_info
            else:
                day_plan["transport"] = "Local transport (auto-rickshaw/taxi) recommended"
            
            # Estimate daily cost
            day_plan["estimated_cost"] = budget_info.get("daily_cost", 0)
            
            itinerary.append(day_plan)
        
        # Build hotels and tips
        hotels, tips = self._build_hotels_and_tips(rag_data, context, destination, preferences)
        
        # Extract recommendations from context
        recommendations = self._extract_recommendations(context, destination, preferences)

        plan = {
            "destination": destination,
            "budget": budget_info,
            "itinerary": itinerary,
            "hotels": hotels,
            "tips": tips,
            "recommendations": recommendations,
            "rag_enhanced": bool(context and len(context) > 100),
        }
        
        logger.info("Generated RAG-enhanced fallback plan (RAG data used: %s)", plan["rag_enhanced"])
        return plan
    
    def _extract_time_specific_activities(self, context: str, day: int, time_period: str) -> list[str]:
        """Extract activities for specific day and time period from context."""
        activities = []
        if not context:
            return activities
        
        import re
        # Look for patterns like "Day X: Morning - Activity" or "morning": ["activity"]
        patterns = [
            rf'Day\s+{day}.*?{time_period}[:\-]\s*([^\n]+)',
            rf'"{time_period}":\s*\[(.*?)\]',
            rf'{time_period}\s*-\s*([^\n]+)',
        ]
        
        for pattern in patterns:
            matches = re.findall(pattern, context, re.IGNORECASE | re.DOTALL)
            for match in matches:
                if isinstance(match, str):
                    # Clean and split activities
                    acts = [a.strip() for a in match.split(',') if a.strip()]
                    activities.extend(acts)
        
        return activities[:5]  # Limit to 5 activities per time period
    
    def _extract_transport_info(self, context: str, destination: str) -> str:
        """Extract transport information from context."""
        if not context:
            return ""
        
        import re
        # Look for transport recommendations
        transport_patterns = [
            rf'transport[:\-]\s*([^\n]+)',
            rf'getting to {destination}[:\-]\s*([^\n]+)',
            rf'local transport[:\-]\s*([^\n]+)',
        ]
        
        for pattern in transport_patterns:
            match = re.search(pattern, context, re.IGNORECASE)
            if match:
                return match.group(1).strip()[:200]  # Limit length
        
        return ""
    
    def _get_default_day_activities(self, destination: str, day: int, duration: int, preferences: list) -> list[str]:
        """Generate default activities for a day based on destination and preferences."""
        activities = []
        
        if day == 1:
            activities = [
                f"Arrive in {destination.capitalize()} and check into accommodation",
                f"Explore main attractions in {destination.capitalize()}",
                "Try local cuisine for lunch",
                "Evening walk and local market visit"
            ]
        elif day == duration:
            activities = [
                f"Visit remaining key attractions in {destination.capitalize()}",
                "Shopping for souvenirs and local handicrafts",
                "Farewell dinner at a recommended restaurant",
                "Prepare for departure"
            ]
        else:
            activities = [
                f"Day {day} exploration of {destination.capitalize()}",
                "Visit cultural and historical sites",
                "Local dining experience",
                "Evening leisure activities"
            ]
        
        # Add preference-based activities
        if "beach" in preferences:
            activities.insert(1, "Beach activities and water sports")
        if "culture" in preferences or "history" in preferences:
            activities.insert(1, "Visit heritage sites and museums")
        if "food" in preferences:
            activities.insert(2, "Food tour and local delicacies")
        
        return activities
    
    def _build_hotels_and_tips(
        self,
        rag_data: Dict[str, Any],
        context: str,
        destination: str,
        preferences: list
    ) -> tuple[list[Dict[str, Any]], list[str]]:
        """Build hotels and tips from RAG context."""
        hotels = []
        rag_hotels = rag_data.get("hotels", [])
        
        if rag_hotels:
            # Use RAG-retrieved hotel information
            for i, hotel_info in enumerate(rag_hotels[:2], 1):
                hotels.append({
                    "name": f"RAG-Recommended Option {i}",
                    "approx_price_per_night": hotel_info.get("price_info", "₹1,000–₹3,000"),
                    "area": "Based on RAG search results",
                    "rating": 4.0 + (i * 0.2),
                    "notes": "Recommended based on vector search of travel database.",
                    "source": "RAG"
                })
        
        # Add default hotels if RAG didn't provide enough
        if len(hotels) < 2:
            hotels.extend([
                {
                    "name": "Budget Stay",
                    "approx_price_per_night": "₹1,000–₹2,000",
                    "area": "Convenient central area",
                    "rating": 4.0,
                    "notes": "Good for budget travellers; book early in peak season.",
                    "source": "default"
                },
                {
                    "name": "Comfort Hotel",
                    "approx_price_per_night": "₹2,500–₹4,000",
                    "area": "Safe, well-connected neighbourhood",
                    "rating": 4.4,
                    "notes": "Balanced comfort and cost; ideal for couples or families.",
                    "source": "default"
                }
            ])
            hotels = hotels[:2]  # Keep only 2

        # Generate context-aware tips
        tips = [
            "Book trains/buses and hotels at least 2–3 weeks in advance.",
            "Carry a small day-pack with water, sunscreen, and power bank.",
            "Use reputable ride-hailing apps or official taxis when travelling late.",
            "Keep digital copies of your ID and booking confirmations.",
            "Check local weather and festivals before finalising dates.",
        ]
        
        # Add RAG-based tips if available
        if context and "budget" in context.lower():
            tips.insert(0, "Budget information retrieved from travel database - verify current prices before booking.")
        if context and any(pref in context.lower() for pref in preferences):
            tips.insert(0, f"Activities matched your preferences: {', '.join(preferences)}")
        
        return hotels, tips
    
    def _extract_recommendations(self, context: str, destination: str, preferences: list) -> Dict[str, Any]:
        """Extract travel recommendations from RAG context."""
        recommendations = {
            "best_time_to_visit": "",
            "target_audience": [],
            "best_for": [],
            "avoid_season": "",
            "tips": []
        }
        
        if not context:
            return recommendations
        
        import re
        
        # Extract best time to visit
        time_match = re.search(r'best time[:\-]\s*([^\n]+)', context, re.IGNORECASE)
        if time_match:
            recommendations["best_time_to_visit"] = time_match.group(1).strip()
        
        # Extract target audience
        audience_match = re.search(r'target_audience[:\-]\s*\[(.*?)\]', context, re.IGNORECASE | re.DOTALL)
        if audience_match:
            audience_text = audience_match.group(1)
            recommendations["target_audience"] = [a.strip().strip('"') for a in audience_text.split(',') if a.strip()]
        
        # Extract best for
        best_for_match = re.search(r'best for[:\-]\s*([^\n]+)', context, re.IGNORECASE)
        if best_for_match:
            best_for_text = best_for_match.group(1)
            recommendations["best_for"] = [b.strip() for b in best_for_text.split(',') if b.strip()]
        
        # Extract avoid season
        avoid_match = re.search(r'avoid[:\-]\s*([^\n]+)', context, re.IGNORECASE)
        if avoid_match:
            recommendations["avoid_season"] = avoid_match.group(1).strip()
        
        # Extract tips
        tips_match = re.search(r'tips[:\-]\s*\[(.*?)\]', context, re.IGNORECASE | re.DOTALL)
        if tips_match:
            tips_text = tips_match.group(1)
            recommendations["tips"] = [t.strip().strip('"') for t in tips_text.split(',') if t.strip()]
        
        return recommendations

