from __future__ import annotations

import json
from typing import Tuple

import streamlit as st

from agent.planner import TravelPlannerAgent
from agent.tools import TravelTools
from retriever.embedder import Embedder
from retriever.vector_store import EndeeVectorStore
from utils.config import logger


st.set_page_config(
    page_title="AI Travel Planner (Endee-powered)",
    page_icon="✈️",
    layout="wide",
)


@st.cache_resource
def init_agent() -> tuple[TravelPlannerAgent | None, bool]:
    """
    Create and cache the travel planner agent for the app.
    Returns (agent, endee_available) tuple.
    """
    try:
        embedder = Embedder()
        store = EndeeVectorStore()
        endee_available = store.is_available()
        tools = TravelTools(embedder, store)
        agent = TravelPlannerAgent(tools)
        return agent, endee_available
    except Exception as exc:  # pragma: no cover - startup failure path
        logger.exception("Failed to initialise TravelPlannerAgent")
        return None, False


def render_header() -> None:
    st.title("✈️ AI Travel Planning Agent using Endee Vector Database")
    st.markdown(
        """
This app demonstrates an **agentic AI workflow** for travel planning:

- **Intent understanding** of your natural-language request  
- **Retrieval-Augmented Generation (RAG)** over a travel dataset stored in **Endee**  
- **Multi-step reasoning** to create a complete, structured trip plan  
        """
    )


def render_sidebar(endee_available: bool) -> None:
    st.sidebar.header("Example queries")
    st.sidebar.write(
        "- Plan a 3-day Goa trip under ₹10,000\n"
        "- Weekend budget trip to Jaipur from Delhi\n"
        "- 4-day family trip to Agra and Delhi with focus on history\n"
    )
    st.sidebar.markdown("---")
    
    if endee_available:
        st.sidebar.success("✅ Endee vector database is connected")
    else:
        st.sidebar.warning(
            "⚠️ Endee service not available\n\n"
            "The app is running in **fallback mode** with limited RAG capabilities. "
            "To enable full Endee integration:\n\n"
            "1. Start your Endee service\n"
            "2. Set `ENDEE_BASE_URL` in `.env` or environment\n"
            "3. Run `python scripts/ingest.py` to populate data\n"
            "4. Refresh this page"
        )


def render_plan(plan: dict) -> None:
    if not plan:
        st.warning("No plan generated.")
        return

    if "error" in plan:
        st.error(plan["error"])
        return

    destination = plan.get("destination", "Not specified")
    budget = plan.get("budget", {})
    itinerary = plan.get("itinerary", [])
    hotels = plan.get("hotels", [])
    tips = plan.get("tips", [])

    top_col1, top_col2 = st.columns(2)
    with top_col1:
        st.subheader("📍 Destination")
        st.write(f"**{destination}**")
    with top_col2:
        st.subheader("💰 Budget (estimate)")
        if budget:
            st.write(f"Daily: ₹{budget.get('daily_cost', 0)}")
            st.write(f"Total: ₹{budget.get('total_cost', 0)}")
            breakdown = budget.get("breakdown", {})
            if breakdown:
                st.caption(
                    ", ".join(f"{k}: ₹{v}" for k, v in breakdown.items())
                )

    st.markdown("---")
    st.subheader("📅 Detailed Itinerary & Journey Plan")
    if itinerary:
        for day in itinerary:
            day_label = day.get("title") or f"Day {day.get('day', '?')}"
            with st.expander(day_label, expanded=(day.get("day", 1) == 1)):
                # Display timing-based activities if available
                timing = day.get("timing", {})
                if timing:
                    if timing.get("early_morning"):
                        st.markdown("**🌅 Early Morning:**")
                        for act in timing["early_morning"]:
                            st.write(f"  • {act}")
                    if timing.get("morning"):
                        st.markdown("**☀️ Morning:**")
                        for act in timing["morning"]:
                            st.write(f"  • {act}")
                    if timing.get("afternoon"):
                        st.markdown("**🌤️ Afternoon:**")
                        for act in timing["afternoon"]:
                            st.write(f"  • {act}")
                    if timing.get("evening"):
                        st.markdown("**🌆 Evening:**")
                        for act in timing["evening"]:
                            st.write(f"  • {act}")
                else:
                    # Fallback to simple activities list
                    st.markdown("**Activities:**")
                    for activity in day.get("activities", []):
                        st.write(f"  • {activity}")
                
                # Display transport information
                transport = day.get("transport")
                if transport:
                    st.markdown(f"**🚗 Transport:** {transport}")
                
                # Display estimated cost for the day
                day_cost = day.get("estimated_cost", 0)
                if day_cost:
                    st.caption(f"Estimated cost for this day: ₹{day_cost}")
                
                # Show source if available
                source = day.get("source")
                if source:
                    st.caption(f"*Source: {source}*")
    else:
        st.write("Itinerary details were not provided.")

    st.subheader("🏨 Suggested hotels / stays")
    if hotels:
        for hotel in hotels:
            st.markdown(f"**{hotel.get('name', 'Hotel')}**")
            price = hotel.get("approx_price_per_night")
            if price:
                st.write(f"Price per night: {price}")
            area = hotel.get("area")
            if area:
                st.write(f"Area: {area}")
            rating = hotel.get("rating")
            if rating:
                st.write(f"Rating: {rating} ⭐")
            notes = hotel.get("notes")
            if notes:
                st.caption(notes)
            st.markdown("---")
    else:
        st.write("Hotel suggestions were not provided.")

    st.subheader("💡 Travel Tips")
    if tips:
        for tip in tips:
            st.write(f"• {tip}")
    else:
        st.write("No specific tips available.")
    
    # Display recommendations if available
    recommendations = plan.get("recommendations", {})
    if recommendations and any(recommendations.values()):
        st.markdown("---")
        st.subheader("🎯 Recommendations")
        
        if recommendations.get("best_time_to_visit"):
            st.markdown(f"**Best Time to Visit:** {recommendations['best_time_to_visit']}")
        
        if recommendations.get("target_audience"):
            st.markdown(f"**Ideal For:** {', '.join(recommendations['target_audience'])}")
        
        if recommendations.get("best_for"):
            st.markdown(f"**Best For:** {', '.join(recommendations['best_for'])}")
        
        if recommendations.get("avoid_season"):
            st.markdown(f"**Avoid:** {recommendations['avoid_season']}")
        
        if recommendations.get("tips"):
            st.markdown("**Destination-Specific Tips:**")
            for tip in recommendations["tips"]:
                st.write(f"  • {tip}")

    st.markdown("---")
    with st.expander("Raw JSON output"):
        st.json(plan)


def main() -> None:
    render_header()
    
    agent, endee_available = init_agent()
    render_sidebar(endee_available)
    
    if agent is None:
        st.error(
            "❌ The travel agent could not be initialised.\n\n"
            "**Possible causes:**\n"
            "- Missing dependencies (check logs)\n"
            "- Configuration errors\n\n"
            "Please check the terminal/console for detailed error messages."
        )
        return

    if not endee_available:
        st.info(
            "ℹ️ **Running in fallback mode**: The app will work but with limited "
            "semantic search capabilities. Travel plans will be generated using "
            "rule-based logic. For full RAG features, start the Endee service."
        )

    query = st.text_input(
        "Describe your travel goal",
        placeholder="e.g. Plan a 3-day Goa trip under ₹10,000",
    )

    if st.button("Generate travel plan"):
        if not query.strip():
            st.warning("Please enter a travel query first.")
            return

        mode_text = "Endee-powered RAG" if endee_available else "fallback mode"
        with st.spinner(f"Planning your trip ({mode_text})..."):
            try:
                plan = agent.plan_trip(query.strip())
                render_plan(plan)
            except Exception as exc:  # pragma: no cover - UI error path
                logger.exception("Failed to generate plan")
                st.error(f"Failed to generate plan: {exc}")


if __name__ == "__main__":
    main()

