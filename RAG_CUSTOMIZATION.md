# 🎯 RAG (Retrieval-Augmented Generation) Customization Guide

This document explains how the entire project is customized for RAG-based travel planning.

---

## 🔍 RAG Architecture Overview

The project implements a **comprehensive RAG system** where:

1. **User Query** → Intent Extraction
2. **Intent** → Multiple RAG Queries (different aspects)
3. **Retrieved Context** → LLM/Fallback Plan Generation
4. **Output** → Data-driven travel plans

---

## 🚀 Key RAG Features

### 1. Multi-Aspect Retrieval

Instead of a single query, the system performs **multiple targeted RAG queries**:

- **Destinations**: `"{destination} travel destination attractions"`
- **Accommodations**: `"{destination} hotels accommodation stays budget"`
- **Food**: `"{destination} food restaurants cuisine dining"`
- **Activities**: `"{destination} {preferences} activities things to do"`
- **General**: `"{destination} {duration} day trip travel guide"`

**Location:** `agent/tools.py` → `retrieve_multiple_aspects()`

### 2. Rich Context Formatting

Retrieved documents include **all metadata** for comprehensive context:

- Title and description
- Destination
- Document type (destination, accommodation, food, etc.)
- Price ranges
- Areas/locations
- Activities list
- Budget categories
- Relevance scores

**Location:** `agent/tools.py` → `format_context()`

### 3. RAG-Enhanced Prompts

LLM prompts explicitly instruct to:
- **PRIMARILY use retrieved context**
- Extract specific hotels, prices, and areas from context
- Use actual activities from retrieved data
- Incorporate real budget information
- Only use general knowledge when context is missing

**Location:** `utils/prompts.py` → `PLAN_GENERATION_PROMPT`

### 4. RAG-Aware Fallback Mode

Even without LLM, the fallback plan:
- Extracts information from RAG context
- Uses retrieved activities in itinerary
- Incorporates hotel data from context
- Generates context-aware tips
- Marks plans as "RAG-enhanced"

**Location:** `agent/planner.py` → `_fallback_plan()`

### 5. Semantic Fallback Search

When Endee is unavailable, fallback mode:
- Computes embeddings for all documents
- Performs **cosine similarity search**
- Returns semantically similar results
- Maintains RAG quality even offline

**Location:** `retriever/vector_store.py` → `_fallback_search()`

---

## 📊 RAG Flow Diagram

```
User Query: "Plan a 3-day Goa trip under ₹10,000"
    │
    ├─→ Step 1: Extract Intent
    │   └─→ {destination: "Goa", duration: 3, budget: "₹10,000", preferences: ["beach", "budget"]}
    │
    ├─→ Step 2: Multi-Aspect RAG Retrieval
    │   ├─→ Query 1: "goa travel destination attractions" → [3 results]
    │   ├─→ Query 2: "goa hotels accommodation stays budget" → [3 results]
    │   ├─→ Query 3: "goa food restaurants cuisine dining" → [3 results]
    │   ├─→ Query 4: "goa beach budget activities things to do" → [3 results]
    │   └─→ Query 5: "goa 3 day trip travel guide" → [3 results]
    │
    ├─→ Step 3: Format Comprehensive Context
    │   └─→ Rich formatted context with all metadata
    │
    └─→ Step 4: Generate Plan (RAG-Enhanced)
        ├─→ With LLM: Uses context as primary source
        └─→ Without LLM: Extracts data from context for fallback plan
```

---

## 🔧 RAG Customization Details

### Context Extraction (`_extract_from_rag_context`)

Extracts structured data from RAG context:
- Hotels with price information
- Activities and attractions
- Places to visit
- Food options
- Context-aware tips

**Used by:** Fallback plan generation

### Multiple Retrieval Queries

Different query strategies for different aspects:
- **Broad queries** for general information
- **Specific queries** for hotels/accommodations
- **Preference-based queries** for activities
- **Duration-aware queries** for itineraries

### Context-Aware Itinerary

- Distributes RAG-retrieved activities across days
- Uses actual place names from context
- References specific attractions mentioned
- Marks activities with "RAG-enhanced" source

### Hotel Recommendations from RAG

- Extracts hotel names from context
- Uses actual price ranges from retrieved data
- Includes areas/locations from context
- Falls back to defaults only if RAG provides insufficient data

---

## 📈 RAG Quality Improvements

### Before RAG Customization:
- Single generic query
- Basic context formatting
- Hardcoded fallback responses
- No semantic search in fallback
- LLM prompts didn't emphasize context usage

### After RAG Customization:
- ✅ Multiple aspect-specific queries
- ✅ Rich metadata in context
- ✅ RAG-enhanced fallback plans
- ✅ Semantic similarity in fallback mode
- ✅ Explicit context-first LLM instructions
- ✅ Context extraction for structured data
- ✅ RAG source tracking in outputs

---

## 🧪 Testing RAG Functionality

### Test 1: Verify Multi-Aspect Retrieval

```python
from agent.tools import TravelTools
from retriever.embedder import Embedder
from retriever.vector_store import EndeeVectorStore

tools = TravelTools(Embedder(), EndeeVectorStore())
intent = {"destination": "Goa", "duration": 3, "preferences": ["beach"]}
aspects = tools.retrieve_multiple_aspects(intent)
print(f"Retrieved {len(aspects)} aspects")
```

### Test 2: Check Context Formatting

```python
results = tools.retrieve_travel_data("Goa beach trip")
context = tools.format_context(results)
print(context)  # Should show rich metadata
```

### Test 3: Verify RAG-Enhanced Fallback

```python
from agent.planner import TravelPlannerAgent

agent = TravelPlannerAgent(tools)
plan = agent.plan_trip("Plan a 3-day Goa trip")
print(plan.get("rag_enhanced"))  # Should be True if context was used
```

---

## 🎯 RAG Best Practices Implemented

1. **Multiple Queries**: Different queries for different information needs
2. **Rich Context**: Include all relevant metadata in context
3. **Context-First**: LLM prompts prioritize retrieved data
4. **Graceful Degradation**: RAG works even without LLM or Endee
5. **Source Attribution**: Track which data came from RAG
6. **Semantic Search**: Use embeddings even in fallback mode
7. **Structured Extraction**: Parse context for structured outputs

---

## 📝 RAG Configuration

### Retrieval Parameters

- **Top K per aspect**: 3 (configurable in `retrieve_multiple_aspects`)
- **Overall top K**: 5 (default in `retrieve_travel_data`)
- **Embedding model**: `all-MiniLM-L6-v2` (384 dimensions)

### Context Formatting

- Includes relevance scores
- Shows document types
- Presents price ranges
- Lists activities
- Provides areas/locations

### LLM Integration

- Temperature: 0.3 (balanced creativity/consistency)
- System prompt: Emphasizes context usage
- JSON output: Structured for easy parsing
- Error handling: Falls back to RAG-enhanced plan

---

## 🔄 RAG Workflow Summary

1. **User Query** → Natural language input
2. **Intent Extraction** → Structured parameters
3. **Multi-Aspect RAG** → 5 different queries
4. **Context Assembly** → Rich formatted context
5. **Plan Generation**:
   - **With LLM**: Context-first generation
   - **Without LLM**: RAG-enhanced extraction
6. **Output** → Data-driven travel plan

---

## ✅ RAG Customization Checklist

- [x] Multiple aspect-specific retrieval queries
- [x] Rich context formatting with metadata
- [x] RAG-enhanced LLM prompts
- [x] Context extraction for fallback mode
- [x] Semantic search in fallback
- [x] RAG source tracking
- [x] Context-aware itinerary generation
- [x] Hotel recommendations from RAG
- [x] Activity distribution from context
- [x] Budget information from retrieved data

---

## 🚀 Future RAG Enhancements

Potential improvements:
- Hybrid search (dense + sparse vectors)
- Reranking of retrieved results
- Multi-hop retrieval (follow-up queries)
- Context compression for long contexts
- Query expansion and reformulation
- Temporal filtering (seasonal recommendations)
- User preference learning

---

**The entire project is now fully customized for RAG-based travel planning! 🎉**
