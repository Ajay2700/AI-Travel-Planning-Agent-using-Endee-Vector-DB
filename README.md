# AI Travel Planning Agent using Endee Vector Database

[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.30.0-red.svg)](https://streamlit.io/)
[![Endee](https://img.shields.io/badge/Endee-Vector%20DB-green.svg)](https://github.com/endee-io/endee)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

A production-ready **AI-powered travel planning system** that uses **Endee** as the core vector database for semantic search, RAG (Retrieval-Augmented Generation), and intelligent recommendations. This project demonstrates practical AI/ML applications including agentic workflows, semantic search, and context-aware travel planning.

---

## 📋 Table of Contents

- [Project Overview](#project-overview)
- [Problem Statement](#problem-statement)
- [Core Features & Use Cases](#core-features--use-cases)
- [System Design & Architecture](#system-design--architecture)
- [How Endee is Used](#how-endee-is-used)
- [Technical Approach](#technical-approach)
- [Setup & Installation](#setup--installation)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [Example Queries](#example-queries)
- [Future Improvements](#future-improvements)

---

## 🎯 Project Overview

This project implements a **production-ready AI Agent system** for personalized, comprehensive travel planning using **Endee Vector Database** as the core retrieval engine. The system takes natural language queries like:

> *"Plan a 3-day Goa trip under ₹10,000"*

And generates **detailed, structured travel plans** with:
- **Day-by-day itineraries** with timing and transport details
- **Budget breakdowns** with cost estimates
- **Hotel recommendations** based on vector search
- **Activity suggestions** tailored to preferences
- **Travel tips** and recommendations

The system leverages:
- **Semantic Search** via Endee for finding relevant travel information
- **RAG (Retrieval-Augmented Generation)** for context-aware plan generation
- **Agentic AI Workflows** for multi-step reasoning and planning
- **Recommendation Engine** for personalized suggestions

---

## 🎯 Problem Statement

### Current Challenges in Travel Planning

Planning a trip, especially for budget-conscious travelers in India, typically requires:

1. **Manual Research Overload**
   - Searching multiple travel websites (booking sites, blogs, forums)
   - Comparing prices, locations, and reviews across platforms
   - Spending hours to gather fragmented information

2. **Information Fragmentation**
   - Hotels, activities, food, and transport info scattered across sources
   - Difficulty in finding budget-appropriate options
   - Lack of comprehensive, structured planning tools

3. **Time-Consuming Process**
   - Manual itinerary creation
   - Budget calculation and tracking
   - Balancing preferences with constraints

4. **Limited Personalization**
   - Generic travel guides don't account for individual preferences
   - No intelligent recommendations based on user goals
   - Missing context-aware suggestions

### Solution: AI-Powered Travel Planning Agent

This project addresses these challenges by building an **intelligent AI agent** that:

- ✅ **Understands natural language** travel goals
- ✅ **Retrieves relevant information** using semantic search over a curated travel database
- ✅ **Generates comprehensive plans** with detailed itineraries, budgets, and recommendations
- ✅ **Provides personalized suggestions** based on user preferences and constraints
- ✅ **Uses vector search** for fast, accurate information retrieval

---

## 🚀 Core Features & Use Cases

This project demonstrates **four key AI/ML use cases** where vector search is core:

### 1. **Semantic Search** 🔍

**Implementation:**
- Uses **Endee vector database** to store travel document embeddings
- Performs **cosine similarity search** to find relevant destinations, hotels, activities
- Supports **multi-aspect queries** (destinations, accommodations, food, activities)

**Example:**
```python
# Query: "beach destinations with budget hotels"
# System finds: Goa, Mumbai, Andaman (semantically similar)
# Returns: Relevant documents with similarity scores
```

**Location:** `retriever/vector_store.py` → `search()`, `agent/tools.py` → `retrieve_travel_data()`

---

### 2. **RAG (Retrieval-Augmented Generation)** 📚

**Implementation:**
- **Multi-aspect retrieval**: 5 different queries per request (destinations, hotels, food, activities, general)
- **Rich context formatting**: All metadata included (prices, areas, activities, scores)
- **RAG-enhanced prompts**: LLM explicitly instructed to use retrieved context as primary source
- **RAG-aware fallback**: Uses retrieved data even without LLM

**Workflow:**
```
User Query → Intent Extraction → Multi-Aspect RAG Retrieval → 
Rich Context Assembly → LLM/Fallback Plan Generation → Structured Output
```

**Location:** `agent/tools.py` → `retrieve_multiple_aspects()`, `agent/planner.py` → `_generate_plan()`

---

### 3. **Recommendations** 💡

**Implementation:**
- **Content-based recommendations**: Based on destination characteristics and user preferences
- **Vector similarity recommendations**: Find similar destinations using embeddings
- **Context-aware suggestions**: Best time to visit, target audience, what to avoid

**Features:**
- Destination recommendations based on preferences
- Hotel recommendations with price ranges and areas
- Activity recommendations matching user interests
- Travel tips and best practices

**Location:** `agent/planner.py` → `_extract_recommendations()`, `data/travel_data.json` → recommendation entries

---

### 4. **Agentic AI Workflows** 🤖

**Implementation:**
- **Multi-step reasoning**: Intent extraction → Retrieval → Planning
- **Tool-based architecture**: Modular tools for retrieval, budget estimation
- **Memory management**: Conversation history tracking
- **Error handling**: Graceful fallbacks at each step

**Workflow:**
```
Step 1: Extract Intent
  ├─→ Parse destination, budget, duration, preferences
  └─→ Use LLM or heuristic extraction

Step 2: Retrieve Context
  ├─→ Multi-aspect RAG queries
  ├─→ Semantic search via Endee
  └─→ Format rich context

Step 3: Generate Plan
  ├─→ Use retrieved context
  ├─→ Estimate budget
  ├─→ Create detailed itinerary
  └─→ Extract recommendations
```

**Location:** `agent/planner.py` → `plan_trip()`, `agent/tools.py` → Tool implementations

---

## 🏗️ System Design & Architecture

### High-Level Architecture

```
┌─────────────────┐
│  Streamlit UI   │  User Interface
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ TravelPlanner   │  Agent Orchestration
│     Agent       │  (Multi-step reasoning)
└────────┬────────┘
         │
    ┌────┴────┐
    │         │
    ▼         ▼
┌─────────┐ ┌──────────────┐
│  Tools  │ │   Memory     │
│         │ │  (History)   │
└────┬────┘ └──────────────┘
     │
     ├─→ Retrieval Tool
     │   └─→ EndeeVectorStore
     │       └─→ Endee API (HTTP)
     │
     └─→ Budget Estimator
         └─→ Rule-based calculation

┌─────────────────┐
│  Embedder       │  Sentence Transformers
│  (all-MiniLM)   │  Text → Embeddings
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Endee Vector   │  Vector Database
│     Database    │  (Port 8080)
└─────────────────┘
```

### Component Details

#### 1. **Agent Layer** (`agent/`)
- **`planner.py`**: Main orchestration agent
  - Intent extraction (LLM/heuristic)
  - Context retrieval coordination
  - Plan generation (LLM/fallback)
  - Recommendation extraction
  
- **`tools.py`**: Agent tools
  - `retrieve_travel_data()`: Semantic search
  - `retrieve_multiple_aspects()`: Multi-aspect RAG
  - `estimate_budget()`: Budget calculation
  - `format_context()`: Rich context formatting

- **`memory.py`**: Conversation history

#### 2. **Retriever Layer** (`retriever/`)
- **`embedder.py`**: Sentence-transformers wrapper
  - Model: `all-MiniLM-L6-v2` (384 dimensions)
  - Text → Vector embeddings
  
- **`vector_store.py`**: Endee integration
  - `add_documents()`: Store embeddings + metadata
  - `search()`: Vector similarity search
  - Fallback mode for offline operation

#### 3. **Utils Layer** (`utils/`)
- **`config.py`**: Environment variables, logging
- **`prompts.py`**: LLM prompt templates

#### 4. **Data Layer** (`data/`)
- **`travel_data.json`**: Curated travel dataset
  - Destinations, hotels, food, activities
  - Detailed itineraries with timing
  - Transport information
  - Recommendations

#### 5. **UI Layer** (`app.py`)
- Streamlit interface
- Plan visualization
- Detailed journey display

---

## 🔧 How Endee is Used

Endee serves as the **core vector database** for all semantic search and RAG operations in this project.

### 1. **Data Ingestion** (`scripts/ingest.py`)

```python
# Load travel documents
docs = load_travel_data("data/travel_data.json")

# Generate embeddings
embeddings = embedder.embed_batch(texts)

# Store in Endee
vector_store.add_documents(docs, embeddings)
```

**Process:**
1. Load travel data from JSON
2. Generate embeddings using sentence-transformers
3. Store vectors + metadata in Endee index `travel_plans`
4. Endee handles vector storage and indexing

### 2. **Semantic Search** (`retriever/vector_store.py`)

```python
# Search for relevant documents
query_vector = embedder.embed_text("Goa beach hotels budget")
results = vector_store.search(query_vector, top_k=5)

# Returns: List of documents with similarity scores
```

**Endee Operations:**
- **Index Creation**: `POST /api/v1/index/create`
  - Name: `travel_plans`
  - Dimension: `384` (matches embedding model)
  - Metric: `cosine` similarity

- **Vector Insertion**: `POST /api/v1/index/{index_name}/vectors`
  - Embeddings + document metadata
  - Automatic indexing for fast search

- **Similarity Search**: `POST /api/v1/index/{index_name}/search`
  - Query vector → Top-K similar documents
  - Returns relevance scores

### 3. **RAG Integration** (`agent/tools.py`)

```python
# Multi-aspect retrieval
aspects = tools.retrieve_multiple_aspects(intent)
# Returns: {
#   "destinations": [...],
#   "accommodations": [...],
#   "food": [...],
#   "activities": [...],
#   "general": [...]
# }

# Format for LLM context
context = tools.format_context(all_results)
```

**Endee's Role:**
- Stores all travel document embeddings
- Enables fast semantic search across multiple aspects
- Provides relevance scores for context prioritization
- Supports filtering and metadata queries

### 4. **Fallback Mode**

When Endee is unavailable, the system:
- Uses in-memory storage
- Performs cosine similarity search locally
- Maintains RAG quality with semantic search
- Gracefully degrades without service interruption

**Location:** `retriever/vector_store.py` → `_fallback_search()`

---

## 🛠️ Technical Approach

### Technology Stack

| Component | Technology | Purpose |
|-----------|-----------|---------|
| **Vector DB** | Endee | Core semantic search & storage |
| **Embeddings** | sentence-transformers | Text → Vector conversion |
| **LLM** | OpenAI API (optional) | Intent extraction & plan generation |
| **UI Framework** | Streamlit | Interactive web interface |
| **Language** | Python 3.10+ | Implementation |
| **Containerization** | Docker | Endee deployment |

### Key Design Decisions

1. **Modular Architecture**
   - Separation of concerns (agent, retriever, utils)
   - Tool-based agent design
   - Easy to extend and maintain

2. **RAG-First Approach**
   - All plans generated from retrieved context
   - LLM prompts prioritize context usage
   - Fallback mode maintains RAG quality

3. **Multi-Aspect Retrieval**
   - Different queries for different information needs
   - Comprehensive context assembly
   - Better plan quality

4. **Graceful Degradation**
   - Works without Endee (fallback mode)
   - Works without LLM (rule-based planning)
   - Always produces usable output

5. **Production-Ready**
   - Error handling at all levels
   - Comprehensive logging
   - Clean, documented code

---

## 📦 Setup & Installation

### Prerequisites

- **Python 3.10+**
- **Docker** (for Endee) or access to Endee service
- **Git** (for cloning)

### Step 1: Fork and Star Endee Repository

1. Visit https://github.com/endee-io/endee
2. Click **⭐ Star** (company requirement)
3. Click **🍴 Fork** to create your copy

### Step 2: Clone This Project

```bash
git clone <your-repo-url>
cd ai_travel_agent  # or your project directory name
```

### Step 3: Set Up Python Environment

```bash
# Create virtual environment
python -m venv venv

# Activate virtual environment
# Windows:
venv\Scripts\Activate.ps1
# Linux/Mac:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### Step 4: Start Endee Service

**Option A: Docker (Recommended)**

```bash
# Start Endee using Docker Compose
docker compose -f docker-compose.endee.yml up -d

# Verify it's running
docker ps | grep endee
# Should show: endee-server container running on port 8080
```

**Option B: Build from Source**

```bash
# Clone your forked Endee repository
git clone https://github.com/YOUR_USERNAME/endee
cd endee

# Build Endee
chmod +x ./install.sh
./install.sh --release --avx2

# Run Endee
chmod +x ./run.sh
./run.sh
```

Endee will be available at `http://localhost:8080`

### Step 5: Configure Environment Variables

Create a `.env` file in the project root:

```env
# Endee Configuration
ENDEE_BASE_URL=http://localhost:8080
ENDEE_INDEX_NAME=travel_plans
ENDEE_AUTH_TOKEN=  # Optional: leave empty for no authentication

# OpenAI Configuration (Optional - enables LLM-based planning)
OPENAI_API_KEY=your_openai_api_key_here
OPENAI_MODEL=gpt-4o-mini  # or gpt-4, gpt-3.5-turbo

# Logging
LOG_LEVEL=INFO
```

**Note:** The system works without `OPENAI_API_KEY` using rule-based planning.

### Step 6: Ingest Travel Data

```bash
# Make sure virtual environment is activated
python scripts/ingest.py
```

**Expected Output:**
```
INFO - Loaded 20 travel documents from data/travel_data.json
INFO - Embedder initialized with model 'all-MiniLM-L6-v2'
INFO - Initializing EndeeVectorStore base_url=http://localhost:8080 index=travel_plans
INFO - Created Endee index 'travel_plans'
INFO - Added 20 documents to Endee index 'travel_plans'
INFO - Ingestion finished successfully
```

If Endee is not available, it will use fallback mode and still work.

### Step 7: Run the Application

```bash
streamlit run app.py
```

The app will open in your browser at `http://localhost:8501`

---

## 🎮 Usage

### Basic Usage

1. **Open the Streamlit app** (http://localhost:8501)

2. **Enter a travel query**, for example:
   - "Plan a 3-day Goa trip under ₹10,000"
   - "Weekend budget trip to Jaipur from Delhi"
   - "4-day family trip to Agra and Delhi with focus on history"

3. **Click "Generate travel plan"**

4. **View your comprehensive travel plan**:
   - Destination and budget breakdown
   - **Detailed day-by-day itinerary** with:
     - Morning, afternoon, evening activities
     - Transport recommendations
     - Estimated daily costs
   - Hotel recommendations with prices
   - Travel tips
   - Personalized recommendations

### Example Query & Output

**Input:**
```
Plan a 3-day Goa trip under ₹10,000
```

**Output Structure:**
```json
{
  "destination": "Goa",
  "budget": {
    "currency": "INR",
    "daily_cost": 2000,
    "total_cost": 6000,
    "breakdown": {
      "accommodation": 1200,
      "food": 500,
      "transport": 300
    }
  },
  "itinerary": [
    {
      "day": 1,
      "title": "Day 1 in Goa",
      "timing": {
        "morning": ["Calangute Beach (9 AM)", "Water sports"],
        "afternoon": ["Baga Beach (1-4 PM)", "Beach shack lunch"],
        "evening": ["Anjuna Flea Market (5-8 PM)", "Nightlife"]
      },
      "transport": "Scooter rental (₹300-₹500/day)",
      "estimated_cost": 2000
    },
    ...
  ],
  "hotels": [
    {
      "name": "Budget Stay",
      "approx_price_per_night": "₹800–₹1,500",
      "area": "North Goa",
      "rating": 4.0
    }
  ],
  "tips": [...],
  "recommendations": {
    "best_time_to_visit": "October to March",
    "target_audience": ["beach lovers", "budget travelers"],
    "best_for": ["water sports", "nightlife"]
  }
}
```

---

## 📁 Project Structure

```
ai_travel_agent/
│
├── app.py                      # Streamlit UI entrypoint
│
├── agent/                      # Agent layer
│   ├── planner.py             # Main agent orchestration
│   ├── tools.py                # Retrieval & budget tools
│   └── memory.py               # Conversation memory
│
├── retriever/                  # Retrieval layer
│   ├── embedder.py            # Sentence-transformers wrapper
│   └── vector_store.py        # Endee vector DB integration
│
├── scripts/
│   └── ingest.py              # Data ingestion into Endee
│
├── data/
│   └── travel_data.json       # Travel dataset (20+ documents)
│
├── utils/
│   ├── config.py              # Configuration & logging
│   └── prompts.py             # LLM prompt templates
│
├── docker-compose.endee.yml   # Docker config for Endee
├── requirements.txt           # Python dependencies
├── README.md                  # This file
└── RAG_CUSTOMIZATION.md       # RAG implementation details
```

---

## 🔍 Example Queries

Try these example queries to see the system in action:

1. **Budget Trip:**
   ```
   Plan a 3-day Goa trip under ₹10,000
   ```

2. **Heritage Focus:**
   ```
   4-day family trip to Agra and Delhi with focus on history
   ```

3. **Weekend Getaway:**
   ```
   Weekend budget trip to Jaipur from Delhi
   ```

4. **Beach Vacation:**
   ```
   Plan a 5-day beach vacation in Goa with water sports
   ```

5. **Cultural Tour:**
   ```
   Cultural tour of Rajasthan: Jaipur, Udaipur, Jodhpur in 7 days
   ```

---

## 🚀 Future Improvements

- [ ] **Hybrid Search**: Combine dense (vector) + sparse (keyword) search
- [ ] **Reranking**: Improve result quality with cross-encoder reranking
- [ ] **Multi-hop Retrieval**: Follow-up queries based on initial results
- [ ] **Real-time Data**: Integrate live APIs for flights, hotels, weather
- [ ] **User Profiles**: Save preferences and travel history
- [ ] **Multi-language Support**: Queries and outputs in multiple languages
- [ ] **Interactive Refinement**: "Make it more budget-friendly" type queries
- [ ] **Deployment**: Dockerfile, CI/CD, cloud hosting recipes
- [ ] **Advanced Filtering**: Endee metadata filters for precise queries
- [ ] **Analytics**: Track popular destinations and query patterns

---

## 📊 Performance & Scalability

- **Vector Search**: Sub-millisecond query latency (Endee)
- **Embedding Generation**: ~50ms per query (CPU)
- **Plan Generation**: 2-5 seconds (with LLM), <1s (fallback)
- **Scalability**: Endee handles millions of vectors efficiently
- **Fallback Mode**: Works offline with in-memory search

---

## 🤝 Contributing

This is an internship evaluation project. For contributions:

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Submit a pull request

---

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

---

## 🙏 Acknowledgments

- **Endee** (https://github.com/endee-io/endee) - High-performance vector database
- **sentence-transformers** - Embedding models
- **Streamlit** - UI framework
- **OpenAI** - LLM API (optional)

---

## 📞 Support

For issues or questions:
- Check the logs in terminal/console
- Verify Endee is running: `docker ps | grep endee`
- Test Endee API: `curl http://localhost:8080/api/v1/health`
- Review `RAG_CUSTOMIZATION.md` for RAG details

---

**Built with ❤️ using Endee Vector Database**