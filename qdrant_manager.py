"""
Qdrant Vector Database Manager for EchoMind
Handles storage and retrieval of phrase patterns for personalization
"""

import os
from datetime import datetime
from typing import Dict, List, Optional
from pathlib import Path

import google.generativeai as genai
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    VectorParams,
    PointStruct,
    FieldCondition,
    MatchValue,
    Filter,
)

# Qdrant setup
QDRANT_PATH = Path(__file__).resolve().parent / "qdrant_storage"
QDRANT_COLLECTION = "echomind_phrases"
EMBEDDING_MODEL = "models/embedding-001"
EMBEDDING_DIM = 768  # Gemini embedding dimension

# Initialize Qdrant client in local mode (no server needed)
client = QdrantClient(path=str(QDRANT_PATH))

# Counter for unique point IDs
_point_counter = {}


def _get_next_point_id(child_id: str) -> int:
    """Get next unique point ID for a child"""
    if child_id not in _point_counter:
        _point_counter[child_id] = 0
    _point_counter[child_id] += 1
    return _point_counter[child_id]


def init_qdrant() -> None:
    """Initialize Qdrant collection if it doesn't exist"""
    try:
        # Check if collection exists
        collections = client.get_collections().collections
        collection_names = [col.name for col in collections]

        if QDRANT_COLLECTION not in collection_names:
            # Create collection
            client.create_collection(
                collection_name=QDRANT_COLLECTION,
                vectors_config=VectorParams(
                    size=EMBEDDING_DIM,
                    distance=Distance.COSINE,
                ),
            )
            print(f"✓ Qdrant collection '{QDRANT_COLLECTION}' created")
        else:
            print(f"✓ Qdrant collection '{QDRANT_COLLECTION}' already exists")
    except Exception as e:
        print(f"Error initializing Qdrant: {e}")
        raise


def generate_embedding(text: str) -> Optional[List[float]]:
    """Generate embedding for a text using Gemini"""
    try:
        response = genai.embed_content(
            model=EMBEDDING_MODEL,
            content=text,
        )
        return response["embedding"]
    except Exception as e:
        print(f"Error generating embedding: {e}")
        return None


def store_phrase(
    child_id: str,
    category: str,
    phrase: str,
    context: Dict[str, str],
) -> bool:
    """Store a phrase selection with context in Qdrant for personalization"""
    try:
        # Build context string for embedding
        context_str = (
            f"Category: {category}. "
            f"Time of day: {context.get('time_of_day', 'unknown')}. "
            f"Day: {context.get('day_of_week', 'unknown')}. "
            f"Location: {context.get('location', 'unknown')}"
        )

        # Generate embedding of the context (skip if quota exceeded)
        embedding = generate_embedding(context_str)
        if not embedding:
            # Graceful degradation - store without embedding
            # Will still log the phrase for future use, just won't do similarity search
            embedding = [0.0] * EMBEDDING_DIM  # Dummy vector

        # Prepare payload (metadata)
        payload = {
            "child_id": child_id,
            "category": category,
            "phrase": phrase,
            "timestamp": datetime.now().isoformat(),
            "time_of_day": context.get("time_of_day", "unknown"),
            "day_of_week": context.get("day_of_week", "unknown"),
            "location": context.get("location", "unknown"),
            "context_str": context_str,
        }

        # Create point with unique ID
        point_id = _get_next_point_id(child_id)
        point = PointStruct(
            id=hash((child_id, point_id)) % (2**31),  # Convert to positive int
            vector=embedding,
            payload=payload,
        )

        # Upsert point into Qdrant
        client.upsert(
            collection_name=QDRANT_COLLECTION,
            points=[point],
        )

        print(f"✓ Stored phrase: '{phrase}' (Category: {category})")
        return True
    except Exception as e:
        print(f"Error storing phrase: {e}")
        return False


def get_similar_contexts(
    child_id: str,
    category: str,
    context: Dict[str, str],
    limit: int = 3,
) -> List[Dict]:
    """
    Retrieve similar past contexts from Qdrant.
    Returns the most similar phrase selections to help inform AI suggestions.
    Gracefully handles embedding quota errors.
    """
    try:
        # Build context string (same as in store_phrase)
        context_str = (
            f"Category: {category}. "
            f"Time of day: {context.get('time_of_day', 'unknown')}. "
            f"Day: {context.get('day_of_week', 'unknown')}. "
            f"Location: {context.get('location', 'unknown')}"
        )

        # Generate embedding of current context
        embedding = generate_embedding(context_str)
        if not embedding:
            # Embedding quota exceeded - just return empty, don't fail
            return []

        # Try to search - handle both client APIs
        try:
            search_filter = Filter(
                must=[
                    FieldCondition(
                        key="child_id",
                        match=MatchValue(value=child_id),
                    )
                ]
            )

            search_result = client.search(
                collection_name=QDRANT_COLLECTION,
                query_vector=embedding,
                query_filter=search_filter,
                limit=limit,
                with_payload=True,
            )

            # Format results
            similar = []
            for hit in search_result:
                similar.append({
                    "phrase": hit.payload.get("phrase"),
                    "category": hit.payload.get("category"),
                    "time_of_day": hit.payload.get("time_of_day"),
                    "similarity_score": hit.score,
                })
            return similar
        except AttributeError:
            # search() method not available in this Qdrant version, return empty
            return []
    except Exception as e:
        # Silently fail on any error - don't break the app
        return []


def get_top_phrases_in_category(child_id: str, category: str, limit: int = 5) -> List[str]:
    """Get the most frequently used phrases in a specific category for a child"""
    try:
        # Query all points for this child and category
        search_filter = Filter(
            must=[
                FieldCondition(
                    key="child_id",
                    match=MatchValue(value=child_id),
                ),
                FieldCondition(
                    key="category",
                    match=MatchValue(value=category),
                ),
            ]
        )

        # Try to get all points (not a real search, just filter)
        try:
            zero_vector = [0.0] * EMBEDDING_DIM
            search_result = client.search(
                collection_name=QDRANT_COLLECTION,
                query_vector=zero_vector,
                query_filter=search_filter,
                limit=limit * 5,  # Get more to deduplicate
                with_payload=True,
            )

            # Count occurrences and get top phrases
            phrase_counts = {}
            for hit in search_result:
                phrase = hit.payload.get("phrase", "")
                if phrase:
                    phrase_counts[phrase] = phrase_counts.get(phrase, 0) + 1

            # Sort by frequency and return top phrases
            top_phrases = sorted(phrase_counts.items(), key=lambda x: x[1], reverse=True)
            return [phrase for phrase, count in top_phrases[:limit]]
        except AttributeError:
            # search() method not available, graceful degradation
            return []
    except Exception as e:
        # Silently fail - don't break the app
        return []


def get_personalization_context(
    child_id: str,
    category: str,
    context: Dict[str, str],
) -> str:
    """
    Build a personalization context string based on child's history.
    This will be added to the Gemini prompt to personalize suggestions.
    """
    try:
        # Get similar contexts
        similar = get_similar_contexts(child_id, category, context, limit=3)

        # Get top phrases in category
        top_phrases = get_top_phrases_in_category(child_id, category, limit=3)

        # Build personalization string
        personalization = ""

        if similar:
            phrases_from_similar = [s["phrase"] for s in similar if s.get("phrase")]
            if phrases_from_similar:
                personalization += f"In similar situations, this child has said: {', '.join(phrases_from_similar)}. "

        if top_phrases:
            personalization += f"This child frequently uses these phrases in this category: {', '.join(top_phrases)}. "

        if not personalization:
            personalization = "This is the child's first time in this category or context. Suggest clear, simple phrases based on the category."

        return personalization
    except Exception as e:
        print(f"Error building personalization context: {e}")
        return ""
