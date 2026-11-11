"""
Unit tests for feature_storefront.py.
Validates OpenAI configuration, vector similarity, intent routing, semantic RAG search,
catalog embedding, and Streamlit UI components with mocked external API calls.
"""

import os
from unittest.mock import MagicMock, patch

import pytest
import feature_storefront


def test_get_openai_api_key_from_env():
    """Verify OpenAI API key retrieval from environment variables."""
    with patch.dict(os.environ, {"OPENAI_API_KEY": "sk-test-env-key-12345"}):
        key = feature_storefront.get_openai_api_key()
        assert key == "sk-test-env-key-12345"


def test_get_openai_api_key_from_secrets():
    """Verify fallback to Streamlit secrets when environment variable is unset."""
    with patch.dict(os.environ, {}, clear=True), \
         patch.object(feature_storefront.st, "secrets", {"OPENAI_API_KEY": "sk-secret-key-67890"}):
        key = feature_storefront.get_openai_api_key()
        assert key == "sk-secret-key-67890"


def test_get_openai_api_key_missing():
    """Verify None is returned when no API key is configured."""
    with patch.dict(os.environ, {}, clear=True), \
         patch.object(feature_storefront.st, "secrets", {}):
        key = feature_storefront.get_openai_api_key()
        assert key is None


def test_cosine_similarity_identical_vectors():
    """Verify identical vectors yield a cosine similarity of 1.0."""
    v1 = [1.0, 2.0, 3.0]
    v2 = [1.0, 2.0, 3.0]
    sim = feature_storefront.cosine_similarity(v1, v2)
    assert sim == pytest.approx(1.0, abs=1e-5)


def test_cosine_similarity_orthogonal_vectors():
    """Verify orthogonal vectors yield a cosine similarity of 0.0."""
    v1 = [1.0, 0.0]
    v2 = [0.0, 1.0]
    sim = feature_storefront.cosine_similarity(v1, v2)
    assert sim == pytest.approx(0.0, abs=1e-5)


def test_cosine_similarity_opposite_vectors():
    """Verify opposite vectors yield a cosine similarity of -1.0."""
    v1 = [1.0, 1.0]
    v2 = [-1.0, -1.0]
    sim = feature_storefront.cosine_similarity(v1, v2)
    assert sim == pytest.approx(-1.0, abs=1e-5)


@pytest.mark.parametrize(
    "query,expected_intent",
    [
        ("I need a dress for a beach wedding next Saturday", "wedding"),
        ("What should I wear for an office tech conference?", "office"),
        ("Looking for a chic cocktail party outfit", "party"),
        ("Packing lightweight clothes for a summer VACATION", "vacation"),
        ("Breathable leggings for my morning workout", "workout"),
        ("Need something elegant for dinner date tonight", "date"),
        ("Formal suit recommendations for a job interview", "interview"),
    ],
)
def test_detect_event_intent_matches(query, expected_intent):
    """Verify deterministic intent router accurately maps occasion keywords."""
    assert feature_storefront.detect_event_intent(query) == expected_intent


def test_detect_event_intent_no_match():
    """Verify intent router returns None when no known occasion keyword is present."""
    assert feature_storefront.detect_event_intent("Just browsing casual blue cotton t-shirts") is None


def test_tools_schema_structure():
    """Verify the agent tool schema specifies required tool functions."""
    tool_names = [tool["function"]["name"] for tool in feature_storefront.TOOLS_SCHEMA]
    assert "search_catalog" in tool_names
    assert "generate_outfit" in tool_names
    assert "check_local_inventory" in tool_names
    assert "add_to_cart" in tool_names


def test_perform_rag_search():
    """Verify perform_rag_search computes embeddings and returns ranked item IDs."""
    mock_client = MagicMock()
    mock_emb_data = MagicMock()
    mock_emb_data.embedding = [1.0, 0.0]
    mock_response = MagicMock()
    mock_response.data = [mock_emb_data]
    mock_client.embeddings.create.return_value = mock_response

    catalog_embeddings = {
        "item_high": [0.99, 0.01],
        "item_mid": [0.70, 0.70],
        "item_low": [0.10, 0.99],
    }

    results = feature_storefront.perform_rag_search(
        mock_client, catalog_embeddings, "summer linen", top_k=2
    )

    mock_client.embeddings.create.assert_called_once_with(
        input="summer linen", model="text-embedding-3-small"
    )
    assert len(results) == 2
    assert results[0] == "item_high"
    assert results[1] == "item_mid"


def test_load_and_embed_catalog():
    """Verify catalog loading and batch embedding generation with mocked OpenAI client."""
    mock_df_data = {
        "id": ["101", "102"],
        "baseColour": ["Blue", "Red"],
        "masterCategory": ["Apparel", "Apparel"],
        "subCategory": ["Topwear", "Bottomwear"],
        "productDisplayName": ["Silk Blouse", "Chino Pants"],
        "usage": ["Formal", "Casual"],
        "gender": ["Women", "Men"],
    }
    mock_df = MagicMock()
    mock_df.head.return_value = mock_df
    mock_df.iterrows.return_value = [
        (0, {k: v[0] for k, v in mock_df_data.items()}),
        (1, {k: v[1] for k, v in mock_df_data.items()}),
    ]

    mock_client = MagicMock()
    mock_emb_resp = MagicMock()
    mock_item = MagicMock()
    mock_item.embedding = [0.1, 0.2, 0.3]
    mock_emb_resp.data = [mock_item]
    mock_client.embeddings.create.return_value = mock_emb_resp

    with patch.object(feature_storefront.pd, "read_csv", return_value=mock_df), \
         patch.object(feature_storefront, "OpenAI", return_value=mock_client):

        catalog, embeddings = feature_storefront.load_and_embed_catalog.__wrapped__(
            "sk-mock-key"
        ) if hasattr(feature_storefront.load_and_embed_catalog, "__wrapped__") else feature_storefront.load_and_embed_catalog("sk-mock-key")

        assert "101" in catalog
        assert "102" in catalog
        assert catalog["101"]["name"] == "Silk Blouse"
        assert catalog["102"]["name"] == "Chino Pants"
        assert "101" in embeddings
        assert "102" in embeddings


def test_render_retailnext_app_missing_api_key():
    """Verify app displays error and halts execution when API key is missing."""
    with patch.object(feature_storefront, "get_openai_api_key", return_value=None), \
         patch.object(feature_storefront.st, "stop", side_effect=SystemExit) as mock_stop:
        try:
            feature_storefront.render_retailnext_app()
        except SystemExit:
            pass

        feature_storefront.st.error.assert_called_once()
        mock_stop.assert_called_once()
