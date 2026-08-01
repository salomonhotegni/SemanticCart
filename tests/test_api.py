from types import SimpleNamespace

import pandas as pd
import pytest
from fastapi.testclient import TestClient

from semanticcart.api import create_app
from semanticcart.events import InMemoryEventStore


class StubBundle:
    """Provide the bundle attributes used by the API."""

    version = "semanticcart-test-api"
    item_index = SimpleNamespace(
        item_to_index={
            "a": 0,
            "b": 1,
            "c": 2,
        }
    )

    def model_info(self) -> dict:
        """Return representative model metadata."""
        return {
            "model_version": self.version,
            "model": "test-hybrid",
            "dataset": "synthetic",
            "users": 2,
            "semantic_items": 3,
        }


class StubRecommendationService:
    """Provide deterministic recommendations for API boundary tests."""

    def __init__(self) -> None:
        self.bundle = StubBundle()
        self.ranking_config = {
            "session_length": 1,
        }
        self.last_session: list[str] | None = None

    def recommend(
        self,
        user_id: str,
        k: int = 10,
        session_item_ids: list[str] | None = None,
    ) -> pd.DataFrame:
        """Return one deterministic recommendation."""
        if k > 2:
            raise ValueError(
                "k must be between 1 and 2."
            )

        self.last_session = (
            None
            if session_item_ids is None
            else list(session_item_ids)
        )
        strategy = (
            "anonymous_session_semantic"
            if self.last_session
            else "popularity_fallback"
        )

        return pd.DataFrame(
            [
                {
                    "user_id": user_id,
                    "item_id": "c",
                    "rank": 1,
                    "recommendation_score": 0.9,
                    "relevance_score": 0.8,
                    "collaborative_score": None,
                    "semantic_score": 0.8,
                    "strategy": strategy,
                    "title": "Product C",
                    "main_category": "Games",
                    "categories": "Video Games",
                    "store": "Test Store",
                    "price": 29.99,
                    "image_url": "https://example.com/c.jpg",
                    "popularity": 5,
                }
            ]
        ).head(k)

    def similar_products(
        self,
        item_id: str,
        k: int = 10,
    ) -> pd.DataFrame:
        """Return one deterministic similar product."""
        if item_id not in {"a", "b", "c"}:
            raise ValueError(
                f"Unknown product: {item_id}"
            )
        if k > 2:
            raise ValueError(
                "k must be between 1 and 2."
            )

        return pd.DataFrame(
            [
                {
                    "item_id": "b",
                    "rank": 1,
                    "similarity_score": 0.75,
                    "title": "Product B",
                    "main_category": "Games",
                    "categories": "Video Games",
                    "store": "Test Store",
                    "price": None,
                    "image_url": "",
                    "popularity": 3,
                }
            ]
        ).head(k)


@pytest.fixture
def api_runtime():
    """Create a test client with injected lightweight dependencies."""
    service = StubRecommendationService()
    event_store = InMemoryEventStore()
    application = create_app(
        service=service,
        event_store=event_store,
    )

    with TestClient(application) as client:
        yield client, service, event_store


def test_health_reports_loaded_version(
    api_runtime,
) -> None:
    client, _, _ = api_runtime

    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {
        "status": "ok",
        "model_version": "semanticcart-test-api",
    }


def test_model_info_returns_bundle_metadata(
    api_runtime,
) -> None:
    client, _, _ = api_runtime

    response = client.get("/model-info")

    assert response.status_code == 200
    assert response.json()["dataset"] == "synthetic"
    assert (
        response.json()["model_version"]
        == "semanticcart-test-api"
    )


def test_returns_popularity_fallback_without_session(
    api_runtime,
) -> None:
    client, service, _ = api_runtime

    response = client.get(
        "/recommendations/new-user",
        params={"k": 1},
    )

    assert response.status_code == 200

    body = response.json()

    assert body["user_id"] == "new-user"
    assert body["count"] == 1
    assert (
        body["recommendations"][0]["strategy"]
        == "popularity_fallback"
    )
    assert service.last_session is None


def test_event_drives_session_recommendation(
    api_runtime,
) -> None:
    client, service, event_store = api_runtime

    event_response = client.post(
        "/events",
        json={
            "user_id": "  new-user  ",
            "item_id": "  a  ",
            "event_type": "view",
        },
    )

    assert event_response.status_code == 201
    assert event_response.json()["status"] == "accepted"
    assert event_response.json()["user_id"] == "new-user"
    assert event_response.json()["item_id"] == "a"
    assert event_store.event_count("new-user") == 1

    recommendation_response = client.get(
        "/recommendations/new-user",
        params={"k": 1},
    )

    assert recommendation_response.status_code == 200
    assert service.last_session == ["a"]
    assert (
        recommendation_response.json()
        ["recommendations"][0]["strategy"]
        == "anonymous_session_semantic"
    )


def test_explicit_session_overrides_stored_events(
    api_runtime,
) -> None:
    client, service, _ = api_runtime

    client.post(
        "/events",
        json={
            "user_id": "new-user",
            "item_id": "a",
            "event_type": "view",
        },
    )

    response = client.get(
        "/recommendations/new-user",
        params={
            "k": 1,
            "session_item_ids": "b",
        },
    )

    assert response.status_code == 200
    assert service.last_session == ["b"]


def test_rejects_event_for_unknown_product(
    api_runtime,
) -> None:
    client, _, event_store = api_runtime

    response = client.post(
        "/events",
        json={
            "user_id": "new-user",
            "item_id": "missing",
            "event_type": "view",
        },
    )

    assert response.status_code == 404
    assert event_store.event_count("new-user") == 0


def test_rejects_invalid_event_type(
    api_runtime,
) -> None:
    client, _, _ = api_runtime

    response = client.post(
        "/events",
        json={
            "user_id": "new-user",
            "item_id": "a",
            "event_type": "wishlist",
        },
    )

    assert response.status_code == 422


def test_returns_similar_products(
    api_runtime,
) -> None:
    client, _, _ = api_runtime

    response = client.get(
        "/similar-products/a",
        params={"k": 1},
    )

    assert response.status_code == 200

    body = response.json()

    assert body["product_id"] == "a"
    assert body["count"] == 1
    assert body["similar_products"][0]["item_id"] == "b"


def test_unknown_similar_product_returns_404(
    api_runtime,
) -> None:
    client, _, _ = api_runtime

    response = client.get(
        "/similar-products/missing",
        params={"k": 1},
    )

    assert response.status_code == 404
    assert (
        response.json()["detail"]
        == "Unknown product: missing"
    )


@pytest.mark.parametrize(
    "path",
    [
        "/recommendations/user-1?k=3",
        "/similar-products/a?k=3",
    ],
)
def test_runtime_depth_errors_return_422(
    api_runtime,
    path: str,
) -> None:
    client, _, _ = api_runtime

    response = client.get(path)

    assert response.status_code == 422
    assert "k must be between" in (
        response.json()["detail"]
    )


@pytest.mark.parametrize(
    "path",
    [
        "/recommendations/user-1?k=0",
        "/similar-products/a?k=0",
    ],
)
def test_query_validation_rejects_nonpositive_depth(
    api_runtime,
    path: str,
) -> None:
    client, _, _ = api_runtime

    response = client.get(path)

    assert response.status_code == 422