"""Expose the SemanticCart recommendation runtime through FastAPI."""

import json
import os
from contextlib import asynccontextmanager
from functools import partial
from pathlib import Path
from typing import Annotated, Any
from datetime import datetime

import pandas as pd
from fastapi import (
    FastAPI,
    HTTPException,
    Query,
    Request,
    status,
)
from semanticcart.events import (
    EventStore,
    EventType,
    InMemoryEventStore,
)
from pydantic import BaseModel, ConfigDict
from starlette.concurrency import run_in_threadpool

from semanticcart.recommendation_service import (
    RecommendationService,
)
from semanticcart.serving import ServingBundle
from semanticcart.postgres_events import (
    PostgresEventStore,
)
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles


DEMO_DIRECTORY = (
    Path(__file__).resolve().parent / "demo"
)

class ProductMetadata(BaseModel):
    """Describe catalogue metadata returned by recommendation endpoints."""

    model_config = ConfigDict(extra="forbid")

    item_id: str
    title: str
    main_category: str
    categories: str
    store: str
    price: float | None
    image_url: str
    popularity: int


class RecommendationItem(ProductMetadata):
    """Describe one ranked personalized recommendation."""

    user_id: str
    rank: int
    recommendation_score: float
    relevance_score: float | None
    collaborative_score: float | None
    semantic_score: float | None
    strategy: str


class SimilarProductItem(ProductMetadata):
    """Describe one semantically similar catalogue product."""

    rank: int
    similarity_score: float


class RecommendationResponse(BaseModel):
    """Wrap a personalized recommendation list with serving metadata."""

    user_id: str
    model_version: str
    count: int
    recommendations: list[RecommendationItem]


class SimilarProductsResponse(BaseModel):
    """Wrap similar products with their source and model version."""

    product_id: str
    model_version: str
    count: int
    similar_products: list[SimilarProductItem]


class HealthResponse(BaseModel):
    """Describe service readiness and loaded runtime backends."""

    status: str
    model_version: str
    event_store: str

class InteractionEventRequest(BaseModel):
    """Validate an interaction submitted by an API client."""

    model_config = ConfigDict(
        extra="forbid",
        str_strip_whitespace=True,
    )

    user_id: str
    item_id: str
    event_type: EventType
    occurred_at: datetime | None = None


class InteractionEventResponse(BaseModel):
    """Confirm an accepted and normalized interaction event."""

    status: str
    event_id: str
    user_id: str
    item_id: str
    event_type: EventType
    occurred_at: datetime


def _frame_records(
    frame: pd.DataFrame,
) -> list[dict[str, Any]]:
    """Convert a DataFrame into JSON-safe records with null values."""
    return json.loads(
        frame.to_json(
            orient="records",
            double_precision=15,
        )
    )


def _runtime_service(
    request: Request,
) -> RecommendationService:
    """Return the initialized service or report startup unavailability."""
    service = getattr(
        request.app.state,
        "recommendation_service",
        None,
    )

    if service is None:
        raise HTTPException(
            status_code=503,
            detail="Recommendation service is unavailable.",
        )

    return service


def _runtime_event_store(
    request: Request,
) -> EventStore:
    """Return the initialized online event store."""
    event_store = getattr(
        request.app.state,
        "event_store",
        None,
    )

    if event_store is None:
        raise HTTPException(
            status_code=503,
            detail="Event store is unavailable.",
        )

    return event_store


def _build_event_store(
    database_url: str | None = None,
) -> EventStore:
    """Build PostgreSQL storage when configured, otherwise memory."""
    configured_url = (
        database_url
        if database_url is not None
        else os.getenv(
            "SEMANTICCART_DATABASE_URL"
        )
    )

    if (
        configured_url is None
        or not configured_url.strip()
    ):
        return InMemoryEventStore()

    return PostgresEventStore(
        configured_url
    )


def create_app(
    serving_root: str | Path | None = None,
    service: RecommendationService | None = None,
    event_store: EventStore | None = None,
    database_url: str | None = None,
    verify_checksums: bool = True,
) -> FastAPI:
    """Create an application with an injected or startup-loaded service.

    Args:
        serving_root: Directory containing CURRENT and versioned bundles.
        service: Optional preloaded service, primarily for isolated tests.
        verify_checksums: Whether startup verifies all bundle checksums.
        event_store: Optional persistent or process-local event storage.
        database_url: Optional PostgreSQL URL used when no store is injected.

    Returns:
        A configured FastAPI application.
    """
    configured_root = serving_root or os.getenv(
        "SEMANTICCART_SERVING_ROOT",
        "data/artifacts/video_games_5core/serving",
    )
    configured_root = Path(configured_root)

    @asynccontextmanager
    async def lifespan(
        application: FastAPI,
    ):
        runtime = service

        if runtime is None:
            load_bundle = partial(
                ServingBundle.load,
                configured_root,
                verify_checksums=verify_checksums,
            )
            bundle = await run_in_threadpool(
                load_bundle
            )
            runtime = RecommendationService(bundle)

        runtime_store = (
            event_store
            if event_store is not None
            else _build_event_store(
                database_url
            )
        )

        await run_in_threadpool(
            runtime_store.open
        )

        application.state.recommendation_service = (
            runtime
        )
        application.state.event_store = (
            runtime_store
        )

        try:
            yield
        finally:
            application.state.event_store = None
            application.state.recommendation_service = (
                None
            )
            await run_in_threadpool(
                runtime_store.close
            )

    application = FastAPI(
        title="SemanticCart API",
        description=(
            "Hybrid, session-aware product recommendation service."
        ),
        version="0.1.0",
        lifespan=lifespan,
    )
    application.mount(
        "/demo/assets",
        StaticFiles(directory=DEMO_DIRECTORY),
        name="demo-assets",
    )

    @application.get(
        "/demo",
        response_class=FileResponse,
        include_in_schema=False,
    )
    async def demo() -> FileResponse:
        """Serve the interactive recommendation demo."""
        return FileResponse(
            DEMO_DIRECTORY / "index.html"
        )

    @application.get(
        "/health",
        response_model=HealthResponse,
        tags=["operations"],
    )
    async def health(
        request: Request,
    ) -> HealthResponse:
        runtime = _runtime_service(request)
        event_store = _runtime_event_store(
            request
        )

        return HealthResponse(
            status="ok",
            model_version=runtime.bundle.version,
            event_store=event_store.backend,
        )

    @application.get(
        "/model-info",
        response_model=dict[str, Any],
        tags=["operations"],
    )
    async def model_info(
        request: Request,
    ) -> dict[str, Any]:
        runtime = _runtime_service(request)
        return runtime.bundle.model_info()

    @application.get(
        "/recommendations/{user_id}",
        response_model=RecommendationResponse,
        tags=["recommendations"],
    )
    async def recommendations(
        request: Request,
        user_id: str,
        k: Annotated[int, Query(ge=1)] = 10,
        session_item_ids: Annotated[
            list[str] | None,
            Query(),
        ] = None,
    ) -> RecommendationResponse:
        runtime = _runtime_service(request)
        effective_session = session_item_ids

        if effective_session is None:
            event_store = _runtime_event_store(
                request
            )
            recent_items = (
                event_store.recent_item_ids(
                    user_id,
                    max_items=int(
                        runtime.ranking_config[
                            "session_length"
                        ]
                    ),
                )
            )
            effective_session = (
                recent_items or None
            )
            
        try:
            frame = await run_in_threadpool(
                runtime.recommend,
                user_id,
                k,
                effective_session,
            )
        except ValueError as error:
            raise HTTPException(
                status_code=422,
                detail=str(error),
            ) from error

        records = _frame_records(frame)

        return RecommendationResponse(
            user_id=user_id,
            model_version=runtime.bundle.version,
            count=len(records),
            recommendations=records,
        )

    @application.get(
        "/similar-products/{product_id}",
        response_model=SimilarProductsResponse,
        tags=["recommendations"],
    )
    async def similar_products(
        request: Request,
        product_id: str,
        k: Annotated[int, Query(ge=1)] = 10,
    ) -> SimilarProductsResponse:
        runtime = _runtime_service(request)

        try:
            frame = await run_in_threadpool(
                runtime.similar_products,
                product_id,
                k,
            )
        except ValueError as error:
            message = str(error)
            status_code = (
                404
                if message.startswith("Unknown product")
                else 422
            )
            raise HTTPException(
                status_code=status_code,
                detail=message,
            ) from error

        records = _frame_records(frame)

        return SimilarProductsResponse(
            product_id=product_id,
            model_version=runtime.bundle.version,
            count=len(records),
            similar_products=records,
        )
    @application.post(
        "/events",
        response_model=InteractionEventResponse,
        status_code=status.HTTP_201_CREATED,
        tags=["interactions"],
    )
    async def record_event(
        request: Request,
        payload: InteractionEventRequest,
    ) -> InteractionEventResponse:
        runtime = _runtime_service(request)
        event_store = _runtime_event_store(
            request
        )

        if (
            payload.item_id
            not in runtime.bundle.item_index.item_to_index
        ):
            raise HTTPException(
                status_code=404,
                detail=(
                    f"Unknown product: "
                    f"{payload.item_id}"
                ),
            )

        try:
            event = event_store.append(
                user_id=payload.user_id,
                item_id=payload.item_id,
                event_type=payload.event_type,
                occurred_at=payload.occurred_at,
            )
        except ValueError as error:
            raise HTTPException(
                status_code=422,
                detail=str(error),
            ) from error

        return InteractionEventResponse(
            status="accepted",
            event_id=event.event_id,
            user_id=event.user_id,
            item_id=event.item_id,
            event_type=event.event_type,
            occurred_at=event.occurred_at,
        )
        
    return application


app = create_app()