"use strict";

const elements = {
  serviceState: document.querySelector("#service-state"),
  serviceStatus: document.querySelector("#service-status"),
  modelVersion: document.querySelector("#model-version"),
  modelDataset: document.querySelector("#model-dataset"),
  modelItems: document.querySelector("#model-items"),
  modeButtons: document.querySelectorAll("[data-mode]"),
  panels: document.querySelectorAll("[data-panel]"),
  recommendationForm: document.querySelector("#recommendation-form"),
  similarForm: document.querySelector("#similar-form"),
  eventForm: document.querySelector("#event-form"),
  eventReceipt: document.querySelector("#event-receipt"),
  resultsContext: document.querySelector("#results-context"),
  resultsTitle: document.querySelector("#results-title"),
  resultCount: document.querySelector("#result-count"),
  loadingState: document.querySelector("#loading-state"),
  emptyState: document.querySelector("#empty-state"),
  errorState: document.querySelector("#error-state"),
  productGrid: document.querySelector("#product-grid"),
  productTemplate: document.querySelector("#product-card-template"),
  toast: document.querySelector("#toast"),
};

const modeCopy = {
  recommendations: {
    context: "Personalized ranking",
    title: "Recommended products",
  },
  similar: {
    context: "Semantic retrieval",
    title: "Similar products",
  },
  event: {
    context: "Online interaction",
    title: "Current product results",
  },
};

let toastTimeout;

async function apiRequest(path, options = {}) {
  const response = await fetch(path, {
    ...options,
    headers: {
      Accept: "application/json",
      ...options.headers,
    },
  });

  const contentType = response.headers.get("content-type") || "";
  const body = contentType.includes("application/json")
    ? await response.json()
    : null;

  if (!response.ok) {
    const detail = body?.detail;
    const message = Array.isArray(detail)
      ? detail.map((entry) => entry.msg).join("; ")
      : detail;

    throw new Error(
      message || `Request failed with status ${response.status}.`,
    );
  }

  return body;
}

function setMode(mode, focusField = true) {
  elements.modeButtons.forEach((button) => {
    const active = button.dataset.mode === mode;
    button.classList.toggle("is-active", active);
    button.setAttribute("aria-pressed", String(active));
  });

  elements.panels.forEach((panel) => {
    panel.hidden = panel.dataset.panel !== mode;
  });

  elements.resultsContext.textContent = modeCopy[mode].context;
  elements.resultsTitle.textContent = modeCopy[mode].title;

  if (focusField) {
    const panel = document.querySelector(`[data-panel="${mode}"]`);
    panel?.querySelector("input, select, textarea")?.focus();
  }
}

function setServiceState(status, text) {
  elements.serviceState.classList.remove("is-online", "is-offline");
  elements.serviceState.classList.add(`is-${status}`);
  elements.serviceStatus.textContent = text;
}

function setResultsState(state, message = "") {
  elements.loadingState.hidden = state !== "loading";
  elements.emptyState.hidden = state !== "empty";
  elements.errorState.hidden = state !== "error";
  elements.productGrid.hidden = state !== "results";

  if (state === "loading") {
    elements.loadingState.textContent = message || "Loading products";
  }

  if (state === "empty") {
    elements.emptyState.textContent = message || "No products found";
  }

  if (state === "error") {
    elements.errorState.textContent = message;
  }
}

function setFormBusy(form, busy) {
  form.setAttribute("aria-busy", String(busy));
  form.querySelectorAll("button, input, select, textarea").forEach(
    (control) => {
      control.disabled = busy;
    },
  );
}

function showToast(message, isError = false) {
  window.clearTimeout(toastTimeout);
  elements.toast.textContent = message;
  elements.toast.classList.toggle("is-error", isError);
  elements.toast.hidden = false;

  toastTimeout = window.setTimeout(() => {
    elements.toast.hidden = true;
  }, 3500);
}

function humanize(value) {
  if (!value) {
    return "";
  }

  const text = String(value).replaceAll("_", " ");
  return text.charAt(0).toUpperCase() + text.slice(1);
}

function formatScore(value) {
  const number = Number(value);
  return Number.isFinite(number) ? number.toFixed(4) : "Unavailable";
}

function formatPrice(value) {
  if (value === null || value === undefined || value === "") {
    return "Not listed";
  }

  const number = Number(value);

  if (!Number.isFinite(number)) {
    return "Not listed";
  }

  return new Intl.NumberFormat("en-US", {
    style: "currency",
    currency: "USD",
  }).format(number);
}

function safeImageUrl(value) {
  if (!value) {
    return null;
  }

  try {
    const url = new URL(value, window.location.href);
    return ["http:", "https:"].includes(url.protocol)
      ? url.href
      : null;
  } catch {
    return null;
  }
}

function currentUserId() {
  const data = new FormData(elements.recommendationForm);
  return String(data.get("user_id") || "").trim() || "demo-user";
}

async function recordEvent(
  userId,
  itemId,
  eventType = "view",
) {
  return apiRequest("/events", {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
    },
    body: JSON.stringify({
      user_id: userId,
      item_id: itemId,
      event_type: eventType,
    }),
  });
}

function renderProducts(products, kind) {
  elements.productGrid.replaceChildren();

  if (!Array.isArray(products) || products.length === 0) {
    elements.resultCount.textContent = "0 products";
    setResultsState("empty", "No matching products");
    return;
  }

  products.forEach((product) => {
    const fragment = elements.productTemplate.content.cloneNode(true);
    const card = fragment.querySelector(".product-card");
    const imageFrame = fragment.querySelector(".product-image");
    const image = fragment.querySelector("img");
    const title = product.title || product.item_id || "Unknown product";
    const score = kind === "recommendations"
      ? product.recommendation_score
      : product.similarity_score;
    const strategy = kind === "recommendations"
      ? humanize(product.strategy)
      : "Semantic match";

    card.dataset.itemId = product.item_id;
    fragment.querySelector(".product-rank").textContent =
      `#${product.rank} · ${strategy}`;
    fragment.querySelector(".product-title").textContent = title;
    fragment.querySelector(".product-category").textContent =
      product.main_category || product.categories || "Uncategorized";
    fragment.querySelector(".product-store").textContent =
      product.store || "Store unavailable";
    fragment.querySelector(".product-score").textContent =
      formatScore(score);
    fragment.querySelector(".product-price").textContent =
      formatPrice(product.price);

    const imageUrl = safeImageUrl(product.image_url);

    if (imageUrl) {
      image.src = imageUrl;
      image.alt = `Product image for ${title}`;
      imageFrame.classList.add("has-image");

      image.addEventListener("error", () => {
        image.removeAttribute("src");
        imageFrame.classList.remove("has-image");
      });
    }

    fragment.querySelector(".similar-action").addEventListener(
      "click",
      () => {
        setMode("similar", false);
        elements.similarForm.elements.product_id.value =
          product.item_id;
        elements.similarForm.requestSubmit();
      },
    );

    const eventButton = fragment.querySelector(".event-action");

    eventButton.addEventListener("click", async () => {
      eventButton.disabled = true;

      try {
        await recordEvent(
          currentUserId(),
          product.item_id,
          "view",
        );
        showToast(`View recorded for ${product.item_id}.`);
      } catch (error) {
        showToast(error.message, true);
      } finally {
        eventButton.disabled = false;
      }
    });

    elements.productGrid.append(fragment);
  });

  const label = products.length === 1 ? "product" : "products";
  elements.resultCount.textContent = `${products.length} ${label}`;
  setResultsState("results");
}

function sessionItems(value) {
  return String(value || "")
    .split(/[\s,]+/)
    .map((item) => item.trim())
    .filter(Boolean);
}

async function loadRecommendations() {
  const form = elements.recommendationForm;
  const data = new FormData(form);
  const userId = String(data.get("user_id") || "").trim();
  const parameters = new URLSearchParams({
    k: String(data.get("k")),
  });

  sessionItems(data.get("session_item_ids")).forEach((itemId) => {
    parameters.append("session_item_ids", itemId);
  });

  setFormBusy(form, true);
  elements.productGrid.replaceChildren();
  elements.resultCount.textContent = "0 products";
  setResultsState("loading", "Ranking products");

  try {
    const body = await apiRequest(
      `/recommendations/${encodeURIComponent(userId)}?${parameters}`,
    );
    renderProducts(body.recommendations, "recommendations");
  } catch (error) {
    setResultsState("error", error.message);
    showToast(error.message, true);
  } finally {
    setFormBusy(form, false);
  }
}

async function loadSimilarProducts() {
  const form = elements.similarForm;
  const data = new FormData(form);
  const productId = String(data.get("product_id") || "").trim();
  const parameters = new URLSearchParams({
    k: String(data.get("k")),
  });

  setFormBusy(form, true);
  elements.productGrid.replaceChildren();
  elements.resultCount.textContent = "0 products";
  setResultsState("loading", "Retrieving semantic matches");

  try {
    const body = await apiRequest(
      `/similar-products/${encodeURIComponent(productId)}?${parameters}`,
    );
    renderProducts(body.similar_products, "similar");
  } catch (error) {
    setResultsState("error", error.message);
    showToast(error.message, true);
  } finally {
    setFormBusy(form, false);
  }
}

async function submitEvent() {
  const form = elements.eventForm;
  const data = new FormData(form);
  const userId = String(data.get("user_id") || "").trim();
  const itemId = String(data.get("item_id") || "").trim();
  const eventType = String(data.get("event_type") || "");

  setFormBusy(form, true);
  elements.eventReceipt.hidden = true;

  try {
    const event = await recordEvent(userId, itemId, eventType);
    const occurredAt = new Date(event.occurred_at).toLocaleString();

    elements.eventReceipt.textContent =
      `${humanize(event.event_type)} accepted for ` +
      `${event.item_id} at ${occurredAt}.`;
    elements.eventReceipt.hidden = false;
    elements.recommendationForm.elements.user_id.value = userId;
    showToast("Interaction recorded.");
  } catch (error) {
    showToast(error.message, true);
  } finally {
    setFormBusy(form, false);
  }
}

async function loadRuntimeInfo() {
  try {
    const [health, model] = await Promise.all([
      apiRequest("/health"),
      apiRequest("/model-info"),
    ]);

    const backend = health.event_store === "postgresql"
      ? "PostgreSQL"
      : humanize(health.event_store);

    setServiceState("online", `Online · ${backend}`);
    elements.modelVersion.textContent = model.model_version;
    elements.modelVersion.title = model.model_version;
    elements.modelDataset.textContent = model.dataset;
    elements.modelItems.textContent =
      Number(model.semantic_items).toLocaleString();
  } catch (error) {
    setServiceState("offline", "Service unavailable");
    showToast(error.message, true);
  }
}

function bindEvents() {
  elements.modeButtons.forEach((button) => {
    button.addEventListener("click", () => {
      setMode(button.dataset.mode);
    });
  });

  elements.recommendationForm.addEventListener("submit", (event) => {
    event.preventDefault();
    loadRecommendations();
  });

  elements.similarForm.addEventListener("submit", (event) => {
    event.preventDefault();
    loadSimilarProducts();
  });

  elements.eventForm.addEventListener("submit", (event) => {
    event.preventDefault();
    submitEvent();
  });
}

async function bootstrap() {
  bindEvents();
  setMode("recommendations", false);
  await loadRuntimeInfo();
  elements.recommendationForm.requestSubmit();
}

window.addEventListener("DOMContentLoaded", bootstrap);