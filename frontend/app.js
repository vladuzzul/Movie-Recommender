const movieGrid = document.querySelector("#movieGrid");
const searchInput = document.querySelector("#searchInput");
const clearButton = document.querySelector("#clearButton");
const recommendButton = document.querySelector("#recommendButton");
const ratedList = document.querySelector("#ratedList");
const ratedCount = document.querySelector("#ratedCount");
const recommendations = document.querySelector("#recommendations");
const statusText = document.querySelector("#statusText");

const state = {
  movies: [],
  ratings: new Map(),
  searchTimer: null,
};

const escapeHtml = (value) =>
  String(value ?? "").replace(/[&<>"']/g, (char) => ({
    "&": "&amp;",
    "<": "&lt;",
    ">": "&gt;",
    '"': "&quot;",
    "'": "&#39;",
  })[char]);

async function fetchJson(url, options = {}) {
  const response = await fetch(url, options);
  if (!response.ok) {
    const error = await response.json().catch(() => ({}));
    throw new Error(error.detail || "Request failed");
  }
  return response.json();
}

function movieSubtitle(movie) {
  const parts = [];
  if (movie.year) parts.push(movie.year);
  if (movie.rating) parts.push(`${movie.rating.toFixed(1)} / 5`);
  return parts.join(" · ");
}

function renderMovies() {
  if (!state.movies.length) {
    movieGrid.className = "movie-grid empty-state";
    movieGrid.textContent = "No movies found";
    return;
  }

  movieGrid.className = "movie-grid";
  movieGrid.innerHTML = state.movies.map((movie) => {
    const currentRating = state.ratings.get(movie.title) || 0;
    const stars = [1, 2, 3, 4, 5].map((value) => `
      <button
        class="star-button ${value <= currentRating ? "active" : ""}"
        type="button"
        aria-label="${value} stele pentru ${escapeHtml(movie.title)}"
        data-title="${escapeHtml(movie.title)}"
        data-rating="${value}"
      >★</button>
    `).join("");

    return `
      <article class="movie-card">
        <div class="movie-meta">${escapeHtml(movieSubtitle(movie))}</div>
        <div>
          <h3 class="movie-title">${escapeHtml(movie.title)}</h3>
          <div class="movie-meta">${escapeHtml(movie.genres)}</div>
        </div>
        <div class="stars">${stars}</div>
      </article>
    `;
  }).join("");
}

function renderRated() {
  const items = [...state.ratings.entries()];
  ratedCount.textContent = items.length;
  recommendButton.disabled = items.length === 0;

  if (!items.length) {
    ratedList.className = "rated-list empty-state";
    ratedList.textContent = "No rated movie";
    return;
  }

  ratedList.className = "rated-list";
  ratedList.innerHTML = items.map(([title, rating]) => `
    <div class="rated-item">
      <div>
        <strong>${escapeHtml(title)}</strong>
        <div class="movie-meta">${"★".repeat(rating)}${"☆".repeat(5 - rating)}</div>
      </div>
      <button class="remove-button" type="button" aria-label="Sterge ${escapeHtml(title)}" data-remove="${escapeHtml(title)}">×</button>
    </div>
  `).join("");
}

function renderRecommendations(items) {
  if (!items.length) {
    recommendations.className = "recommendations empty-state";
    recommendations.textContent = "There are no suggestions for the current selection.";
    return;
  }

  recommendations.className = "recommendations";
  recommendations.innerHTML = items.map((item) => `
    <article class="recommendation-card">
      <div class="rec-meta">${escapeHtml([item.year, item.rating ? `${item.rating.toFixed(1)} / 5` : null].filter(Boolean).join(" · "))}</div>
      <h3 class="rec-title">${escapeHtml(item.title)}</h3>
      <div class="rec-meta">${escapeHtml(item.genres)}</div>
      <div class="score-pill">${Math.round(item.score)} matching</div>
      <div class="rec-meta">Based on ${escapeHtml(item.based_on.join(", "))}</div>
    </article>
  `).join("");
}

async function loadMovies(query = "") {
  const params = new URLSearchParams({ limit: "24" });
  if (query.trim()) params.set("q", query.trim());
  state.movies = await fetchJson(`/api/movies?${params.toString()}`);
  renderMovies();
}

movieGrid.addEventListener("click", (event) => {
  const button = event.target.closest("[data-rating]");
  if (!button) return;

  state.ratings.set(button.dataset.title, Number(button.dataset.rating));
  renderMovies();
  renderRated();
});

ratedList.addEventListener("click", (event) => {
  const button = event.target.closest("[data-remove]");
  if (!button) return;

  state.ratings.delete(button.dataset.remove);
  renderMovies();
  renderRated();
});

searchInput.addEventListener("input", () => {
  clearTimeout(state.searchTimer);
  state.searchTimer = setTimeout(() => {
    loadMovies(searchInput.value).catch((error) => {
      movieGrid.className = "movie-grid empty-state error-text";
      movieGrid.textContent = error.message;
    });
  }, 180);
});

clearButton.addEventListener("click", () => {
  searchInput.value = "";
  state.ratings.clear();
  renderRated();
  renderRecommendations([]);
  statusText.textContent = "";
  loadMovies();
});

recommendButton.addEventListener("click", async () => {
  recommendButton.disabled = true;
  statusText.textContent = "Calculating";

  try {
    const payload = {
      ratings: [...state.ratings.entries()].map(([title, rating]) => ({ title, rating })),
    };
    const items = await fetchJson("/api/recommendations", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });
    renderRecommendations(items);
    statusText.textContent = `${items.length} results`;
  } catch (error) {
    recommendations.className = "recommendations empty-state error-text";
    recommendations.textContent = error.message;
    statusText.textContent = "";
  } finally {
    recommendButton.disabled = state.ratings.size === 0;
  }
});

loadMovies().catch((error) => {
  movieGrid.className = "movie-grid empty-state error-text";
  movieGrid.textContent = error.message;
});
