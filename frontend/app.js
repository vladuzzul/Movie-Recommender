const STORAGE_KEY = "cinemaQuestRatings";

const page = document.body.dataset.page;

const escapeHtml = (value) =>
  String(value ?? "").replace(/[&<>"']/g, (char) => ({
    "&": "&amp;",
    "<": "&lt;",
    ">": "&gt;",
    '"': "&quot;",
    "'": "&#39;",
  })[char]);

const getRatings = () => {
  try {
    const saved = JSON.parse(localStorage.getItem(STORAGE_KEY) || "[]");
    return new Map(saved.map((item) => [item.title, item.rating]));
  } catch {
    return new Map();
  }
};

const saveRatings = (ratings) => {
  const payload = [...ratings.entries()].map(([title, rating]) => ({ title, rating }));
  localStorage.setItem(STORAGE_KEY, JSON.stringify(payload));
};

const clearRatings = () => {
  localStorage.removeItem(STORAGE_KEY);
};

const fetchJson = async (url, options = {}) => {
  const response = await fetch(url, options);
  if (!response.ok) {
    const error = await response.json().catch(() => ({}));
    throw new Error(error.detail || "Request failed");
  }
  return response.json();
};

const movieSubtitle = (movie) => {
  const parts = [];
  if (movie.year) parts.push(movie.year);
  if (movie.rating) parts.push(`${movie.rating.toFixed(1)} / 5`);
  return parts.join(" - ");
};

const posterClass = (title) => `poster alt-${String(title).length % 4}`;

const posterInitial = (title) => {
  const firstWord = String(title).trim().split(/\s+/)[0] || "?";
  return firstWord.slice(0, 2).toUpperCase();
};

const starText = (rating) => `${"★".repeat(rating)}${"-".repeat(5 - rating)}`;

function renderRatedList(container, ratings, mode = "full") {
  const items = [...ratings.entries()];

  if (!items.length) {
    container.className = "rated-list empty-state";
    container.textContent = "No movies rated yet.";
    return;
  }

  container.className = "rated-list";
  container.innerHTML = items.map(([title, rating]) => `
    <div class="rated-item">
      <div>
        <strong>${escapeHtml(title)}</strong>
        <div class="movie-meta">${escapeHtml(starText(rating))}</div>
      </div>
      ${mode === "editable" ? `<button class="remove-button" type="button" aria-label="Remove ${escapeHtml(title)}" data-remove="${escapeHtml(title)}">x</button>` : ""}
    </div>
  `).join("");
}

function initHome() {
  const count = document.querySelector("#homeRatingCount");
  if (!count) return;

  const ratings = getRatings();
  const totalPower = [...ratings.values()].reduce((sum, rating) => sum + rating, 0);
  count.textContent = ratings.size
    ? `${ratings.size} ratings - ${totalPower} star power`
    : "0 ratings loaded";
}

function initRatePage() {
  const movieGrid = document.querySelector("#movieGrid");
  const searchInput = document.querySelector("#searchInput");
  const clearButton = document.querySelector("#clearButton");
  const ratedList = document.querySelector("#ratedList");
  const ratedCount = document.querySelector("#ratedCount");
  const starPower = document.querySelector("#starPower");
  const recommendButton = document.querySelector("#recommendButton");

  const state = {
    movies: [],
    ratings: getRatings(),
    searchTimer: null,
  };

  const updateHud = () => {
    const totalPower = [...state.ratings.values()].reduce((sum, rating) => sum + rating, 0);
    ratedCount.textContent = state.ratings.size;
    starPower.textContent = totalPower;
    recommendButton.classList.toggle("disabled", state.ratings.size === 0);
    recommendButton.setAttribute("aria-disabled", state.ratings.size === 0 ? "true" : "false");
    renderRatedList(ratedList, state.ratings, "editable");
  };

  const renderMovies = () => {
    if (!state.movies.length) {
      movieGrid.className = "movie-grid empty-state";
      movieGrid.textContent = "No movies found.";
      return;
    }

    movieGrid.className = "movie-grid";
    movieGrid.innerHTML = state.movies.map((movie) => {
      const currentRating = state.ratings.get(movie.title) || 0;
      const stars = [1, 2, 3, 4, 5].map((value) => `
        <button
          class="star-button ${value <= currentRating ? "active" : ""}"
          type="button"
          aria-label="${value} stars for ${escapeHtml(movie.title)}"
          data-title="${escapeHtml(movie.title)}"
          data-rating="${value}"
        >★</button>
      `).join("");

      return `
        <article class="movie-card">
          <div>
            <div class="movie-meta">${escapeHtml(movieSubtitle(movie))}</div>
            <h3 class="movie-title">${escapeHtml(movie.title)}</h3>
            <div class="movie-meta">${escapeHtml(movie.genres)}</div>
          </div>
          <div class="stars">${stars}</div>
        </article>
      `;
    }).join("");
  };

  const loadMovies = async (query = "") => {
    const params = new URLSearchParams({ limit: "30" });
    if (query.trim()) params.set("q", query.trim());
    state.movies = await fetchJson(`/api/movies?${params.toString()}`);
    renderMovies();
  };

  movieGrid.addEventListener("click", (event) => {
    const button = event.target.closest("[data-rating]");
    if (!button) return;

    state.ratings.set(button.dataset.title, Number(button.dataset.rating));
    saveRatings(state.ratings);
    renderMovies();
    updateHud();
  });

  ratedList.addEventListener("click", (event) => {
    const button = event.target.closest("[data-remove]");
    if (!button) return;

    state.ratings.delete(button.dataset.remove);
    saveRatings(state.ratings);
    renderMovies();
    updateHud();
  });

  clearButton.addEventListener("click", () => {
    state.ratings.clear();
    clearRatings();
    searchInput.value = "";
    updateHud();
    loadMovies();
  });

  recommendButton.addEventListener("click", (event) => {
    if (state.ratings.size === 0) {
      event.preventDefault();
    }
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

  updateHud();
  loadMovies().catch((error) => {
    movieGrid.className = "movie-grid empty-state error-text";
    movieGrid.textContent = error.message;
  });
}

function initSuggestionsPage() {
  const ratedList = document.querySelector("#ratedList");
  const recommendations = document.querySelector("#recommendations");
  const statusText = document.querySelector("#statusText");
  const intro = document.querySelector("#resultsIntro");
  const clearButton = document.querySelector("#clearButton");
  const ratings = getRatings();

  renderRatedList(ratedList, ratings);

  clearButton.addEventListener("click", () => {
    clearRatings();
    window.location.href = "/rate";
  });

  if (!ratings.size) {
    statusText.textContent = "No ratings";
    intro.textContent = "Rate a few movies first to unlock this screen.";
    recommendations.className = "reward-grid empty-state";
    recommendations.innerHTML = `No reward chest yet. <a class="ghost-link" href="/rate">Rate movies</a>`;
    return;
  }

  const payload = {
    ratings: [...ratings.entries()].map(([title, rating]) => ({ title, rating })),
  };

  statusText.textContent = "Calculating";
  intro.textContent = `Using ${ratings.size} rated movies to unlock your recommendations.`;

  fetchJson("/api/recommendations", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  })
    .then((items) => {
      statusText.textContent = `${items.length} unlocked`;
      recommendations.className = "reward-grid";
      recommendations.innerHTML = items.map((item, index) => `
        <article class="recommendation-card">
          <div class="${posterClass(item.title)}">${escapeHtml(index + 1)}</div>
          <div class="rec-meta">${escapeHtml([item.year, item.rating ? `${item.rating.toFixed(1)} / 5` : null].filter(Boolean).join(" - "))}</div>
          <h3 class="rec-title">${escapeHtml(item.title)}</h3>
          <div class="rec-meta">${escapeHtml(item.genres)}</div>
          <div class="score-pill">${Math.round(item.score)} match points</div>
          <div class="rec-meta">Unlocked by ${escapeHtml(item.based_on.join(", "))}</div>
        </article>
      `).join("");
    })
    .catch((error) => {
      statusText.textContent = "Error";
      recommendations.className = "reward-grid empty-state error-text";
      recommendations.textContent = error.message;
    });
}

if (page === "home") {
  initHome();
}

if (page === "rate") {
  initRatePage();
}

if (page === "suggestions") {
  initSuggestionsPage();
}
