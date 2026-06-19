const API_BASE = "";

let selectedProgramId = null;
let selectedAdminProgramId = null;
let selectedProgram = null;
let adminProgramsCache = [];
let adminStatusFilter = "all";
let currentJobId = null;
let jobTimer = null;

const healthEl = document.getElementById("health");
const userTabEl = document.getElementById("user-tab");
const adminTabEl = document.getElementById("admin-tab");
const userViewEl = document.getElementById("user-view");
const adminViewEl = document.getElementById("admin-view");

const programListEl = document.getElementById("program-list");
const refreshProgramsEl = document.getElementById("refresh-programs");
const detailsEl = document.getElementById("details");
const chatBoxEl = document.getElementById("chat-box");
const chatFormEl = document.getElementById("chat-form");
const chatInputEl = document.getElementById("chat-input");
const qaModelEl = document.getElementById("qa-model");
const suggestionsEl = document.getElementById("suggestions");
const chatModelBadgeEl = document.getElementById("chat-model-badge");

const customUrlEl = document.getElementById("custom-url");
const modelFastEl = document.getElementById("model-fast");
const modelClassificationEl = document.getElementById("model-classification");
const startCrawlEl = document.getElementById("start-crawl");
const checkAllLinksEl = document.getElementById("check-all-links");
const jobStatusEl = document.getElementById("job-status");
const jobMessageEl = document.getElementById("job-message");
const jobLogEl = document.getElementById("job-log");
const discoveredListEl = document.getElementById("discovered-list");

const adminProgramSelectEl = document.getElementById("admin-program-select");
const adminStatusFiltersEl = document.getElementById("admin-status-filters");
const adminReviewStatusEl = document.getElementById("admin-review-status");
const adminJsonEl = document.getElementById("admin-json");
const adminSourceEl = document.getElementById("admin-source");
const adminStatusEl = document.getElementById("admin-status");
const adminLoadEl = document.getElementById("admin-load");
const adminSaveEl = document.getElementById("admin-save");

function escapeHtml(text) {
  return String(text ?? "")
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#39;");
}

function displayValue(value) {
  if (value == null) return "";
  if (typeof value === "string" || typeof value === "number" || typeof value === "boolean") {
    return String(value);
  }
  if (Array.isArray(value)) {
    return value.map(displayValue).filter(Boolean).join(" · ");
  }
  if (typeof value === "object") {
    return Object.values(value).map(displayValue).filter(Boolean).join(" · ");
  }
  return String(value);
}

function setRole(role) {
  const isAdmin = role === "admin";
  userViewEl.classList.toggle("hidden", isAdmin);
  adminViewEl.classList.toggle("hidden", !isAdmin);
  userTabEl.classList.toggle("active", !isAdmin);
  adminTabEl.classList.toggle("active", isAdmin);
}

function statusClass(status) {
  if (status === "approved") return "status-ok";
  if (status === "rejected") return "status-rejected";
  return "status-review";
}

function normalizeStatus(status) {
  return String(status || "needs_review").trim() || "needs_review";
}

function renderAdminStatusFilters(programs) {
  if (!adminStatusFiltersEl) return;

  const counts = {
    all: programs.length,
    needs_review: 0,
    approved: 0,
    rejected: 0,
  };

  for (const program of programs) {
    const status = normalizeStatus(program.status);
    counts[status] = (counts[status] || 0) + 1;
  }

  const filters = [
    ["all", "All"],
    ["needs_review", "needs_review"],
    ["approved", "approved"],
    ["rejected", "rejected"],
  ];

  adminStatusFiltersEl.innerHTML = "";
  for (const [value, label] of filters) {
    const button = document.createElement("button");
    button.type = "button";
    button.dataset.status = value;
    button.className = `status-filter ${adminStatusFilter === value ? "active" : ""}`;
    button.innerHTML = `
      <span>${escapeHtml(label)}</span>
      <strong>${escapeHtml(counts[value] ?? 0)}</strong>
    `;
    button.addEventListener("click", () => {
      adminStatusFilter = value;
      renderAdminStatusFilters(adminProgramsCache);
      renderAdminProgramOptions(adminProgramsCache);
    });
    adminStatusFiltersEl.appendChild(button);
  }
}

function filterAdminPrograms(programs) {
  if (adminStatusFilter === "all") return programs;
  return programs.filter((program) => normalizeStatus(program.status) === adminStatusFilter);
}

function addChatLine(role, text) {
  const line = document.createElement("div");
  line.className = `chat-line ${role === "User" ? "user-line" : "assistant-line"}`;
  line.innerHTML = `<strong>${escapeHtml(role)}:</strong><span>${escapeHtml(text)}</span>`;
  chatBoxEl.appendChild(line);
  chatBoxEl.scrollTop = chatBoxEl.scrollHeight;
  return line;
}

function updateChatLine(line, text) {
  const span = line?.querySelector("span");
  if (!span) return;
  span.textContent = text;
  chatBoxEl.scrollTop = chatBoxEl.scrollHeight;
}

function renderSuggestions(questions) {
  const fallbackQuestions = [
    "Ποιοι είναι δικαιούχοι;",
    "Τι ποσό χρηματοδότησης καλύπτει;",
    "Ποια είναι η προθεσμία;",
    "Ποιες παρεμβάσεις καλύπτονται;",
  ];

  suggestionsEl.innerHTML = "";
  for (const q of questions?.length ? questions : fallbackQuestions) {
    const btn = document.createElement("button");
    btn.type = "button";
    btn.textContent = q;
    btn.addEventListener("click", () => {
      chatInputEl.value = q;
      chatFormEl.requestSubmit();
    });
    suggestionsEl.appendChild(btn);
  }
}

function syncAdminProgramSelect() {
  if (!adminProgramSelectEl) return;
  adminProgramSelectEl.value = selectedAdminProgramId || selectedProgramId || "";
}

function renderAdminProgramOptions(programs) {
  const visiblePrograms = filterAdminPrograms(programs);
  if (selectedAdminProgramId && !visiblePrograms.some((program) => program.id === selectedAdminProgramId)) {
    selectedAdminProgramId = null;
    adminJsonEl.value = "";
    adminSourceEl.textContent = "Επιλέξτε πρόγραμμα.";
    adminStatusEl.textContent = "";
  }

  adminProgramSelectEl.innerHTML = `<option value="">Επιλέξτε πρόγραμμα...</option>`;
  for (const program of visiblePrograms) {
    const status = normalizeStatus(program.status);
    const option = document.createElement("option");
    option.value = program.id;
    option.textContent = `${program.title} (${status})`;
    adminProgramSelectEl.appendChild(option);
  }
  syncAdminProgramSelect();
}

function renderDetails(program) {
  selectedProgram = program;
  const raw = program.raw || {};
  const interventions = Array.isArray(raw.eligible_interventions) ? raw.eligible_interventions : [];
  const contact = Array.isArray(raw.contact_info) ? raw.contact_info : [];

  detailsEl.classList.remove("empty-state");
  detailsEl.innerHTML = `
    <div class="title-row">
      <div>
        <h2>${escapeHtml(program.title)}</h2>
        <p class="muted">${escapeHtml(program.provider)}</p>
      </div>
      <span class="badge ${statusClass(program.status)}">${escapeHtml(program.status)}</span>
    </div>
    <dl class="summary-grid">
      <div><dt>Προθεσμία</dt><dd>${escapeHtml(program.deadline)}</dd></div>
      <div><dt>Σύνδεσμος</dt><dd>${program.link ? `<a href="${escapeHtml(program.link)}" target="_blank" rel="noreferrer">Άνοιγμα πηγής</a>` : "Δεν υπάρχει"}</dd></div>
    </dl>
    <section>
      <h3>Περιγραφή</h3>
      <p>${escapeHtml(program.description)}</p>
    </section>
    <section>
      <h3>Επιλεξιμότητα</h3>
      <p>${escapeHtml(program.eligibility)}</p>
    </section>
    <section>
      <h3>Χρηματοδότηση</h3>
      <p>${escapeHtml(program.funding)}</p>
    </section>
    ${interventions.length ? `
      <section>
        <h3>Παρεμβάσεις</h3>
        <ul class="bullet-list">${interventions.map((item) => `<li>${escapeHtml(item)}</li>`).join("")}</ul>
      </section>
    ` : ""}
    ${contact.length ? `
      <section>
        <h3>Επαφή</h3>
        <ul class="bullet-list">${contact.map((item) => `<li>${escapeHtml(displayValue(item))}</li>`).join("")}</ul>
      </section>
    ` : ""}
  `;
}

async function loadProgramDetails(programId) {
  const res = await fetch(`${API_BASE}/programs/${programId}`);
  if (!res.ok) {
    throw new Error("Failed to load program details");
  }
  return res.json();
}

async function loadHealth() {
  try {
    const res = await fetch(`${API_BASE}/health`);
    const data = await res.json();
    healthEl.textContent = `${data.program_count} δημόσια · ${data.candidate_count} candidates`;
  } catch {
    healthEl.textContent = "API offline";
  }
}

async function loadPrograms(selectFirst = false) {
  const [publicRes, adminRes] = await Promise.all([
    fetch(`${API_BASE}/programs`),
    fetch(`${API_BASE}/admin/programs`),
  ]);
  if (!publicRes.ok || !adminRes.ok) {
    throw new Error("Failed to load programs");
  }
  const programs = await publicRes.json();
  adminProgramsCache = await adminRes.json();

  programListEl.innerHTML = "";
  if (!programs.length) {
    programListEl.innerHTML = `<li class="muted">Δεν υπάρχουν approved προγράμματα ακόμη.</li>`;
  }
  for (const p of programs) {
    const li = document.createElement("li");
    const btn = document.createElement("button");
    btn.type = "button";
    btn.dataset.programId = p.id;
    btn.innerHTML = `
      <strong>${escapeHtml(p.title)}</strong>
      <span>${escapeHtml(p.provider)}</span>
      <small>${escapeHtml(p.deadline)}</small>
      <em class="${statusClass(p.status)}">${escapeHtml(p.status)}</em>
    `;

    btn.addEventListener("click", async () => {
      for (const b of programListEl.querySelectorAll("button")) {
        b.classList.remove("active");
      }
      btn.classList.add("active");

      selectedProgramId = p.id;
      selectedAdminProgramId = p.id;
      syncAdminProgramSelect();
      const details = await loadProgramDetails(p.id);
      renderDetails(details);
      chatBoxEl.innerHTML = "";
      addChatLine("Assistant", `Έχει επιλεγεί: ${details.title}`);
      adminStatusEl.textContent = "";
      adminSourceEl.textContent = "Επιλέχθηκε πρόγραμμα. Πατήστε Φόρτωση.";
    });

    li.appendChild(btn);
    programListEl.appendChild(li);
  }
  renderAdminStatusFilters(adminProgramsCache);
  renderAdminProgramOptions(adminProgramsCache);

  const currentBtn = selectedProgramId
    ? programListEl.querySelector(`button[data-program-id="${selectedProgramId}"]`)
    : null;
  const firstBtn = programListEl.querySelector("button");
  if (currentBtn) {
    currentBtn.click();
  } else if (selectFirst && firstBtn) {
    firstBtn.click();
  }
  await loadHealth();
}

async function pollJob() {
  if (!currentJobId) return;
  const res = await fetch(`${API_BASE}/admin/jobs/${currentJobId}`);
  if (!res.ok) return;
  const job = await res.json();

  jobStatusEl.textContent = job.status;
  jobStatusEl.className = `badge ${job.status === "failed" ? "status-review" : "status-ok"}`;
  jobMessageEl.textContent = job.message;
  jobLogEl.textContent = job.logs.length ? job.logs.join("\n") : "Δεν υπάρχουν μηνύματα ακόμη.";
  jobLogEl.classList.toggle("empty-log", !job.logs.length);
  discoveredListEl.innerHTML = "";

  for (const item of job.discovered || []) {
    const row = document.createElement("a");
    row.href = item.url;
    row.target = "_blank";
    row.rel = "noreferrer";
    row.className = "discovered-item";
    row.innerHTML = `
      <strong>${escapeHtml(item.title || item.url)}</strong>
      <span>${escapeHtml(item.reason || "")} · score ${escapeHtml(item.score ?? "")}</span>
    `;
    discoveredListEl.appendChild(row);
  }

  if (["completed", "failed"].includes(job.status)) {
    clearInterval(jobTimer);
    jobTimer = null;
    await loadPrograms(false);
  }
}

async function loadAdminRaw() {
  if (!selectedAdminProgramId) {
    adminStatusEl.textContent = "Επιλέξτε πρόγραμμα πρώτα.";
    return;
  }

  const res = await fetch(`${API_BASE}/admin/programs/${selectedAdminProgramId}/raw`);
  if (!res.ok) {
    adminStatusEl.textContent = "Αποτυχία φόρτωσης JSON.";
    return;
  }

  const data = await res.json();
  adminSourceEl.textContent = data.source_file;
  adminReviewStatusEl.value = data.data.review_status || "needs_review";
  adminJsonEl.value = JSON.stringify(data.data, null, 2);
  adminStatusEl.textContent = "Φορτώθηκε.";
}

chatFormEl.addEventListener("submit", async (event) => {
  event.preventDefault();
  if (!selectedProgramId) {
    addChatLine("Assistant", "Επιλέξτε πρώτα πρόγραμμα.");
    return;
  }

  const message = chatInputEl.value.trim();
  if (!message) return;

  addChatLine("User", message);
  chatInputEl.value = "";
  const waitingLine = addChatLine("Assistant", "Στέλνω την ερώτηση στο μοντέλο...");
  const submitButton = chatFormEl.querySelector("button[type='submit']");
  if (submitButton) submitButton.disabled = true;
  const slowNoticeTimer = setTimeout(() => {
    updateChatLine(
      waitingLine,
      "Περιμένω απάντηση από το μοντέλο. Αν αργεί, πιθανόν είναι απασχολημένο με άλλο crawl ή προηγούμενη ερώτηση."
    );
    chatModelBadgeEl.textContent = "Waiting for model";
  }, 8000);

  try {
    const res = await fetch(`${API_BASE}/chat/programs/${selectedProgramId}`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        message,
        qa_model: qaModelEl.value.trim() || null,
      }),
    });

    if (!res.ok) {
      updateChatLine(waitingLine, "Δεν μπόρεσα να πάρω απάντηση.");
      return;
    }

    const data = await res.json();
    chatModelBadgeEl.textContent = data.used_llm ? data.model : "LLM unavailable";
    updateChatLine(waitingLine, data.reply || "Δεν υπάρχει απάντηση.");
    renderSuggestions(data.suggested_questions);
  } catch (error) {
    updateChatLine(waitingLine, `Δεν μπόρεσα να πάρω απάντηση: ${error.message}`);
  } finally {
    clearTimeout(slowNoticeTimer);
    if (submitButton) submitButton.disabled = false;
  }
});

async function startAdminCrawl(includeKnownLinks) {
  const customUrl = customUrlEl.value.trim();
  const body = {
    urls: customUrl ? [customUrl] : [],
    include_known_links: includeKnownLinks,
    model_fast: modelFastEl.value.trim() || null,
    model_classification: modelClassificationEl.value.trim() || null,
  };

  const res = await fetch(`${API_BASE}/admin/crawl`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });

  if (!res.ok) {
    jobMessageEl.textContent = "Αποτυχία εκκίνησης crawl.";
    return;
  }

  const data = await res.json();
  currentJobId = data.job_id;
  jobStatusEl.textContent = data.status;
  jobMessageEl.textContent = data.message;
  jobLogEl.textContent = "Αναμονή για μηνύματα εκτέλεσης...";
  jobLogEl.classList.add("empty-log");
  discoveredListEl.innerHTML = "";
  if (jobTimer) clearInterval(jobTimer);
  jobTimer = setInterval(pollJob, 1500);
  pollJob();
}

startCrawlEl.addEventListener("click", () => startAdminCrawl(false));
checkAllLinksEl.addEventListener("click", () => startAdminCrawl(true));

adminProgramSelectEl.addEventListener("change", async () => {
  selectedAdminProgramId = adminProgramSelectEl.value || null;
  adminJsonEl.value = "";
  if (!selectedAdminProgramId) {
    adminSourceEl.textContent = "Επιλέξτε πρόγραμμα.";
    adminStatusEl.textContent = "";
    return;
  }

  adminSourceEl.textContent = "Επιλέχθηκε πρόγραμμα.";
  adminStatusEl.textContent = "";
  await loadAdminRaw();
});

adminLoadEl.addEventListener("click", loadAdminRaw);

adminSaveEl.addEventListener("click", async () => {
  if (!selectedAdminProgramId) {
    adminStatusEl.textContent = "Επιλέξτε πρόγραμμα πρώτα.";
    return;
  }

  let parsed;
  try {
    parsed = JSON.parse(adminJsonEl.value);
  } catch {
    adminStatusEl.textContent = "Μη έγκυρο JSON.";
    return;
  }
  parsed.review_status = adminReviewStatusEl.value || "needs_review";

  const res = await fetch(`${API_BASE}/admin/programs/${selectedAdminProgramId}/raw`, {
    method: "PUT",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ data: parsed }),
  });

  if (!res.ok) {
    adminStatusEl.textContent = "Αποτυχία αποθήκευσης.";
    return;
  }

  adminStatusEl.textContent = "Αποθηκεύτηκε.";
  await loadPrograms(false);
});

userTabEl.addEventListener("click", () => setRole("user"));
adminTabEl.addEventListener("click", () => setRole("admin"));
refreshProgramsEl.addEventListener("click", () => loadPrograms(false));

renderSuggestions();
loadHealth();
loadPrograms(true).catch((error) => {
  detailsEl.innerHTML = `<p class="empty-state">Αποτυχία φόρτωσης: ${escapeHtml(error.message)}</p>`;
});
