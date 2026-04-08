const API_BASE = "";

let selectedProgramId = null;
let selectedProgramTitle = "";

const programListEl = document.getElementById("program-list");
const detailsEl = document.getElementById("details");
const chatBoxEl = document.getElementById("chat-box");
const chatFormEl = document.getElementById("chat-form");
const chatInputEl = document.getElementById("chat-input");
const suggestionsEl = document.getElementById("suggestions");

const toggleAdminEl = document.getElementById("toggle-admin");
const adminPanelEl = document.getElementById("admin-panel");
const adminJsonEl = document.getElementById("admin-json");
const adminSourceEl = document.getElementById("admin-source");
const adminStatusEl = document.getElementById("admin-status");
const adminLoadEl = document.getElementById("admin-load");
const adminSaveEl = document.getElementById("admin-save");

function escapeHtml(text) {
  return String(text)
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#39;");
}

function addChatLine(role, text) {
  const line = document.createElement("div");
  line.className = "chat-line";
  line.innerHTML = `<strong>${escapeHtml(role)}:</strong> ${escapeHtml(text)}`;
  chatBoxEl.appendChild(line);
  chatBoxEl.scrollTop = chatBoxEl.scrollHeight;
}

function renderSuggestions() {
  const questions = [
    "Am I eligible?",
    "What documents do I need?",
    "When is the deadline?",
  ];

  suggestionsEl.innerHTML = "";
  for (const q of questions) {
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

function renderDetails(program) {
  detailsEl.innerHTML = `
    <p><strong>Title:</strong> ${escapeHtml(program.title || "")}</p>
    <p><strong>Provider:</strong> ${escapeHtml(program.provider || "")}</p>
    <p><strong>Description:</strong> ${escapeHtml(program.description || "")}</p>
    <p><strong>Eligibility:</strong> ${escapeHtml(program.eligibility || "")}</p>
    <p><strong>Funding:</strong> ${escapeHtml(program.funding || "")}</p>
    <p><strong>Deadline:</strong> ${escapeHtml(program.deadline || "")}</p>
    <p><strong>Link:</strong> ${program.link ? `<a href="${escapeHtml(program.link)}" target="_blank" rel="noreferrer">${escapeHtml(program.link)}</a>` : ""}</p>
  `;
}

async function loadProgramDetails(programId) {
  const res = await fetch(`${API_BASE}/programs/${programId}`);
  if (!res.ok) {
    throw new Error("Failed to load program details");
  }
  return res.json();
}

async function loadPrograms() {
  const res = await fetch(`${API_BASE}/programs`);
  if (!res.ok) {
    throw new Error("Failed to load programs");
  }
  const programs = await res.json();

  programListEl.innerHTML = "";
  for (const p of programs) {
    const li = document.createElement("li");
    const btn = document.createElement("button");
    btn.type = "button";
    btn.textContent = p.title;
    btn.dataset.programId = p.id;

    btn.addEventListener("click", async () => {
      for (const b of programListEl.querySelectorAll("button")) {
        b.classList.remove("active");
      }
      btn.classList.add("active");

      selectedProgramId = p.id;
      selectedProgramTitle = p.title;
      const details = await loadProgramDetails(p.id);
      renderDetails(details);
      chatBoxEl.innerHTML = "";
      addChatLine("System", `Selected: ${selectedProgramTitle}`);
      adminStatusEl.textContent = "";
    });

    li.appendChild(btn);
    programListEl.appendChild(li);
  }

  const firstBtn = programListEl.querySelector("button");
  if (firstBtn) {
    firstBtn.click();
  }
}

chatFormEl.addEventListener("submit", async (event) => {
  event.preventDefault();
  if (!selectedProgramId) {
    addChatLine("System", "Please select a program first.");
    return;
  }

  const message = chatInputEl.value.trim();
  if (!message) {
    return;
  }

  addChatLine("User", message);
  chatInputEl.value = "";

  const res = await fetch(`${API_BASE}/chat/programs/${selectedProgramId}`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ message }),
  });

  if (!res.ok) {
    addChatLine("Assistant", "Error while getting chat response.");
    return;
  }

  const data = await res.json();
  addChatLine("Assistant", data.reply || "No response");
});

toggleAdminEl.addEventListener("click", () => {
  adminPanelEl.classList.toggle("hidden");
});

adminLoadEl.addEventListener("click", async () => {
  if (!selectedProgramId) {
    adminStatusEl.textContent = "Select a program first.";
    return;
  }

  const res = await fetch(`${API_BASE}/admin/programs/${selectedProgramId}/raw`);
  if (!res.ok) {
    adminStatusEl.textContent = "Failed to load JSON.";
    return;
  }

  const data = await res.json();
  adminSourceEl.textContent = `Source: ${data.source_file}`;
  adminJsonEl.value = JSON.stringify(data.data, null, 2);
  adminStatusEl.textContent = "Loaded.";
});

adminSaveEl.addEventListener("click", async () => {
  if (!selectedProgramId) {
    adminStatusEl.textContent = "Select a program first.";
    return;
  }

  let parsed;
  try {
    parsed = JSON.parse(adminJsonEl.value);
  } catch {
    adminStatusEl.textContent = "Invalid JSON.";
    return;
  }

  const res = await fetch(`${API_BASE}/admin/programs/${selectedProgramId}/raw`, {
    method: "PUT",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ data: parsed }),
  });

  if (!res.ok) {
    adminStatusEl.textContent = "Save failed.";
    return;
  }

  adminStatusEl.textContent = "Saved successfully.";
  await loadPrograms();
});

renderSuggestions();
loadPrograms().catch((error) => {
  detailsEl.innerHTML = `<p>Failed to load data: ${escapeHtml(error.message)}</p>`;
});
