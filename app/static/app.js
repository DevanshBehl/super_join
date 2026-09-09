// Two small behaviours: upload with progress polling, and expandable rows.
// No framework and no build step, on purpose.

async function uploadDocument(event) {
  event.preventDefault();
  const form = event.target;
  const status = document.getElementById("upload-status");
  const data = new FormData(form);
  if (!data.get("file") || !data.get("file").name) {
    status.textContent = "Choose a PDF first.";
    return;
  }
  status.textContent = "Uploading " + data.get("file").name + " ...";
  const response = await fetch("/api/documents", { method: "POST", body: data });
  if (!response.ok) {
    status.textContent = "Upload failed: " + (await response.text());
    return;
  }
  const body = await response.json();
  status.textContent = "Ingesting " + data.get("file").name + " (doc " + body.doc_id.slice(0, 12) + ") ...";
  pollStatus(body.doc_id, status);
}

async function pollStatus(docId, target) {
  const response = await fetch("/api/documents/" + docId + "/status");
  if (!response.ok) {
    target.textContent = "Status unavailable.";
    return;
  }
  const body = await response.json();
  const stages = (body.stages || []).map((s) => s.stage + " " + s.n_items).join(" | ");
  target.textContent =
    "status: " + body.status +
    (body.progress_stage ? " (running " + body.progress_stage + ")" : "") +
    " | facts " + body.n_verified_facts + " verified of " + body.n_facts +
    (stages ? " | " + stages : "");
  if (body.status === "done" || body.status === "failed") {
    setTimeout(() => window.location.reload(), 1200);
    return;
  }
  setTimeout(() => pollStatus(docId, target), 1500);
}

document.addEventListener("DOMContentLoaded", () => {
  const form = document.getElementById("upload-form");
  if (form) form.addEventListener("submit", uploadDocument);
  document.querySelectorAll("[data-poll-doc]").forEach((el) => {
    pollStatus(el.dataset.pollDoc, el);
  });
});
