const $ = (sel, root = document) => root.querySelector(sel);
const $$ = (sel, root = document) => Array.from(root.querySelectorAll(sel));

const statusEl = $("#status");
const viewEl = $("#view");

const LS_BASE_URL = "xiaozhi.console.baseUrl";

function ensureToastHost() {
  let host = document.getElementById("toastHost");
  if (host) return host;
  host = document.createElement("div");
  host.id = "toastHost";
  host.className = "toastHost";
  document.body.appendChild(host);
  return host;
}

function showToast({ kind = "info", title, body } = {}) {
  const host = ensureToastHost();
  const el = document.createElement("div");
  el.className = `toast toast--${kind}`;

  const titleEl = document.createElement("div");
  titleEl.className = "toast__title";
  titleEl.textContent = title || (kind === "bad" ? "Error" : kind === "ok" ? "OK" : "Info");

  const bodyEl = document.createElement("div");
  bodyEl.className = "toast__body mono";
  bodyEl.textContent = body || "";

  const closeEl = document.createElement("button");
  closeEl.className = "toast__close";
  closeEl.textContent = "×";
  closeEl.title = "Dismiss";
  closeEl.addEventListener("click", () => el.remove());

  el.appendChild(closeEl);
  el.appendChild(titleEl);
  if (body) el.appendChild(bodyEl);
  el.addEventListener("click", () => el.remove());

  host.appendChild(el);
  setTimeout(() => el.remove(), 6500);
}

function setStatus(text, kind = "info") {
  statusEl.textContent = text;
  statusEl.className = `footer__right pill ${kind === "ok" ? "pill--ok" : kind === "bad" ? "pill--bad" : ""}`.trim();
}

function nowMs() {
  return Date.now();
}

function fmtMs(ms) {
  if (!ms) return "";
  try {
    return new Date(Number(ms)).toLocaleString();
  } catch {
    return String(ms);
  }
}

function todayKey() {
  const d = new Date();
  const y = String(d.getFullYear());
  const m = String(d.getMonth() + 1).padStart(2, "0");
  const day = String(d.getDate()).padStart(2, "0");
  return `${y}-${m}-${day}`;
}

function parseJson(text, fallback = {}) {
  const s = String(text ?? "").trim();
  if (!s) return fallback;
  const obj = JSON.parse(s);
  if (obj && typeof obj === "object") return obj;
  return fallback;
}

function computePlannedAtMs(instanceKey, plannedTime) {
  const key = String(instanceKey || "").trim();
  const t = String(plannedTime || "").trim();
  if (!t) return null;
  const m = t.match(/^(\d{1,2}):(\d{2})$/);
  if (!m) return null;
  let hh = Number(m[1]);
  const mm = Number(m[2]);
  if (Number.isNaN(hh) || Number.isNaN(mm) || mm < 0 || mm > 59) return null;

  let base = new Date();
  if (/^\d{4}-\d{2}-\d{2}$/.test(key)) {
    const [Y, M, D] = key.split("-").map((x) => Number(x));
    base = new Date(Y, M - 1, D, 0, 0, 0, 0);
  }

  // "24:00" => next day 00:00
  if (hh === 24 && mm === 0) {
    return new Date(base.getTime() + 24 * 60 * 60 * 1000).getTime();
  }

  if (hh < 0 || hh > 23) return null;
  return new Date(base.getFullYear(), base.getMonth(), base.getDate(), hh, mm, 0, 0).getTime();
}

function escapeHtml(s) {
  return String(s ?? "")
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;");
}

function renderPolicySummary(taskType, policy) {
  const p = policy && typeof policy === "object" ? policy : {};
  const tt = String(taskType || "").trim();

  const pill = (label, value, kind = "info") => {
    const cls = kind === "ok" ? "pill pill--ok" : kind === "bad" ? "pill pill--bad" : "pill";
    return `<span class="${cls}"><span class="mono">${escapeHtml(label)}</span>: <span class="mono">${escapeHtml(value)}</span></span>`;
  };

  if (tt === "wake_up") {
    const deviceId = String(p.device_id || "").trim();
    const cooldown = p.cooldown_sec ?? "";
    const offline = p.offline_retry_sec ?? "";
    const windowMinutes = p.window_minutes ?? "";
    const maxAttempts = p.max_attempts ?? "";
    const nudgeEnabled = p.nudge_enabled ?? "";

    const bits = [];
    bits.push(pill("device_id", deviceId || "(missing)", deviceId ? "ok" : "bad"));
    if (cooldown !== "") bits.push(pill("cooldown_sec", String(cooldown), "ok"));
    if (offline !== "") bits.push(pill("offline_retry_sec", String(offline), "ok"));
    if (windowMinutes !== "") bits.push(pill("window_minutes", String(windowMinutes), "ok"));
    if (maxAttempts !== "") bits.push(pill("max_attempts", String(maxAttempts), "ok"));
    if (nudgeEnabled !== "") bits.push(pill("nudge_enabled", String(!!nudgeEnabled), nudgeEnabled ? "ok" : "info"));

    const known = new Set([
      "device_id",
      "cooldown_sec",
      "offline_retry_sec",
      "window_minutes",
      "max_attempts",
      "nudge_enabled",
    ]);
    const extraKeys = Object.keys(p).filter((k) => !known.has(k));
    if (extraKeys.length) bits.push(pill("extra", `${extraKeys.length} keys`, "info"));

    return `<div class="pillRow">${bits.join(" ")}</div>`;
  }

  const entries = Object.entries(p).slice(0, 6);
  if (!entries.length) return `<span class="pill pill--bad"><span class="mono">policy</span>: <span class="mono">(empty)</span></span>`;
  const bits = entries.map(([k, v]) => pill(k, typeof v === "string" ? v : JSON.stringify(v), "info"));
  const extra = Object.keys(p).length - entries.length;
  if (extra > 0) bits.push(pill("extra", `${extra} keys`, "info"));
  return `<div class="pillRow">${bits.join(" ")}</div>`;
}

function buildUrl(baseUrl, path, query) {
  const base = String(baseUrl || "").replace(/\/+$/, "");
  const p = String(path || "");
  const u = new URL(base + (p.startsWith("/") ? p : `/${p}`));
  for (const [k, v] of Object.entries(query || {})) {
    if (v === undefined || v === null || String(v).trim() === "") continue;
    u.searchParams.set(k, String(v));
  }
  return u.toString();
}

async function apiFetch(state, path, { method = "GET", json, query } = {}) {
  const url = buildUrl(state.baseUrl, path, query);
  const headers = { Accept: "application/json" };
  let body = undefined;
  if (json !== undefined) {
    headers["Content-Type"] = "application/json";
    body = JSON.stringify(json);
  }
  const resp = await fetch(url, { method, headers, body });
  const text = await resp.text();
  let data = null;
  try {
    data = text ? JSON.parse(text) : null;
  } catch {
    data = { raw: text };
  }
  if (!resp.ok) {
    const msg = (data && data.message) || `${resp.status} ${resp.statusText}`;
    const err = new Error(msg);
    err.status = resp.status;
    err.data = data;
    throw err;
  }
  return data;
}

function renderTaskEngine(state) {
  const html = `
    <div class="panel">
      <h2 class="panel__title">TASK ENGINE</h2>
      <div class="pillRow" style="margin-top:6px;">
        <span class="pill"><span class="mono">module</span>: <span class="mono">task_engine</span></span>
        <span class="pill"><span class="mono">ui</span>: <span class="mono">http-only</span></span>
      </div>
      <div class="pill" style="margin-top:10px;">
        This module talks to backend APIs only. It stays decoupled so we can add other modules (global settings, etc.) later.
      </div>
    </div>

    <div class="panel">
      <h2 class="panel__title">TASKS</h2>
      <div class="row">
        <label class="field">
          <span class="field__label">Account ID (optional)</span>
          <input class="input" id="filterAccountId" placeholder="e.g. u123" spellcheck="false" />
        </label>
        <button class="btn btn--primary" id="btnLoadTasks">Load Tasks</button>
        <span class="pill" id="tasksMeta">tasks: -</span>
      </div>
      <div style="margin-top:12px;overflow:auto;">
        <table class="table" id="tasksTable">
          <thead>
            <tr>
              <th>account_id</th>
              <th>task_type</th>
              <th>enabled</th>
              <th>updated_at</th>
              <th>policy</th>
              <th>policy_summary</th>
              <th>actions</th>
            </tr>
          </thead>
          <tbody></tbody>
        </table>
      </div>
      <h2 class="panel__title">UPSERT TASK</h2>
      <div class="row">
        <label class="field">
          <span class="field__label">account_id</span>
          <input class="input" id="taskAccountId" spellcheck="false" />
        </label>
        <label class="field">
          <span class="field__label">task_type</span>
          <input class="input" id="taskType" placeholder="wake_up" spellcheck="false" />
        </label>
        <label class="field field--inline" style="min-width:auto;">
          <input type="checkbox" id="taskEnabled" checked />
          <span class="field__label">enabled</span>
        </label>
        <button class="btn btn--primary" id="btnUpsertTask">Upsert</button>
      </div>
      <div style="margin-top:10px;">
        <label class="field" style="min-width:100%;">
          <span class="field__label">policy (JSON)</span>
          <textarea class="textarea mono" id="taskPolicy" spellcheck="false">{}</textarea>
        </label>
      </div>
      <div class="pill" style="margin-top:10px;">
        wake_up requires <span class="mono">policy.device_id</span>
      </div>
    </div>

    <div class="panel">
      <div class="panel__titleRow">
        <h2 class="panel__title">INSTANCES</h2>
        <button class="iconBtn" id="btnInstancesHelp" title="Help">!</button>
      </div>
      <div class="row">
        <label class="field">
          <span class="field__label">account_id</span>
          <input class="input" id="instAccountId" placeholder="u123" spellcheck="false" />
        </label>
        <label class="field">
          <span class="field__label">task_type</span>
          <input class="input" id="instTaskType" placeholder="wake_up" spellcheck="false" />
        </label>
        <button class="btn btn--primary" id="btnLoadInstances">Load Instances</button>
        <span class="pill" id="instancesMeta">instances: -</span>
      </div>

      <div style="margin-top:12px;overflow:auto;">
        <table class="table" id="instancesTable">
          <thead>
            <tr>
              <th>instance_key</th>
              <th>status</th>
              <th>planned</th>
              <th>window</th>
              <th>next_action</th>
              <th>run/max</th>
              <th>attempts</th>
              <th>actions</th>
            </tr>
          </thead>
          <tbody></tbody>
        </table>
      </div>

      <div style="margin-top:14px;border-top:1px solid rgba(255,184,0,0.18);padding-top:14px;">
        <h3 class="panel__title" style="font-size:14px;margin:0 0 12px 0;">KICKOFF (also create/schedule instance)</h3>
        <div class="row">
          <label class="field">
            <span class="field__label">account_id</span>
            <input class="input" id="kickAccountId" placeholder="u123" spellcheck="false" />
          </label>
          <label class="field">
            <span class="field__label">task_type</span>
            <input class="input" id="kickTaskType" placeholder="wake_up" spellcheck="false" />
          </label>
          <label class="field">
            <span class="field__label">instance_key</span>
            <input class="input" id="kickInstanceKey" placeholder="${todayKey()}" spellcheck="false" />
          </label>
          <label class="field">
            <span class="field__label">planned_time (HH:MM)</span>
            <input class="input" id="kickPlannedTime" placeholder="09:40" spellcheck="false" />
          </label>
          <label class="field">
            <span class="field__label">window_minutes</span>
            <input class="input" id="kickWindowMinutes" placeholder="30" spellcheck="false" />
          </label>
          <label class="field">
            <span class="field__label">max_runs</span>
            <input class="input" id="kickMaxRuns" placeholder="(optional)" spellcheck="false" />
          </label>
          <button class="btn btn--primary" id="btnKickoff">Kickoff</button>
        </div>
        <div class="pill" style="margin-top:10px;">
          If you set planned_time, this kickoff will schedule next_action_at_ms to that planned time (no immediate run). Use Run(now) for debugging.
        </div>
      </div>
    </div>

    <div class="panel">
      <h2 class="panel__title">ATTEMPTS</h2>
      <div class="row">
        <label class="field">
          <span class="field__label">account_id</span>
          <input class="input" id="attAccountId" placeholder="u123" spellcheck="false" />
        </label>
        <label class="field">
          <span class="field__label">task_type</span>
          <input class="input" id="attTaskType" placeholder="wake_up" spellcheck="false" />
        </label>
        <label class="field">
          <span class="field__label">instance_key (optional)</span>
          <input class="input" id="attInstanceKey" placeholder="${todayKey()}" spellcheck="false" />
        </label>
        <button class="btn btn--primary" id="btnLoadAttempts">Load Attempts</button>
      </div>
      <div style="margin-top:12px;" id="attemptsBox"></div>
    </div>

    <div class="panel">
      <h2 class="panel__title">CURL QUICKREF</h2>
      <div class="mono" style="font-size:12px;line-height:1.5;">
curl -X POST ${escapeHtml(state.baseUrl)}/tasks \\\n  -H 'Content-Type: application/json' \\\n  -d '{\"account_id\":\"u123\",\"task_type\":\"wake_up\",\"enabled\":true,\"policy\":{\"device_id\":\"5a:84:...\"}}'\n\n\
curl -X POST ${escapeHtml(state.baseUrl)}/tasks/u123/wake_up/kickoff \\\n  -H 'Content-Type: application/json' \\\n  -d '{}'\n\n\
curl -X POST ${escapeHtml(state.baseUrl)}/tasks/u123/wake_up/kickoff \\\n  -H 'Content-Type: application/json' \\\n  -d '{\"instance_key\":\"2026-02-10\",\"planned_time\":\"09:40\",\"next_action_at_ms\":1770716400000}'\n\n\
curl -X POST ${escapeHtml(state.baseUrl)}/tasks/u123/wake_up/run \\\n  -H 'Content-Type: application/json' \\\n  -d '{\"instance_key\":\"2026-02-10\"}'\n\n\
curl ${escapeHtml(state.baseUrl)}/tasks/u123/instances?task_type=wake_up\n\
      </div>
    </div>

    <div class="modal modal--hidden" id="instancesHelpModal" role="dialog" aria-modal="true" aria-hidden="true">
      <div class="modal__backdrop" data-act="close"></div>
      <div class="modal__card">
        <div class="modal__header">
          <div class="modal__title">Instance actions (what each button does)</div>
          <button class="iconBtn" data-act="close" title="Close">×</button>
        </div>
        <div class="modal__body mono">
          <div class="helpGrid">
            <div class="helpGrid__row">
              <div class="helpGrid__key">Kickoff(now)</div>
              <div class="helpGrid__val">POST /kickoff. Creates or refreshes the instance (may fill planned/window if needed) and sets next_action_at_ms = now (or payload value).</div>
            </div>
            <div class="helpGrid__row">
              <div class="helpGrid__key">Run(now)</div>
              <div class="helpGrid__val">POST /run. Only sets next_action_at_ms = now for an existing instance. Does not change planned/window. 404 if instance does not exist.</div>
            </div>
            <div class="helpGrid__row">
              <div class="helpGrid__key">Set PAUSED</div>
              <div class="helpGrid__val">POST /pause. Sets status = PAUSED. Scheduler will skip it even if it is due.</div>
            </div>
            <div class="helpGrid__row">
              <div class="helpGrid__key">Set CANCELED</div>
              <div class="helpGrid__val">POST /cancel. Sets status = CANCELED (terminal). Later kickoff for the same instance_key will return 409 instance_terminal.</div>
            </div>
            <div class="helpGrid__row">
              <div class="helpGrid__key">Get Attempts</div>
              <div class="helpGrid__val">GET /attempts?instance_key=... Read-only. Shows attempt records for that instance.</div>
            </div>
            <div class="helpGrid__row">
              <div class="helpGrid__key">Delete</div>
              <div class="helpGrid__val">DELETE /instances/{instance_key}. Removes the instance and its attempts (cascade).</div>
            </div>
            <div class="helpGrid__row">
              <div class="helpGrid__key">Delete task</div>
              <div class="helpGrid__val">DELETE /tasks/{account_id}/{task_type}. Removes task + all instances/attempts (cascade).</div>
            </div>
          </div>
        </div>
        <div class="modal__footer">
          <button class="btn btn--primary" data-act="close">OK</button>
        </div>
      </div>
    </div>
  `;

  viewEl.innerHTML = html;

  const filterAccountIdEl = $("#filterAccountId");
  const tasksMetaEl = $("#tasksMeta");
  const tasksTableBody = $("#tasksTable tbody");

  const taskAccountIdEl = $("#taskAccountId");
  const taskTypeEl = $("#taskType");
  const taskEnabledEl = $("#taskEnabled");
  const taskPolicyEl = $("#taskPolicy");

  const instAccountIdEl = $("#instAccountId");
  const instTaskTypeEl = $("#instTaskType");
  const instancesMetaEl = $("#instancesMeta");
  const instancesTableBody = $("#instancesTable tbody");

  const kickAccountIdEl = $("#kickAccountId");
  const kickTaskTypeEl = $("#kickTaskType");
  const kickInstanceKeyEl = $("#kickInstanceKey");
  const kickPlannedTimeEl = $("#kickPlannedTime");
  const kickWindowMinutesEl = $("#kickWindowMinutes");
  const kickMaxRunsEl = $("#kickMaxRuns");

  const attAccountIdEl = $("#attAccountId");
  const attTaskTypeEl = $("#attTaskType");
  const attInstanceKeyEl = $("#attInstanceKey");
  const attemptsBoxEl = $("#attemptsBox");
  const instancesHelpModalEl = $("#instancesHelpModal");

  const tasksCache = new Map();
  let lastInstanceRows = [];

  function setSelectedTask(accountId, taskType) {
    if (accountId) taskAccountIdEl.value = accountId;
    if (taskType) taskTypeEl.value = taskType;

    if (accountId) instAccountIdEl.value = accountId;
    if (taskType) instTaskTypeEl.value = taskType;

    if (accountId) kickAccountIdEl.value = accountId;
    if (taskType) kickTaskTypeEl.value = taskType;

    if (accountId) attAccountIdEl.value = accountId;
    if (taskType) attTaskTypeEl.value = taskType;
  }

  async function loadTasks() {
    setStatus("loading tasks...");
    tasksTableBody.innerHTML = "";
    const accountId = String(filterAccountIdEl.value || "").trim();
    try {
      const data = accountId
        ? await apiFetch(state, `/tasks/${encodeURIComponent(accountId)}/all`)
        : await apiFetch(state, "/tasks");
      const tasks = (data && data.data) || [];
      tasksMetaEl.textContent = `tasks: ${tasks.length}`;
      tasksCache.clear();
      for (const t of tasks) {
        tasksCache.set(`${t.account_id}::${t.task_type}`, t);
      }

      tasksTableBody.innerHTML = tasks
        .map((t) => {
          const policyPreview = JSON.stringify(t.policy || {}, null, 0);
          const policySummary = renderPolicySummary(t.task_type, t.policy);
          return `
            <tr data-account="${escapeHtml(t.account_id)}" data-type="${escapeHtml(t.task_type)}">
              <td class="mono">${escapeHtml(t.account_id)}</td>
              <td class="mono">${escapeHtml(t.task_type)}</td>
              <td>${t.enabled ? `<span class="pill pill--ok">on</span>` : `<span class="pill pill--bad">off</span>`}</td>
              <td class="mono">${escapeHtml(fmtMs(t.updated_at_ms))}</td>
              <td class="mono">${escapeHtml(policyPreview)}</td>
              <td>${policySummary}</td>
              <td>
                <button class="btn btn--ghost" data-act="select">Select</button>
                <button class="btn btn--danger" data-act="deleteTask">Delete</button>
              </td>
            </tr>
          `;
        })
        .join("");

      setStatus("tasks loaded", "ok");
    } catch (e) {
      console.error(e);
      setStatus(`load tasks failed: ${e.message}`, "bad");
      showToast({ kind: "bad", title: "Load tasks failed", body: e?.data ? JSON.stringify(e.data) : e.message });
      tasksMetaEl.textContent = "tasks: error";
    }
  }

  async function upsertTask() {
    const account_id = String(taskAccountIdEl.value || "").trim();
    const task_type = String(taskTypeEl.value || "").trim();
    if (!account_id || !task_type) {
      setStatus("account_id/task_type required", "bad");
      showToast({ kind: "bad", title: "Missing fields", body: "account_id and task_type are required." });
      return;
    }
    let policy = {};
    try {
      policy = parseJson(taskPolicyEl.value, {});
      taskPolicyEl.style.borderColor = "";
    } catch (e) {
      taskPolicyEl.style.borderColor = "rgba(255,77,77,0.9)";
      setStatus(`policy json invalid: ${e.message}`, "bad");
      showToast({ kind: "bad", title: "Invalid policy JSON", body: e.message });
      return;
    }

    setStatus("upserting task...");
    try {
      await apiFetch(state, "/tasks", {
        method: "POST",
        json: { account_id, task_type, enabled: !!taskEnabledEl.checked, policy },
      });
      setStatus("task upserted", "ok");
      showToast({ kind: "ok", title: "Task upserted", body: `${account_id}/${task_type}` });
      setSelectedTask(account_id, task_type);
      await loadTasks();
    } catch (e) {
      console.error(e);
      setStatus(`upsert task failed: ${e.message}`, "bad");
      showToast({ kind: "bad", title: "Upsert task failed", body: e?.data ? JSON.stringify(e.data) : e.message });
    }
  }

  async function deleteTask(accountId, taskType) {
    if (!confirm(`Delete task ${accountId}/${taskType}? (instances/attempts will be removed)`)) return;
    setStatus("deleting task...");
    try {
      await apiFetch(state, `/tasks/${encodeURIComponent(accountId)}/${encodeURIComponent(taskType)}`, {
        method: "DELETE",
      });
      setStatus("task deleted", "ok");
      showToast({ kind: "ok", title: "Task deleted", body: `${accountId}/${taskType}` });
      await loadTasks();
    } catch (e) {
      console.error(e);
      setStatus(`delete task failed: ${e.message}`, "bad");
      showToast({ kind: "bad", title: "Delete task failed", body: e?.data ? JSON.stringify(e.data) : e.message });
    }
  }

  async function loadInstances() {
    const accountId = String(instAccountIdEl.value || "").trim();
    const taskType = String(instTaskTypeEl.value || "").trim();
    if (!accountId) {
      setStatus("instances: account_id required", "bad");
      showToast({ kind: "bad", title: "Missing fields", body: "account_id is required to list instances." });
      return;
    }
    setStatus("loading instances...");
    instancesTableBody.innerHTML = "";
    try {
      const data = await apiFetch(state, `/tasks/${encodeURIComponent(accountId)}/instances`, {
        query: { task_type: taskType || undefined },
      });
      const rows = (data && data.data) || [];
      lastInstanceRows = Array.isArray(rows) ? rows : [];
      instancesMetaEl.textContent = `instances: ${rows.length}`;
      instancesTableBody.innerHTML = rows
        .map((r) => {
          const planned = fmtMs(r.planned_at_ms);
          const window = `${fmtMs(r.window_start_at_ms)} → ${fmtMs(r.window_end_at_ms)}`;
          const nextAction = fmtMs(r.next_action_at_ms);
          return `
            <tr data-account="${escapeHtml(r.account_id)}" data-type="${escapeHtml(r.task_type)}" data-key="${escapeHtml(r.instance_key)}">
              <td class="mono">${escapeHtml(r.instance_key)}</td>
              <td class="mono">${escapeHtml(r.status)}</td>
              <td class="mono">${escapeHtml(planned)}</td>
              <td class="mono">${escapeHtml(window)}</td>
              <td class="mono">${escapeHtml(nextAction)}</td>
              <td class="mono">${escapeHtml(String(r.run_count ?? ""))} / ${escapeHtml(String(r.max_runs ?? ""))}</td>
              <td class="mono">${escapeHtml(String(r.attempt_count ?? ""))}</td>
              <td>
                <button class="btn btn--ghost" data-act="kickNow">Kickoff(now)</button>
                <button class="btn btn--ghost" data-act="runNow">Run(now)</button>
                <button class="btn btn--ghost" data-act="pause">Set PAUSED</button>
                <button class="btn btn--ghost" data-act="cancel">Set CANCELED</button>
                <button class="btn btn--ghost" data-act="attempts">Get Attempts</button>
                <button class="btn btn--danger" data-act="deleteInst">Delete</button>
              </td>
            </tr>
          `;
        })
        .join("");
      setStatus("instances loaded", "ok");
    } catch (e) {
      console.error(e);
      setStatus(`load instances failed: ${e.message}`, "bad");
      showToast({ kind: "bad", title: "Load instances failed", body: e?.data ? JSON.stringify(e.data) : e.message });
      instancesMetaEl.textContent = "instances: error";
      lastInstanceRows = [];
    }
  }

  async function kickoff() {
    const accountId = String(kickAccountIdEl.value || "").trim();
    const taskType = String(kickTaskTypeEl.value || "").trim();
    const instanceKey = String(kickInstanceKeyEl.value || "").trim() || todayKey();
    const plannedTime = String(kickPlannedTimeEl.value || "").trim();
    const windowMinutes = String(kickWindowMinutesEl.value || "").trim();
    const maxRuns = String(kickMaxRunsEl.value || "").trim();

    if (!accountId || !taskType) {
      setStatus("kickoff: account_id/task_type required", "bad");
      showToast({ kind: "bad", title: "Missing fields", body: "account_id and task_type are required for kickoff." });
      return;
    }

    const payload = { instance_key: instanceKey };
    if (plannedTime) payload.planned_time = plannedTime;
    if (windowMinutes) payload.window_minutes = Number(windowMinutes);
    if (maxRuns) payload.max_runs = Number(maxRuns);

    if (plannedTime) {
      const plannedAt = computePlannedAtMs(instanceKey, plannedTime);
      if (!plannedAt) {
        setStatus("planned_time must be HH:MM (e.g. 09:40)", "bad");
        showToast({
          kind: "bad",
          title: "Invalid planned_time",
          body: "planned_time must be a valid HH:MM time (e.g. 09:40).",
        });
        return;
      }
      // Align next_action to planned time whenever planned_time is provided.
      payload.next_action_at_ms = plannedAt;
    }

    setStatus("kickoff...");
    try {
      await apiFetch(state, `/tasks/${encodeURIComponent(accountId)}/${encodeURIComponent(taskType)}/kickoff`, {
        method: "POST",
        json: payload,
      });
      setStatus("kickoff ok", "ok");
      showToast({ kind: "ok", title: "Kickoff OK", body: `${accountId}/${taskType} ${instanceKey}` });
      setSelectedTask(accountId, taskType);
      await loadInstances();
      await loadTasks();
    } catch (e) {
      console.error(e);
      setStatus(`kickoff failed: ${e.message}`, "bad");
      const detail = e?.data ? JSON.stringify(e.data) : e.message;
      showToast({ kind: "bad", title: "Kickoff failed", body: detail });
    }
  }

  async function kickNow(accountId, taskType, instanceKey) {
    setStatus("kickoff(now)...");
    try {
      await apiFetch(state, `/tasks/${encodeURIComponent(accountId)}/${encodeURIComponent(taskType)}/kickoff`, {
        method: "POST",
        json: { instance_key: instanceKey, next_action_at_ms: nowMs() },
      });
      setStatus("kickoff(now) ok", "ok");
      showToast({ kind: "ok", title: "Kickoff(now) OK", body: `${accountId}/${taskType} ${instanceKey}` });
      await loadInstances();
    } catch (e) {
      console.error(e);
      setStatus(`kickoff(now) failed: ${e.message}`, "bad");
      showToast({ kind: "bad", title: "Kickoff(now) failed", body: e?.data ? JSON.stringify(e.data) : e.message });
    }
  }

  async function runNow(accountId, taskType, instanceKey) {
    setStatus("run(now)...");
    try {
      await apiFetch(state, `/tasks/${encodeURIComponent(accountId)}/${encodeURIComponent(taskType)}/run`, {
        method: "POST",
        json: { instance_key: instanceKey },
      });
      setStatus("run(now) ok", "ok");
      showToast({ kind: "ok", title: "Run(now) OK", body: `${accountId}/${taskType} ${instanceKey}` });
      await loadInstances();
    } catch (e) {
      console.error(e);
      setStatus(`run(now) failed: ${e.message}`, "bad");
      showToast({ kind: "bad", title: "Run(now) failed", body: e?.data ? JSON.stringify(e.data) : e.message });
    }
  }

  async function pause(accountId, taskType, instanceKey) {
    setStatus("pause...");
    try {
      await apiFetch(state, `/tasks/${encodeURIComponent(accountId)}/${encodeURIComponent(taskType)}/pause`, {
        method: "POST",
        json: { instance_key: instanceKey },
      });
      setStatus("paused", "ok");
      showToast({ kind: "ok", title: "Set PAUSED", body: `${accountId}/${taskType} ${instanceKey}` });
      await loadInstances();
    } catch (e) {
      console.error(e);
      setStatus(`pause failed: ${e.message}`, "bad");
      showToast({ kind: "bad", title: "Set PAUSED failed", body: e?.data ? JSON.stringify(e.data) : e.message });
    }
  }

  async function cancel(accountId, taskType, instanceKey) {
    setStatus("cancel...");
    try {
      await apiFetch(state, `/tasks/${encodeURIComponent(accountId)}/${encodeURIComponent(taskType)}/cancel`, {
        method: "POST",
        json: { instance_key: instanceKey },
      });
      setStatus("canceled", "ok");
      showToast({ kind: "ok", title: "Set CANCELED", body: `${accountId}/${taskType} ${instanceKey}` });
      await loadInstances();
    } catch (e) {
      console.error(e);
      setStatus(`cancel failed: ${e.message}`, "bad");
      showToast({ kind: "bad", title: "Set CANCELED failed", body: e?.data ? JSON.stringify(e.data) : e.message });
    }
  }

  async function deleteInstance(accountId, taskType, instanceKey) {
    if (!confirm(`Delete instance ${accountId}/${taskType}/${instanceKey}? (attempts will be removed)`)) return;
    setStatus("deleting instance...");
    try {
      await apiFetch(
        state,
        `/tasks/${encodeURIComponent(accountId)}/${encodeURIComponent(taskType)}/instances/${encodeURIComponent(instanceKey)}`,
        { method: "DELETE" },
      );
      setStatus("instance deleted", "ok");
      showToast({ kind: "ok", title: "Instance deleted", body: `${accountId}/${taskType} ${instanceKey}` });
      await loadInstances();
    } catch (e) {
      console.error(e);
      setStatus(`delete instance failed: ${e.message}`, "bad");
      showToast({ kind: "bad", title: "Delete instance failed", body: e?.data ? JSON.stringify(e.data) : e.message });
    }
  }

  async function loadAttempts() {
    const accountId = String(attAccountIdEl.value || "").trim();
    const taskType = String(attTaskTypeEl.value || "").trim();
    let instanceKey = String(attInstanceKeyEl.value || "").trim();
    if (!accountId || !taskType) {
      setStatus("attempts: account_id/task_type required", "bad");
      showToast({ kind: "bad", title: "Missing fields", body: "account_id and task_type are required to load attempts." });
      return;
    }

    setStatus("loading attempts...");
    attemptsBoxEl.innerHTML = "";
    try {
      const data = await apiFetch(state, `/tasks/${encodeURIComponent(accountId)}/${encodeURIComponent(taskType)}/attempts`, {
        query: instanceKey ? { instance_key: instanceKey } : {},
      });
      const scope = data?.data?.scope || "instance";
      const attempts = data?.data?.attempts || [];
      const inst = data?.data?.instance;
      const isAll = scope === "all" && !inst;

      const head = isAll
        ? `<div class="pill">scope: <span class="mono">all instances</span> • account: <span class="mono">${escapeHtml(accountId)}</span> • task: <span class="mono">${escapeHtml(taskType)}</span></div>`
        : `<div class="pill">instance: <span class="mono">${escapeHtml(inst?.instance_key || "")}</span> • status: <span class="mono">${escapeHtml(inst?.status || "")}</span></div>`;

      const cols = isAll
        ? `
            <tr>
              <th>at</th>
              <th>instance_key</th>
              <th>instance_status</th>
              <th>result_code</th>
              <th>decision_code</th>
              <th>result_json</th>
              <th>decision_json</th>
            </tr>
          `
        : `
            <tr>
              <th>at</th>
              <th>result_code</th>
              <th>decision_code</th>
              <th>result_json</th>
              <th>decision_json</th>
            </tr>
          `;

      const rows = attempts
        .map((a) => {
          if (isAll) {
            return `
              <tr>
                <td class="mono">${escapeHtml(fmtMs(a.at_ms))}</td>
                <td class="mono">${escapeHtml(a.instance_key || "")}</td>
                <td class="mono">${escapeHtml(a.instance_status || "")}</td>
                <td class="mono">${escapeHtml(a.result_code)}</td>
                <td class="mono">${escapeHtml(a.decision_code)}</td>
                <td class="mono">${escapeHtml(a.result_json)}</td>
                <td class="mono">${escapeHtml(a.decision_json)}</td>
              </tr>
            `;
          }
          return `
            <tr>
              <td class="mono">${escapeHtml(fmtMs(a.at_ms))}</td>
              <td class="mono">${escapeHtml(a.result_code)}</td>
              <td class="mono">${escapeHtml(a.decision_code)}</td>
              <td class="mono">${escapeHtml(a.result_json)}</td>
              <td class="mono">${escapeHtml(a.decision_json)}</td>
            </tr>
          `;
        })
        .join("");

      attemptsBoxEl.innerHTML = `
        ${head}
        <div style="margin-top:12px;overflow:auto;">
          <table class="table">
            <thead>${cols}</thead>
            <tbody>${rows}</tbody>
          </table>
        </div>
      `;
      setStatus("attempts loaded", "ok");
    } catch (e) {
      console.error(e);
      setStatus(`load attempts failed: ${e.message}`, "bad");
      showToast({ kind: "bad", title: "Load attempts failed", body: e?.data ? JSON.stringify(e.data) : e.message });
    }
  }

  $("#btnLoadTasks").addEventListener("click", loadTasks);
  $("#btnUpsertTask").addEventListener("click", upsertTask);
  $("#btnLoadInstances").addEventListener("click", loadInstances);
  $("#btnKickoff").addEventListener("click", kickoff);
  $("#btnLoadAttempts").addEventListener("click", loadAttempts);

  function openModal(el) {
    el.classList.remove("modal--hidden");
    el.setAttribute("aria-hidden", "false");
  }

  function closeModal(el) {
    el.classList.add("modal--hidden");
    el.setAttribute("aria-hidden", "true");
  }

  $("#btnInstancesHelp").addEventListener("click", () => openModal(instancesHelpModalEl));
  instancesHelpModalEl.addEventListener("click", (ev) => {
    const btn = ev.target.closest("[data-act]");
    if (!btn) return;
    const act = btn.getAttribute("data-act");
    if (act === "close") closeModal(instancesHelpModalEl);
  });
  document.addEventListener("keydown", (ev) => {
    if (ev.key === "Escape" && !instancesHelpModalEl.classList.contains("modal--hidden")) {
      closeModal(instancesHelpModalEl);
    }
  });

  tasksTableBody.addEventListener("click", (ev) => {
    const btn = ev.target.closest("button");
    const tr = ev.target.closest("tr");
    if (!btn || !tr) return;
    const accountId = tr.getAttribute("data-account") || "";
    const taskType = tr.getAttribute("data-type") || "";
    const act = btn.getAttribute("data-act");
    if (act === "select") {
      (async () => {
        setSelectedTask(accountId, taskType);
        const t = tasksCache.get(`${accountId}::${taskType}`);
        if (t) {
          taskEnabledEl.checked = !!t.enabled;
          taskPolicyEl.value = JSON.stringify(t.policy || {}, null, 2);
        }
        attInstanceKeyEl.value = "";
        setStatus(`selected ${accountId}/${taskType}`, "ok");
        await loadInstances();
        await loadAttempts();
      })();
      return;
    }
    if (act === "deleteTask") {
      deleteTask(accountId, taskType);
      return;
    }
  });

  instancesTableBody.addEventListener("click", (ev) => {
    const btn = ev.target.closest("button");
    const tr = ev.target.closest("tr");
    if (!btn || !tr) return;
    const accountId = tr.getAttribute("data-account") || "";
    const taskType = tr.getAttribute("data-type") || "";
    const instanceKey = tr.getAttribute("data-key") || "";
    const act = btn.getAttribute("data-act");
    if (act === "kickNow") kickNow(accountId, taskType, instanceKey);
    if (act === "runNow") runNow(accountId, taskType, instanceKey);
    if (act === "pause") pause(accountId, taskType, instanceKey);
    if (act === "cancel") cancel(accountId, taskType, instanceKey);
    if (act === "deleteInst") deleteInstance(accountId, taskType, instanceKey);
    if (act === "attempts") {
      setSelectedTask(accountId, taskType);
      attInstanceKeyEl.value = instanceKey;
      loadAttempts();
    }
  });

  // Seed defaults
  instAccountIdEl.value = String(filterAccountIdEl.value || "").trim();
  kickInstanceKeyEl.value = todayKey();
}

function init() {
  const state = {
    baseUrl: localStorage.getItem(LS_BASE_URL) || window.location.origin,
    view: "taskEngine",
  };

  const baseUrlEl = $("#baseUrl");
  baseUrlEl.value = state.baseUrl;

  $("#saveBaseUrl").addEventListener("click", () => {
    const v = String(baseUrlEl.value || "").trim();
    state.baseUrl = v || window.location.origin;
    localStorage.setItem(LS_BASE_URL, state.baseUrl);
    setStatus("base url saved", "ok");
    if (state.view === "taskEngine") renderTaskEngine(state);
  });

  $$(".nav__item").forEach((btn) => {
    btn.addEventListener("click", () => {
      const view = btn.getAttribute("data-view");
      if (!view) return;
      state.view = view;
      $$(".nav__item").forEach((b) => b.classList.toggle("nav__item--active", b === btn));
      if (view === "taskEngine") renderTaskEngine(state);
      else viewEl.innerHTML = `<div class="panel"><h2 class="panel__title">COMING SOON</h2></div>`;
    });
  });

  renderTaskEngine(state);
  setStatus("ready", "ok");
}

init();

window.addEventListener("unhandledrejection", (ev) => {
  const reason = ev?.reason;
  const msg = reason?.message || String(reason || "unknown_error");
  showToast({ kind: "bad", title: "Unhandled error", body: msg });
});
