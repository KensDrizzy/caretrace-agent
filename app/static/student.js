const AUTH_KEY = "caretrace.auth";
const LAST_SESSION_KEY = "caretrace.student.lastSession";

const state = {
  sessionId: null,
  sending: false,
  profile: null,
  modelName: "mock"
};

const els = {
  serviceState: document.querySelector("#serviceState"),
  modelState: document.querySelector("#modelState"),
  activeAccount: document.querySelector("#activeAccount"),
  switchAccount: document.querySelector("#switchAccount"),
  messages: document.querySelector("#messages"),
  chatForm: document.querySelector("#chatForm"),
  messageInput: document.querySelector("#messageInput"),
  sendButton: document.querySelector("#sendButton"),
  newSession: document.querySelector("#newSession"),
  sessionBadge: document.querySelector("#sessionBadge"),
  sessionDot: document.querySelector(".session-dot"),
  studentGreeting: document.querySelector("#studentGreeting"),
  charCount: document.querySelector("#charCount"),
  recentSessions: document.querySelector("#recentSessions")
};

function readLastSession() {
  try {
    return sessionStorage.getItem(LAST_SESSION_KEY) || null;
  } catch {
    return null;
  }
}

function saveLastSession(sessionId) {
  try {
    if (sessionId) {
      sessionStorage.setItem(LAST_SESSION_KEY, sessionId);
    } else {
      sessionStorage.removeItem(LAST_SESSION_KEY);
    }
  } catch {
    // ignore
  }
}

function clearLastSession() {
  try {
    sessionStorage.removeItem(LAST_SESSION_KEY);
  } catch {
    // ignore
  }
}

function readAuth() {
  try {
    return JSON.parse(sessionStorage.getItem(AUTH_KEY) || "null");
  } catch {
    return null;
  }
}

function clearAuth() {
  sessionStorage.removeItem(AUTH_KEY);
}

function authHeader() {
  const auth = readAuth();
  if (!auth?.token) {
    window.location.replace("/");
    return "";
  }
  return `Basic ${auth.token}`;
}

async function api(path, options = {}) {
  const headers = { ...(options.headers || {}), Authorization: authHeader() };
  const response = await fetch(path, { ...options, headers });
  if (!response.ok) {
    const text = await response.text();
    throw new Error(text || `${response.status} ${response.statusText}`);
  }
  return response;
}

function setPill(el, text, tone = "ok") {
  el.textContent = text;
  el.className = `pill ${tone}`;
}

function isAdmin(profile) {
  return profile.roles?.some((role) => role.authority === "ROLE_ADMIN");
}

function displayModel(model) {
  return (model || "").includes("mindbridge-qwen2.5-7b-ft") ? "微调 Qwen2.5-7B" : model;
}

function studentDisplayName(profile) {
  return profile.displayName === "Demo Student" ? "Student" : (profile.displayName || profile.username || "Student");
}

function greetingForNow() {
  const hour = new Date().getHours();
  if (hour < 6) return "夜深了";
  if (hour < 11) return "早上好";
  if (hour < 14) return "中午好";
  if (hour < 18) return "下午好";
  return "晚上好";
}

function updateCharCount() {
  els.charCount.textContent = `${els.messageInput.value.length} / 1000`;
}

function escapeHtml(text) {
  return (text || "").replace(/[&<>"']/g, (ch) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", "\"": "&quot;", "'": "&#39;" }[ch]));
}

function formatSessionTime(value) {
  const date = value ? new Date(value) : null;
  return date && !isNaN(date) ? date.toLocaleString("zh-CN", { month: "numeric", day: "numeric", hour: "2-digit", minute: "2-digit" }) : "";
}

async function loadRecentSessions() {
  try {
    const response = await api("/api/chat/sessions");
    const sessions = await response.json();
    renderRecentSessions((sessions || []).slice(0, 10));
  } catch (error) {
    console.warn("加载最近对话失败", error);
  }
}

function groupSessionsByDay(sessions) {
  const groups = {};
  const keys = [];
  const now = new Date();
  const today = new Date(now.getFullYear(), now.getMonth(), now.getDate()).getTime();
  const yesterday = today - 24 * 60 * 60 * 1000;

  for (const s of sessions) {
    const d = s.updatedAt ? new Date(s.updatedAt) : null;
    if (!d || isNaN(d)) continue;
    const day = new Date(d.getFullYear(), d.getMonth(), d.getDate()).getTime();
    let key = "older";
    if (day === today) key = "today";
    else if (day === yesterday) key = "yesterday";
    (groups[key] ||= []).push(s);
  }

  const labels = { today: "今天", yesterday: "昨天", older: "更早" };
  const order = ["today", "yesterday", "older"];
  return order.filter((key) => groups[key]).map((key) => ({ key, label: labels[key], sessions: groups[key] }));
}

function renderSessionItem(s, groupKey, isHidden = false) {
  return `
    <li class="${isHidden ? 'hidden recent-collapsible' : ''}" data-group="${escapeHtml(groupKey)}">
      <div class="recent-session-row">
        <button class="recent-session-item" type="button" data-session-id="${escapeHtml(s.sessionId)}">
          <span class="recent-session-title">${escapeHtml(s.title)}</span>
          <span class="recent-session-time">${formatSessionTime(s.updatedAt)}</span>
        </button>
        <button class="recent-session-rename" type="button" data-rename-session-id="${escapeHtml(s.sessionId)}" aria-label="重命名对话" title="重命名对话">
          <svg viewBox="0 0 24 24" aria-hidden="true"><path d="M4 20h4l10.5-10.5a2.8 2.8 0 0 0-4-4L4 16v4Zm10.8-12.8 2 2"/></svg>
        </button>
      </div>
    </li>
  `;
}

function closeSessionRename(row) {
  row.querySelector(".recent-session-rename-form")?.remove();
  row.classList.remove("renaming");
}

function beginSessionRename(button) {
  if (state.sending) return;
  const row = button.closest(".recent-session-row");
  if (!row || row.classList.contains("renaming")) return;
  const sessionButton = row.querySelector(".recent-session-item");
  const titleEl = row.querySelector(".recent-session-title");
  const sessionId = button.dataset.renameSessionId;
  const form = document.createElement("form");
  form.className = "recent-session-rename-form";
  form.innerHTML = `
    <input class="recent-session-rename-input" type="text" maxlength="160" aria-label="新的对话名称">
    <button class="recent-session-rename-save" type="submit" aria-label="保存名称" title="保存">✓</button>
    <button class="recent-session-rename-cancel" type="button" aria-label="取消重命名" title="取消">×</button>
    <span class="recent-session-rename-error" role="alert"></span>
  `;
  const input = form.querySelector("input");
  const saveButton = form.querySelector(".recent-session-rename-save");
  const cancelButton = form.querySelector(".recent-session-rename-cancel");
  const errorEl = form.querySelector(".recent-session-rename-error");
  input.value = titleEl.textContent.trim();
  row.classList.add("renaming");
  row.append(form);
  input.focus();
  input.select();

  cancelButton.addEventListener("click", () => closeSessionRename(row));
  input.addEventListener("keydown", (event) => {
    if (event.key === "Escape") {
      event.preventDefault();
      closeSessionRename(row);
      sessionButton.focus();
    }
  });
  form.addEventListener("submit", async (event) => {
    event.preventDefault();
    const title = input.value.trim().replace(/\s+/g, " ");
    if (!title) {
      errorEl.textContent = "名称不能为空";
      input.focus();
      return;
    }
    input.disabled = true;
    saveButton.disabled = true;
    cancelButton.disabled = true;
    errorEl.textContent = "保存中…";
    try {
      const response = await api(`/api/chat/sessions/${encodeURIComponent(sessionId)}`, {
        method: "PATCH",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ title })
      });
      const renamed = await response.json();
      titleEl.textContent = renamed.title;
      closeSessionRename(row);
      sessionButton.focus();
    } catch (error) {
      input.disabled = false;
      saveButton.disabled = false;
      cancelButton.disabled = false;
      errorEl.textContent = "保存失败，请重试";
      input.focus();
      console.warn("重命名对话失败", error);
    }
  });
}

function renderRecentSessions(sessions) {
  if (!sessions?.length) {
    els.recentSessions.innerHTML = `<li class="recent-empty">暂无历史对话</li>`;
    return;
  }
  const groups = groupSessionsByDay(sessions);
  const MAX_PER_GROUP = 3;
  els.recentSessions.innerHTML = groups.map((group) => {
    const visible = group.sessions.slice(0, MAX_PER_GROUP);
    const hidden = group.sessions.slice(MAX_PER_GROUP);
    const hasMore = hidden.length > 0;
    return `
      <li class="recent-group-label">${escapeHtml(group.label)}</li>
      ${visible.map((s) => renderSessionItem(s, group.key)).join("")}
      ${hidden.map((s) => renderSessionItem(s, group.key, true)).join("")}
      ${hasMore ? `<li class="recent-show-more-wrap" data-show-more="${escapeHtml(group.key)}"><button class="recent-show-more" type="button" data-expanded="false" data-count="${hidden.length}">展开 ${hidden.length} 条</button></li>` : ""}
    `;
  }).join("");
  highlightRecentSession(state.sessionId);
  els.recentSessions.querySelectorAll("[data-session-id]").forEach((btn) => {
    btn.addEventListener("click", () => loadSessionMessages(btn.dataset.sessionId));
  });
  els.recentSessions.querySelectorAll("[data-rename-session-id]").forEach((btn) => {
    btn.addEventListener("click", () => beginSessionRename(btn));
  });
  els.recentSessions.querySelectorAll(".recent-show-more").forEach((btn) => {
    btn.addEventListener("click", () => {
      const key = btn.closest("[data-show-more]").dataset.showMore;
      const expanded = btn.dataset.expanded === "true";
      const items = els.recentSessions.querySelectorAll(`li[data-group="${key}"].recent-collapsible`);
      if (expanded) {
        items.forEach((li) => li.classList.add("hidden"));
        btn.dataset.expanded = "false";
        btn.textContent = `展开 ${btn.dataset.count} 条`;
      } else {
        items.forEach((li) => li.classList.remove("hidden"));
        btn.dataset.expanded = "true";
        btn.textContent = "收起";
      }
    });
  });
}

function highlightRecentSession(sessionId) {
  els.recentSessions.querySelectorAll(".recent-session-item").forEach((btn) => {
    btn.classList.toggle("active", btn.dataset.sessionId === sessionId);
  });
}

async function checkHealth() {
  try {
    const response = await fetch("/actuator/health");
    const body = await response.json();
    setPill(els.serviceState, body.status === "UP" ? "服务正常" : `服务 ${body.status}`, body.status === "UP" ? "ok" : "danger");
  } catch {
    setPill(els.serviceState, "服务 DOWN", "danger");
  }
}

async function loadProfile() {
  try {
    const response = await api("/api/profile");
    const profile = await response.json();
    if (isAdmin(profile)) {
      window.location.replace("/admin.html");
      return null;
    }
    state.profile = profile;
    const displayName = studentDisplayName(profile);
    els.activeAccount.textContent = displayName;
    els.studentGreeting.textContent = `${greetingForNow()}，${displayName}`;
    return profile;
  } catch {
    clearAuth();
    window.location.replace("/");
    return null;
  }
}

async function loadAgentStatus() {
  const response = await api("/api/agent/status");
  const status = await response.json();
  state.modelName = status.model || "mock";
  if (status.realModelEnabled) {
    setPill(els.modelState, `${status.provider} / ${displayModel(state.modelName)}`, "ok");
  } else {
    setPill(els.modelState, "mock 演示", "warn");
  }
}

function clearWelcome() {
  const empty = els.messages.querySelector(".empty");
  if (empty) empty.remove();
}

function addMessage(role, content, showThinking = false) {
  clearWelcome();
  const row = document.createElement("article");
  row.className = `message ${role}`;
  const thinkingHtml = role === "assistant" && showThinking
    ? `<div class="thinking-bubble"><span class="message-thinking"><span class="message-thinking-dot"></span>thinking</span></div>`
    : "";
  row.innerHTML = `
    <div class="message-role">${role === "user" ? "我" : "CareTrace"}</div>
    ${thinkingHtml}
    <div class="bubble" ${showThinking ? 'style="display:none;"' : ''}></div>
  `;
  const bubble = row.querySelector(".bubble");
  if (bubble && content) bubble.textContent = content;
  els.messages.append(row);
  els.messages.scrollTop = els.messages.scrollHeight;
  return {
    bubble,
    thinking: row.querySelector(".message-thinking"),
    row,
    showBubble() {
      const thinkingBubble = row.querySelector(".thinking-bubble");
      if (thinkingBubble) thinkingBubble.remove();
      if (bubble) bubble.style.display = "";
    }
  };
}

function parseSse(buffer, onEvent) {
  const parts = buffer.split("\n\n");
  const rest = parts.pop();
  for (const part of parts) {
    const dataLine = part.split("\n").find((line) => line.startsWith("data: "));
    if (!dataLine) continue;
    onEvent(JSON.parse(dataLine.slice(6)));
  }
  return rest;
}

async function sendMessage(event) {
  event.preventDefault();
  if (state.sending) return;
  const message = els.messageInput.value.trim();
  if (!message) return;
  state.sending = true;
  els.sendButton.disabled = true;
  setPill(els.sessionBadge, "THINKING", "warn");
  els.sessionBadge.classList.add("thinking");
  els.sessionDot.classList.add("thinking");
  els.messageInput.value = "";
  updateCharCount();
  addMessage("user", message);
  const assistant = addMessage("assistant", "", true);
  let raw = "";

  try {
    const response = await api("/api/chat/stream", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ sessionId: state.sessionId, message })
    });
    const reader = response.body.getReader();
    const decoder = new TextDecoder();
    let buffer = "";
    let streamFailed = false;
    while (true) {
      const { value, done } = await reader.read();
      if (done) break;
      buffer += decoder.decode(value, { stream: true });
      buffer = parseSse(buffer, (eventData) => {
        if (eventData.type === "meta") {
          state.sessionId = eventData.sessionId;
          saveLastSession(eventData.sessionId);
          loadRecentSessions();
        }
        if (eventData.type === "token") {
          assistant.showBubble();
          raw += eventData.content || "";
          assistant.bubble.textContent = raw;
          els.messages.scrollTop = els.messages.scrollHeight;
        }
        if (eventData.type === "error") {
          streamFailed = true;
          if (!raw) assistant.bubble.textContent = eventData.message || "MCP 工具调用失败";
          if (assistant.thinking) assistant.thinking.remove();
          setPill(els.sessionBadge, "ERROR", "danger");
          els.sessionBadge.classList.remove("thinking");
          els.sessionDot.classList.remove("thinking");
        }
      });
    }
    if (!streamFailed) {
      setPill(els.sessionBadge, "DONE", "ok");
      els.sessionBadge.classList.remove("thinking");
      els.sessionDot.classList.remove("thinking");
      if (assistant.thinking) assistant.thinking.remove();
    }
  } catch (error) {
    if (!raw) assistant.bubble.textContent = `发送失败：${error.message}`;
    if (assistant.thinking) assistant.thinking.remove();
    setPill(els.sessionBadge, "ERROR", "danger");
    els.sessionBadge.classList.remove("thinking");
    els.sessionDot.classList.remove("thinking");
    els.messageInput.value = message;
    updateCharCount();
  } finally {
    state.sending = false;
    els.sendButton.disabled = false;
  }
}

function resetSession() {
  state.sessionId = null;
  clearLastSession();
  els.sessionBadge.classList.remove("thinking");
  els.sessionDot.classList.remove("thinking");
  els.messages.innerHTML = `
    <div class="empty student-welcome">
      <span class="welcome-kicker">A FRESH START</span>
      <strong>新会话已经准备好</strong>
      <p>不用延续刚才的话题，你可以从此刻最想说的一件事重新开始。</p>
    </div>
  `;
  els.messageInput.value = "";
  updateCharCount();
  setPill(els.sessionBadge, "READY");
}

async function loadSessionMessages(sessionId) {
  if (!sessionId || state.sending) return;
  try {
    const response = await api(`/api/chat/sessions/${encodeURIComponent(sessionId)}`);
    const conversation = await response.json();
    state.sessionId = conversation.sessionId;
    saveLastSession(conversation.sessionId);
    els.messages.innerHTML = "";
    for (const message of conversation.messages) {
      addMessage(message.role.toLowerCase(), message.content);
    }
    highlightRecentSession(conversation.sessionId);
    setPill(els.sessionBadge, "DONE", "ok");
  } catch (error) {
    clearLastSession();
    state.sessionId = null;
    console.warn("加载历史会话失败", error);
  }
}

function logout() {
  clearAuth();
  window.location.assign("/");
}

document.querySelectorAll("[data-quick]").forEach((button) => {
  button.addEventListener("click", () => {
    els.messageInput.value = button.dataset.quick;
    updateCharCount();
    els.messageInput.focus();
  });
});
els.messageInput.addEventListener("input", updateCharCount);
els.messageInput.addEventListener("keydown", (event) => {
  if ((event.metaKey || event.ctrlKey) && event.key === "Enter") {
    event.preventDefault();
    els.chatForm.dispatchEvent(new Event("submit", { cancelable: true }));
  }
});
els.messageInput.addEventListener("paste", () => {
  setTimeout(() => {
    els.messageInput.scrollTop = els.messageInput.scrollHeight;
    els.messages.scrollTop = els.messages.scrollHeight;
  }, 0);
});
els.chatForm.addEventListener("submit", sendMessage);
els.newSession.addEventListener("click", resetSession);
els.switchAccount.addEventListener("click", logout);

checkHealth();
loadProfile().then((profile) => {
  if (profile) {
    loadAgentStatus();
    loadRecentSessions();
    // 登录后保持空白，不自动加载最后一条历史会话；用户点击侧边栏再加载。
    clearLastSession();
    state.sessionId = null;
  }
});
updateCharCount();
