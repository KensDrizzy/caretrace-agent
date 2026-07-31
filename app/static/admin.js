const AUTH_KEY = "caretrace.auth";

const state = {
  profile: null,
  modelName: "mock",
  dashboard: {
    reports: [],
    cases: [],
    excel: [],
    alerts: []
  }
};

const els = {
  serviceState: document.querySelector("#serviceState"),
  modelState: document.querySelector("#modelState"),
  activeAccount: document.querySelector("#activeAccount"),
  switchAccount: document.querySelector("#switchAccount"),
  refreshAdmin: document.querySelector("#refreshAdmin"),
  metricReports: document.querySelector("#metricReports"),
  metricHigh: document.querySelector("#metricHigh"),
  metricCases: document.querySelector("#metricCases"),
  metricExcel: document.querySelector("#metricExcel"),
  metricAlerts: document.querySelector("#metricAlerts"),
  metricHighMeta: document.querySelector("#metricHighMeta"),
  metricCasesMeta: document.querySelector("#metricCasesMeta"),
  metricExcelMeta: document.querySelector("#metricExcelMeta"),
  metricAlertsMeta: document.querySelector("#metricAlertsMeta"),
  lastUpdated: document.querySelector("#lastUpdated"),
  trendTotal: document.querySelector("#trendTotal"),
  activityTrend: document.querySelector("#activityTrend"),
  riskDistribution: document.querySelector("#riskDistribution"),
  emotionDistribution: document.querySelector("#emotionDistribution"),
  intentDistribution: document.querySelector("#intentDistribution"),
  toolDistribution: document.querySelector("#toolDistribution"),
  casesCount: document.querySelector("#casesCount"),
  reportsCount: document.querySelector("#reportsCount"),
  excelCount: document.querySelector("#excelCount"),
  alertCount: document.querySelector("#alertCount"),
  cases: document.querySelector("#cases"),
  reports: document.querySelector("#reports"),
  excelRecords: document.querySelector("#excelRecords"),
  alertRecords: document.querySelector("#alertRecords"),
  conversationState: document.querySelector("#conversationState"),
  conversationDetail: document.querySelector("#conversationDetail"),
  knowledgeState: document.querySelector("#knowledgeState"),
  knowledgeUploadForm: document.querySelector("#knowledgeUploadForm"),
  knowledgeFile: document.querySelector("#knowledgeFile"),
  knowledgeUploadState: document.querySelector("#knowledgeUploadState"),
  rebuildVector: document.querySelector("#rebuildVector"),
  backupVector: document.querySelector("#backupVector")
};

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

function displayTime(value) {
  return value ? new Date(value).toLocaleString() : "";
}

function formatShortDate(value) {
  return new Intl.DateTimeFormat("zh-CN", { month: "numeric", day: "numeric" }).format(value);
}

function percentage(value, total) {
  return total ? Math.round((value / total) * 100) : 0;
}

function statusLabel(status) {
  const labels = {
    SUCCESS: "成功",
    FAILED: "失败",
    OPEN: "待跟进",
    ALERT_SENT: "已预警",
    ACKNOWLEDGED: "已确认"
  };
  return labels[(status || "").toUpperCase()] || status || "未知";
}

function roleLabel(role) {
  const value = (role || "").toUpperCase();
  if (value === "USER") return "学生";
  if (value === "ASSISTANT") return "CareTrace";
  if (value === "SYSTEM") return "系统";
  return role || "未知角色";
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
    if (!isAdmin(profile)) {
      window.location.replace("/student.html");
      return null;
    }
    state.profile = profile;
    els.activeAccount.textContent = profile.displayName || profile.username;
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

async function loadAdminDashboard() {
  els.refreshAdmin.disabled = true;
  els.refreshAdmin.textContent = "刷新中";
  try {
    const [reportsRes, casesRes, excelRes, alertsRes] = await Promise.all([
      api("/api/admin/reports"),
      api("/api/admin/cases"),
      api("/api/admin/excel-records"),
      api("/api/admin/alerts")
    ]);
    const reports = await reportsRes.json();
    const cases = await casesRes.json();
    const excel = await excelRes.json();
    const alerts = await alertsRes.json();
    state.dashboard = { reports, cases, excel, alerts };

    const highRisk = reports.filter((item) => item.riskLevel === "HIGH").length;
    const openCases = cases.filter((item) => item.status === "OPEN").length;
    const excelSuccess = excel.filter((item) => item.status === "SUCCESS").length;
    const alertSuccess = alerts.filter((item) => item.status === "SUCCESS").length;

    els.metricReports.textContent = reports.length;
    els.metricHigh.textContent = highRisk;
    els.metricCases.textContent = cases.length;
    els.metricExcel.textContent = excel.length;
    els.metricAlerts.textContent = alerts.length;
    els.metricHighMeta.textContent = `占比 ${percentage(highRisk, reports.length)}%`;
    els.metricCasesMeta.textContent = `待跟进 ${openCases}`;
    els.metricExcelMeta.textContent = `成功 ${excelSuccess}`;
    els.metricAlertsMeta.textContent = `发送成功 ${alertSuccess}`;
    els.lastUpdated.textContent = `更新于 ${new Date().toLocaleTimeString("zh-CN", { hour: "2-digit", minute: "2-digit" })}`;
    els.casesCount.textContent = `${cases.length} 个个案`;
    els.reportsCount.textContent = `${reports.length} 条报告`;
    els.excelCount.textContent = `${excel.length} 条`;
    els.alertCount.textContent = `${alerts.length} 条`;

    renderTrend(reports);
    renderAnalytics(reports, excel, alerts);
    renderCases(cases);
    renderReports(reports);
    renderToolRecords(els.excelRecords, excel, "excel");
    renderToolRecords(els.alertRecords, alerts, "alert");
  } catch (error) {
    els.lastUpdated.textContent = `刷新失败：${error.message}`;
  } finally {
    els.refreshAdmin.disabled = false;
    els.refreshAdmin.textContent = "刷新数据";
  }
}

async function loadKnowledgeStatus() {
  try {
    const response = await api("/api/admin/knowledge/status");
    const status = await response.json();
    const vector = status.vectorAvailable ? `向量 ${status.vectorChunks ?? 0}` : "向量不可用";
    els.knowledgeState.textContent = `DB ${status.databaseChunks} 片段 · ${vector}`;
  } catch (error) {
    els.knowledgeState.textContent = `读取失败：${error.message}`;
  }
}

function localDateKey(value) {
  const date = new Date(value);
  const year = date.getFullYear();
  const month = String(date.getMonth() + 1).padStart(2, "0");
  const day = String(date.getDate()).padStart(2, "0");
  return `${year}-${month}-${day}`;
}

function renderTrend(reports) {
  const days = [];
  const today = new Date();
  today.setHours(0, 0, 0, 0);
  for (let offset = 6; offset >= 0; offset -= 1) {
    const date = new Date(today);
    date.setDate(today.getDate() - offset);
    days.push({
      date,
      key: localDateKey(date),
      count: 0
    });
  }
  const dayMap = new Map(days.map((item) => [item.key, item]));
  for (const report of reports) {
    const day = dayMap.get(localDateKey(report.createdAt));
    if (day) day.count += 1;
  }
  const max = Math.max(1, ...days.map((item) => item.count));
  const total = days.reduce((sum, item) => sum + item.count, 0);
  els.trendTotal.textContent = `共 ${total} 条`;
  els.activityTrend.innerHTML = "";

  for (const item of days) {
    const column = document.createElement("div");
    column.className = "trend-column";
    column.title = `${formatShortDate(item.date)}：${item.count} 条报告`;

    const value = document.createElement("strong");
    value.textContent = item.count;
    const track = document.createElement("div");
    track.className = "trend-track";
    const bar = document.createElement("span");
    bar.style.height = `${item.count ? Math.max(12, (item.count / max) * 100) : 4}%`;
    track.append(bar);
    const label = document.createElement("small");
    label.textContent = formatShortDate(item.date);

    column.append(value, track, label);
    els.activityTrend.append(column);
  }
}

function countBy(items, key) {
  return items.reduce((result, item) => {
    const value = String(item[key] || "UNKNOWN").toUpperCase();
    result[value] = (result[value] || 0) + 1;
    return result;
  }, {});
}

function renderDistribution(container, definitions, counts) {
  container.innerHTML = "";
  const total = definitions.reduce((sum, item) => sum + (counts[item.key] || 0), 0);
  for (const item of definitions) {
    const count = counts[item.key] || 0;
    const row = document.createElement("div");
    row.className = "distribution-row";

    const label = document.createElement("span");
    label.textContent = item.label;
    const track = document.createElement("div");
    track.className = "distribution-track";
    const fill = document.createElement("span");
    fill.className = `distribution-fill tone-${item.tone}`;
    fill.style.width = `${percentage(count, total)}%`;
    track.append(fill);
    const value = document.createElement("strong");
    value.textContent = count;

    row.append(label, track, value);
    container.append(row);
  }
}

function renderAnalytics(reports, excel, alerts) {
  renderDistribution(els.riskDistribution, [
    { key: "LOW", label: "低风险", tone: "teal" },
    { key: "MEDIUM", label: "中风险", tone: "amber" },
    { key: "HIGH", label: "高风险", tone: "red" }
  ], countBy(reports, "riskLevel"));

  renderDistribution(els.emotionDistribution, [
    { key: "NORMAL", label: "平稳", tone: "teal" },
    { key: "ANXIETY", label: "焦虑", tone: "amber" },
    { key: "DEPRESSED", label: "低落", tone: "orange" },
    { key: "HIGH_RISK", label: "高风险", tone: "red" }
  ], countBy(reports, "emotion"));

  renderDistribution(els.intentDistribution, [
    { key: "CHAT", label: "日常交流", tone: "blue" },
    { key: "CONSULT", label: "心理咨询", tone: "amber" },
    { key: "RISK", label: "风险干预", tone: "red" }
  ], countBy(reports, "intent"));

  const toolCounts = {
    EXCEL_SUCCESS: excel.filter((item) => item.status === "SUCCESS").length,
    EXCEL_FAILED: excel.filter((item) => item.status === "FAILED").length,
    ALERT_SUCCESS: alerts.filter((item) => item.status === "SUCCESS").length,
    ALERT_FAILED: alerts.filter((item) => item.status === "FAILED").length
  };
  renderDistribution(els.toolDistribution, [
    { key: "EXCEL_SUCCESS", label: "台账成功", tone: "teal" },
    { key: "EXCEL_FAILED", label: "台账失败", tone: "red" },
    { key: "ALERT_SUCCESS", label: "预警成功", tone: "blue" },
    { key: "ALERT_FAILED", label: "预警失败", tone: "red" }
  ], toolCounts);
}

function renderToolRecords(container, records, kind) {
  container.innerHTML = "";
  if (!records.length) {
    const empty = document.createElement("div");
    empty.className = "delivery-empty";
    empty.textContent = kind === "excel" ? "暂无 Excel 写入记录" : "暂无预警发送记录";
    container.append(empty);
    return;
  }
  for (const item of records.slice(0, 4)) {
    const card = document.createElement("article");
    card.className = "delivery-record";

    const head = document.createElement("div");
    const title = document.createElement("strong");
    title.textContent = `报告 #${item.reportId}`;
    const badge = document.createElement("span");
    badge.className = `status-badge status-${String(item.status || "").toLowerCase()}`;
    badge.textContent = statusLabel(item.status);
    head.append(title, badge);

    const meta = document.createElement("small");
    meta.textContent = kind === "excel"
      ? `${displayTime(item.createdAt)}${item.filePath ? ` · ${item.filePath}` : ""}`
      : `${item.channel || "通知"} · ${item.recipient || "未配置接收人"} · ${displayTime(item.createdAt)}`;
    const message = document.createElement("p");
    message.textContent = item.message || "无附加信息";
    card.append(head, meta, message);
    container.append(card);
  }
}

function renderCases(cases) {
  els.cases.innerHTML = "";
  if (!cases.length) {
    els.cases.innerHTML = `<div class="empty small"><strong>暂无个案</strong><p>中高风险报告会自动创建风险个案。</p></div>`;
    return;
  }
  for (const item of cases) {
    const card = document.createElement("article");
    card.className = `case-card risk-${item.riskLevel.toLowerCase()}`;

    const head = document.createElement("div");
    head.className = "case-head";
    const titleLine = document.createElement("div");
    titleLine.className = "record-title-line";
    const title = document.createElement("strong");
    title.textContent = `个案 #${item.id}`;
    const badge = document.createElement("span");
    badge.className = `status-badge case-status-${String(item.status || "").toLowerCase()}`;
    badge.textContent = statusLabel(item.status);
    titleLine.append(title, badge);
    const time = document.createElement("span");
    time.textContent = displayTime(item.updatedAt);
    head.append(titleLine, time);

    const meta = document.createElement("small");
    meta.textContent = `报告 #${item.reportId} · ${item.riskLevel} · 负责人：${item.owner || "未分配"}`;
    const summary = document.createElement("p");
    summary.textContent = item.summary || "";
    const handoff = document.createElement("pre");
    handoff.textContent = item.handoffSummary || "";

    card.append(head, meta, summary, handoff);
    els.cases.append(card);
  }
}

function renderReports(reports) {
  els.reports.innerHTML = "";
  if (!reports.length) {
    els.reports.innerHTML = `<div class="empty small"><strong>暂无报告</strong><p>学生咨询或风险场景会在这里沉淀记录。</p></div>`;
    return;
  }
  for (const item of reports) {
    const card = document.createElement("button");
    card.type = "button";
    card.className = `report risk-${item.riskLevel.toLowerCase()}`;
    card.dataset.sessionId = item.sessionId;

    const head = document.createElement("div");
    head.className = "report-head";
    const titleLine = document.createElement("div");
    titleLine.className = "record-title-line";
    const title = document.createElement("strong");
    title.textContent = item.displayName || item.username || "未知学生";
    const badge = document.createElement("span");
    badge.className = `risk-badge risk-${String(item.riskLevel || "low").toLowerCase()}`;
    badge.textContent = item.riskLevel || "LOW";
    titleLine.append(title, badge);
    const time = document.createElement("span");
    time.textContent = displayTime(item.createdAt);
    head.append(titleLine, time);

    const summary = document.createElement("p");
    summary.textContent = item.summary;
    const content = document.createElement("small");
    content.textContent = `${item.emotion || "NORMAL"} / ${item.intent || "CHAT"} · ${item.content}`;
    const action = document.createElement("span");
    action.className = "report-action";
    action.textContent = "查看会话档案";

    card.append(head, summary, content, action);
    card.addEventListener("click", () => loadConversation(item.sessionId));
    els.reports.append(card);
  }
}

async function loadConversation(sessionId) {
  if (!sessionId) {
    els.conversationState.textContent = "该报告缺少会话 ID";
    return;
  }
  els.conversationState.textContent = "正在读取...";
  els.conversationDetail.innerHTML = `<div class="empty small"><strong>加载中</strong><p>正在读取历史消息。</p></div>`;
  for (const card of els.reports.querySelectorAll(".report")) {
    card.classList.toggle("active", card.dataset.sessionId === sessionId);
  }
  try {
    const response = await api(`/api/admin/conversations/${encodeURIComponent(sessionId)}`);
    const conversation = await response.json();
    renderConversation(conversation);
  } catch (error) {
    els.conversationState.textContent = "读取失败";
    els.conversationDetail.innerHTML = "";
    const empty = document.createElement("div");
    empty.className = "empty small";
    const title = document.createElement("strong");
    title.textContent = "无法查看档案";
    const detail = document.createElement("p");
    detail.textContent = error.message;
    empty.append(title, detail);
    els.conversationDetail.append(empty);
  }
}

function renderConversation(conversation) {
  const messages = conversation.messages || [];
  els.conversationState.textContent = `${conversation.title || conversation.sessionId} · ${messages.length} 条消息`;
  els.conversationDetail.innerHTML = "";
  if (!messages.length) {
    els.conversationDetail.innerHTML = `<div class="empty small"><strong>暂无消息</strong><p>这个会话还没有写入消息记录。</p></div>`;
    return;
  }
  for (const message of messages) {
    const role = (message.role || "").toLowerCase();
    const row = document.createElement("article");
    row.className = `conversation-message ${role}`;

    const meta = document.createElement("div");
    meta.className = "conversation-meta";
    const label = document.createElement("strong");
    label.textContent = roleLabel(message.role);
    const time = document.createElement("span");
    time.textContent = displayTime(message.createdAt);
    meta.append(label, time);

    const bubble = document.createElement("div");
    bubble.className = "conversation-bubble";
    bubble.textContent = message.content || "";

    row.append(meta, bubble);
    els.conversationDetail.append(row);
  }
}

async function uploadKnowledgeFile(event) {
  event.preventDefault();
  const file = els.knowledgeFile.files?.[0];
  if (!file) {
    els.knowledgeUploadState.textContent = "请先选择文件";
    return;
  }
  const data = new FormData();
  data.append("file", file);
  els.knowledgeUploadState.textContent = "正在切分入库...";
  try {
    const response = await api("/api/admin/knowledge/file", { method: "POST", body: data });
    const result = await response.json();
    els.knowledgeUploadState.textContent = `${result.source} 已入库 ${result.chunks} 个片段`;
    els.knowledgeFile.value = "";
    loadKnowledgeStatus();
  } catch (error) {
    els.knowledgeUploadState.textContent = `上传失败：${error.message}`;
  }
}

async function runKnowledgeAction(kind) {
  const isBackup = kind === "backup";
  const button = isBackup ? els.backupVector : els.rebuildVector;
  const endpoint = isBackup ? "/api/admin/knowledge/backup" : "/api/admin/knowledge/rebuild-vector";
  const original = button.textContent;
  button.disabled = true;
  els.knowledgeUploadState.textContent = isBackup ? "正在备份向量索引..." : "正在重建向量索引...";
  try {
    const response = await api(endpoint, { method: "POST" });
    const result = await response.json();
    els.knowledgeUploadState.textContent = isBackup
      ? `备份完成：${result.snapshot}`
      : `重建完成：${result.indexedChunks} 个片段`;
    loadKnowledgeStatus();
  } catch (error) {
    els.knowledgeUploadState.textContent = `操作失败：${error.message}`;
  } finally {
    button.disabled = false;
    button.textContent = original;
  }
}

function logout() {
  clearAuth();
  window.location.assign("/");
}

els.switchAccount.addEventListener("click", logout);
els.refreshAdmin.addEventListener("click", () => {
  loadAdminDashboard();
  loadKnowledgeStatus();
});
els.knowledgeUploadForm.addEventListener("submit", uploadKnowledgeFile);
els.rebuildVector.addEventListener("click", () => runKnowledgeAction("rebuild"));
els.backupVector.addEventListener("click", () => runKnowledgeAction("backup"));

checkHealth();
loadProfile().then((profile) => {
  if (!profile) return;
  loadAgentStatus();
  loadAdminDashboard();
  loadKnowledgeStatus();
});
