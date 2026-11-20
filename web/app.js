/* app.js — Clean Client Controller for the ADMET GraphGPS Scientific Workbench */
(() => {
  "use strict";

  const tabs = Array.from(document.querySelectorAll('[role="tab"]'));
  const panels = Array.from(document.querySelectorAll('[role="tabpanel"]'));

  let currentSmiles = "CC(=O)Oc1ccccc1C(=O)O";
  let presetList = [];
  let hasRunXai = false;
  let hasRunOpt = false;
  let hasLoadedBenchmarks = false;

  // =========================================================================
  // 1. Accessible Tabs Controller (Roving Focus + hidden="until-found" + beforematch)
  // =========================================================================
  function selectTab(targetTab, focusTab = false) {
    const targetPanelId = targetTab.dataset.panel;
    for (const tab of tabs) {
      const active = tab === targetTab;
      tab.setAttribute("aria-selected", active ? "true" : "false");
      tab.setAttribute("tabindex", active ? "0" : "-1");
      tab.toggleAttribute("focusgroupstart", active);
    }
    for (const panel of panels) {
      const match = panel.id === targetPanelId;
      if (match) {
        panel.removeAttribute("hidden");
        panel.setAttribute("tabindex", "0");
      } else {
        panel.setAttribute("hidden", "until-found");
        panel.removeAttribute("tabindex");
      }
    }
    targetTab.scrollIntoView({ block: "nearest", inline: "nearest" });
    if (focusTab) targetTab.focus();

    if (targetPanelId === "panel-saliency" && !hasRunXai) {
      document.getElementById("xai-smiles-input").value = currentSmiles;
      runSaliency(currentSmiles, document.getElementById("xai-task").value);
    } else if (targetPanelId === "panel-optimizer" && !hasRunOpt) {
      document.getElementById("opt-smiles-input").value = currentSmiles;
      runOptimization();
    } else if (targetPanelId === "panel-benchmarks" && !hasLoadedBenchmarks) {
      loadBenchmarks();
    }
  }

  tabs.forEach((tab, idx) => {
    tab.addEventListener("click", () => selectTab(tab, false));
    tab.addEventListener("keydown", (e) => {
      let nextIdx = null;
      if (e.key === "ArrowRight") nextIdx = (idx + 1) % tabs.length;
      else if (e.key === "ArrowLeft") nextIdx = (idx - 1 + tabs.length) % tabs.length;
      else if (e.key === "Home") nextIdx = 0;
      else if (e.key === "End") nextIdx = tabs.length - 1;
      if (nextIdx !== null) {
        e.preventDefault();
        selectTab(tabs[nextIdx], true);
      }
    });
  });

  panels.forEach((panel) => {
    panel.addEventListener("beforematch", () => {
      const assocTab = tabs.find((t) => t.dataset.panel === panel.id);
      if (assocTab) selectTab(assocTab, false);
    });
  });

  // =========================================================================
  // 2. Validate Input After Interaction + Accessible ARIA Sync (modern-web-guidance)
  // =========================================================================
  const announcer = document.createElement("div");
  announcer.setAttribute("aria-live", "polite");
  announcer.classList.add("visually-hidden");
  document.body.append(announcer);

  let announceTimer;
  let announcedField = null;
  const dirtyMap = new WeakMap();
  const flagged = new WeakSet();
  const announced = new WeakSet();
  const supportsUserInvalid = CSS.supports("selector(:user-invalid)");

  function linkError(el, shown) {
    const errorId = el.dataset.error;
    if (!errorId) return;
    const ids = (el.getAttribute("aria-describedby") || "")
      .split(/\s+/)
      .filter((id) => id && id !== errorId);
    if (shown) ids.push(errorId);
    if (ids.length) el.setAttribute("aria-describedby", ids.join(" "));
    else el.removeAttribute("aria-describedby");
  }

  function announceError(el) {
    const errEl = document.getElementById(el.dataset.error);
    if (!errEl) return;
    const labelText = el.labels?.[0]?.textContent?.trim() || "Input";
    clearTimeout(announceTimer);
    announcer.textContent = "";
    announcedField = el;
    announceTimer = setTimeout(() => {
      announcer.textContent = `${labelText}: ${errEl.textContent.trim()}`;
    }, 450);
  }

  function syncValidationState(el, announce = false, forceInvalid = false) {
    if (!el.matches?.("input")) return;
    let isInvalid = false;
    if (supportsUserInvalid && !forceInvalid) {
      isInvalid = el.matches(":user-invalid");
    } else {
      const st = dirtyMap.get(el) || { interacted: false, blurred: false };
      isInvalid = (forceInvalid || (st.interacted && st.blurred)) && !el.validity.valid;
      el.classList.toggle("user-invalid-fallback", isInvalid);
      el.classList.toggle("user-valid-fallback", !isInvalid && st.blurred && el.validity.valid);
    }

    if (isInvalid) {
      if (announce && !announced.has(el)) {
        announceError(el);
        announced.add(el);
      }
      el.setAttribute("aria-invalid", "true");
      linkError(el, true);
      flagged.add(el);
    } else if (flagged.has(el) || el.getAttribute("aria-invalid") !== "true") {
      el.setAttribute("aria-invalid", "false");
      linkError(el, false);
      flagged.delete(el);
      announced.delete(el);
      if (announcedField === el) {
        clearTimeout(announceTimer);
        announcer.textContent = "";
      }
    }
  }

  document.addEventListener(
    "blur",
    (e) => {
      const el = e.target;
      if (!el.matches?.("input")) return;
      const st = dirtyMap.get(el) || { interacted: false, blurred: false };
      st.blurred = true;
      dirtyMap.set(el, st);
      syncValidationState(el, true);
    },
    true
  );

  document.addEventListener("input", (e) => {
    const el = e.target;
    if (!el.matches?.("input")) return;
    const st = dirtyMap.get(el) || { interacted: false, blurred: false };
    st.interacted = true;
    dirtyMap.set(el, st);
    if (el.hasAttribute("aria-invalid")) syncValidationState(el, false);
  });

  function validateFormBeforeFetch(formEl) {
    const inputs = Array.from(formEl.querySelectorAll("input[required]"));
    let allOk = true;
    let firstBad = null;
    for (const inp of inputs) {
      inp.value = inp.value.trim();
      const st = dirtyMap.get(inp) || { interacted: true, blurred: true };
      st.interacted = true;
      st.blurred = true;
      dirtyMap.set(inp, st);
      if (!inp.checkValidity()) {
        allOk = false;
        if (!firstBad) firstBad = inp;
        syncValidationState(inp, true, true);
      } else {
        syncValidationState(inp, false, false);
      }
    }
    if (firstBad) firstBad.focus();
    return allOk;
  }

  // =========================================================================
  // 3. Helpers
  // =========================================================================
  function setBadge(el, text, tone) {
    el.textContent = text;
    el.className = `badge badge-${tone}`;
  }

  function setBar(el, pct, colorVar) {
    el.style.width = `${Math.max(0, Math.min(100, pct))}%`;
    el.style.backgroundColor = colorVar;
  }

  function clearEl(el) {
    while (el.firstChild) el.removeChild(el.firstChild);
  }

  function makeCell(tag, text) {
    const cell = document.createElement(tag);
    cell.textContent = text;
    return cell;
  }

  async function apiJson(url, options = {}) {
    const res = await fetch(url, options);
    const data = await res.json();
    if (!res.ok) {
      throw new Error(data.detail || `Request failed (${res.status})`);
    }
    return data;
  }

  function renderPresetBar(containerId, onPick) {
    const container = document.getElementById(containerId);
    clearEl(container);
    for (const p of presetList) {
      const btn = document.createElement("button");
      btn.type = "button";
      btn.className = "chip-btn";
      btn.textContent = `${p.name} (${p.role})`;
      btn.addEventListener("click", () => onPick(p.smiles));
      container.appendChild(btn);
    }
  }

  // =========================================================================
  // 4. Tab 1 — ADMET Profiler
  // =========================================================================
  async function runPrediction(smiles) {
    const alertEl = document.getElementById("profiler-alert");
    const btn = document.getElementById("predict-btn");
    alertEl.hidden = true;
    btn.disabled = true;
    btn.textContent = "Analyzing...";

    try {
      const data = await apiJson("/api/predict", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ smiles }),
      });

      currentSmiles = data.graph_info.canonical_smiles;
      document.getElementById("smiles-input").value = currentSmiles;
      document.getElementById("mol-image").src = data.image_data_uri;
      document.getElementById("canon-smiles").textContent = currentSmiles;
      document.getElementById("predict-latency").textContent = `${data.latency_ms} ms`;
      document.getElementById("meta-formula").textContent = data.graph_info.formula;
      document.getElementById("meta-atoms-bonds").textContent = `${data.graph_info.num_atoms} / ${data.graph_info.num_bonds}`;
      document.getElementById("meta-scaffold").textContent = data.graph_info.scaffold;

      document.getElementById("geom-npr1").textContent = data.geometry_3d.npr1.toFixed(3);
      document.getElementById("geom-npr2").textContent = data.geometry_3d.npr2.toFixed(3);
      document.getElementById("geom-asph").textContent = data.geometry_3d.asphericity.toFixed(3);
      document.getElementById("geom-rog").textContent = `${data.geometry_3d.radius_of_gyration.toFixed(2)} A`;

      const p = data.predictions;
      const rd = data.rdkit_audit;

      document.getElementById("val-bio").textContent = `${p.bioavailability.toFixed(1)}%`;
      setBar(document.getElementById("bar-bio"), p.bioavailability, p.bioavailability >= 60 ? "var(--safe)" : "var(--warn)");
      setBadge(document.getElementById("tag-bio"), p.bioavailability >= 60 ? "High F >= 20%" : "Moderate / Low", p.bioavailability >= 60 ? "safe" : "warn");

      document.getElementById("val-bbb").textContent = p.bbb.toFixed(3);
      setBar(document.getElementById("bar-bbb"), p.bbb * 100, "var(--accent)");
      setBadge(document.getElementById("tag-bbb"), p.bbb >= 0.5 ? "BBB+ (CNS)" : "BBB- (Peripheral)", p.bbb >= 0.5 ? "accent" : "neutral");

      document.getElementById("val-herg").textContent = p.herg.toFixed(3);
      const hTone = p.herg < 0.3 ? "safe" : p.herg < 0.6 ? "warn" : "danger";
      setBar(document.getElementById("bar-herg"), p.herg * 100, `var(--${hTone})`);
      setBadge(document.getElementById("tag-herg"), p.herg < 0.3 ? "Low Risk" : p.herg < 0.6 ? "Moderate Risk" : "High Blocker Risk", hTone);

      document.getElementById("val-qed").textContent = p.qed.toFixed(3);
      setBar(document.getElementById("bar-qed"), p.qed * 100, "var(--accent)");
      document.getElementById("audit-qed").textContent = `RDKit Exact: ${rd.qed.toFixed(3)}`;

      document.getElementById("val-sa").textContent = `${p.sascore.toFixed(2)} / 10`;
      setBar(document.getElementById("bar-sa"), (p.sascore / 10.0) * 100, p.sascore <= 4.0 ? "var(--safe)" : "var(--warn)");
      document.getElementById("audit-sa").textContent = `RDKit Exact: ${rd.sascore.toFixed(2)}`;

      const mc = data.medchem;
      setBadge(
        document.getElementById("badge-lipinski"),
        mc.lipinski_pass ? `Lipinski Pass (${mc.lipinski_violations} viol.)` : `Lipinski Fail (${mc.lipinski_violations} viol.)`,
        mc.lipinski_pass ? "safe" : "danger"
      );
      setBadge(document.getElementById("badge-pains"), mc.pains_pass ? "PAINS Clean" : "PAINS Alert", mc.pains_pass ? "safe" : "danger");

      document.getElementById("mc-mw").textContent = `${mc.mw.toFixed(1)} Da`;
      document.getElementById("mc-logp").textContent = mc.logp.toFixed(2);
      document.getElementById("mc-hb").textContent = `${mc.hbd} / ${mc.hba}`;
      document.getElementById("mc-tpsa").textContent = mc.tpsa.toFixed(1);
      document.getElementById("mc-rot").textContent = String(mc.rotatable_bonds);
    } catch (err) {
      alertEl.textContent = `Error: ${err.message}`;
      alertEl.hidden = false;
    } finally {
      btn.disabled = false;
      btn.textContent = "Analyze Molecule";
    }
  }

  document.getElementById("profiler-form").addEventListener("submit", (e) => {
    e.preventDefault();
    if (!validateFormBeforeFetch(e.target)) return;
    runPrediction(document.getElementById("smiles-input").value);
  });

  document.getElementById("copy-smiles-btn").addEventListener("click", async () => {
    const btn = document.getElementById("copy-smiles-btn");
    try {
      await navigator.clipboard.writeText(currentSmiles);
      btn.textContent = "Copied";
      setTimeout(() => { btn.textContent = "Copy"; }, 1200);
    } catch {
      btn.textContent = "Copy";
    }
  });

  document.getElementById("goto-xai-btn").addEventListener("click", () => {
    document.getElementById("xai-smiles-input").value = currentSmiles;
    hasRunXai = false;
    selectTab(document.getElementById("tab-saliency"), false);
  });

  document.getElementById("goto-opt-btn").addEventListener("click", () => {
    document.getElementById("opt-smiles-input").value = currentSmiles;
    hasRunOpt = false;
    selectTab(document.getElementById("tab-optimizer"), false);
  });

  // =========================================================================
  // 5. Tab 2 — XAI Saliency
  // =========================================================================
  async function runSaliency(smiles, task) {
    const alertEl = document.getElementById("xai-alert");
    const btn = document.getElementById("xai-btn");
    alertEl.hidden = true;
    btn.disabled = true;
    btn.textContent = "Computing...";

    try {
      const data = await apiJson("/api/saliency", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ smiles, task }),
      });
      hasRunXai = true;
      currentSmiles = data.canonical_smiles;
      document.getElementById("xai-smiles-input").value = currentSmiles;
      document.getElementById("xai-image").src = data.image_data_uri;

      const rawVal = data.predictions[task];
      const valText = task === "bioavailability" ? `${rawVal.toFixed(1)}%` : rawVal.toFixed(3);
      document.getElementById("xai-pred-badge").textContent = `${task.toUpperCase()}: ${valText}`;

      const topAtom = data.top_atoms[0];
      document.getElementById("xai-interpretation").textContent =
        `Peak gradient sensitivity focuses on atom #${topAtom.atom_idx} (${topAtom.symbol}, degree ${topAtom.degree}` +
        `${topAtom.is_aromatic ? ", aromatic" : ""}${topAtom.formal_charge !== 0 ? `, charge ${topAtom.formal_charge}` : ""}) ` +
        `with normalized attribution ${topAtom.attribution.toFixed(3)}.`;

      const tbody = document.getElementById("xai-table-body");
      clearEl(tbody);
      for (const atom of data.top_atoms) {
        const tr = document.createElement("tr");
        if (atom.attribution >= 0.7) tr.classList.add("row-highlight");
        tr.appendChild(makeCell("th", `#${atom.atom_idx}`));
        tr.lastChild.setAttribute("scope", "row");
        tr.appendChild(makeCell("td", atom.symbol));
        tr.appendChild(makeCell("td", String(atom.degree)));
        tr.appendChild(makeCell("td", atom.formal_charge >= 0 ? `+${atom.formal_charge}` : String(atom.formal_charge)));
        tr.appendChild(makeCell("td", atom.is_aromatic ? "Yes" : "No"));
        tr.appendChild(makeCell("td", atom.in_ring ? "Yes" : "No"));
        tr.appendChild(makeCell("td", atom.attribution.toFixed(4)));
        tbody.appendChild(tr);
      }
    } catch (err) {
      alertEl.textContent = `Error: ${err.message}`;
      alertEl.hidden = false;
    } finally {
      btn.disabled = false;
      btn.textContent = "Compute Attribution";
    }
  }

  document.getElementById("saliency-form").addEventListener("submit", (e) => {
    e.preventDefault();
    if (!validateFormBeforeFetch(e.target)) return;
    runSaliency(
      document.getElementById("xai-smiles-input").value,
      document.getElementById("xai-task").value
    );
  });

  // =========================================================================
  // 6. Tab 3 — Pareto Lead Optimizer
  // =========================================================================
  function inspectCandidate(c) {
    const rankEl = document.getElementById("insp-rank");
    if (c.is_seed) setBadge(rankEl, "Seed Lead", "danger");
    else if (c.pareto_rank === 1) setBadge(rankEl, "Front 1 (Optimal)", "accent");
    else setBadge(rankEl, `Front ${c.pareto_rank}`, "neutral");

    document.getElementById("insp-smiles").textContent = c.smiles;
    document.getElementById("insp-metrics").textContent =
      `Desirability: ${c.desirability.toFixed(3)}  |  Bioavail: ${c.bioavailability.toFixed(1)}%  |  ` +
      `hERG: ${c.herg.toFixed(3)}  |  BBB: ${c.bbb.toFixed(3)}  |  RDKit QED: ${c.qed_rdkit.toFixed(3)}  |  ` +
      `RDKit SA: ${c.sascore_rdkit.toFixed(2)}  |  Sim: ${c.tanimoto_sim.toFixed(2)}`;
  }

  function renderParetoSvg(candidates) {
    const svg = document.getElementById("pareto-svg");
    clearEl(svg);
    const ns = "http://www.w3.org/2000/svg";
    const W = 560, H = 340;
    const pad = { l: 54, r: 22, t: 20, b: 44 };
    const pw = W - pad.l - pad.r;
    const ph = H - pad.t - pad.b;

    const hergs = candidates.map((c) => c.herg);
    const bios = candidates.map((c) => c.bioavailability);
    let minX = Math.max(0, Math.min(...hergs) * 0.9);
    let maxX = Math.max(...hergs) * 1.08;
    if (maxX - minX < 0.02) {
      minX = Math.max(0, minX - 0.01);
      maxX = Math.min(1, maxX + 0.02);
    }
    let minY = Math.max(0, Math.min(...bios) - 5);
    let maxY = Math.min(102, Math.max(...bios) + 2);
    if (maxY - minY < 10) minY = Math.max(0, maxY - 12);

    const sx = (v) => pad.l + ((v - minX) / (maxX - minX)) * pw;
    const sy = (v) => pad.t + ph - ((v - minY) / (maxY - minY)) * ph;

    // Grid lines
    for (let i = 0; i <= 4; i++) {
      const gy = pad.t + (ph * i) / 4;
      const line = document.createElementNS(ns, "line");
      line.setAttribute("x1", pad.l);
      line.setAttribute("x2", W - pad.r);
      line.setAttribute("y1", gy);
      line.setAttribute("y2", gy);
      line.setAttribute("stroke", "#e2e8f0");
      line.setAttribute("stroke-dasharray", "3 3");
      svg.appendChild(line);

      const val = maxY - ((maxY - minY) * i) / 4;
      const lbl = document.createElementNS(ns, "text");
      lbl.setAttribute("x", pad.l - 8);
      lbl.setAttribute("y", gy + 3);
      lbl.setAttribute("text-anchor", "end");
      lbl.setAttribute("font-size", "10");
      lbl.setAttribute("fill", "#64748b");
      lbl.textContent = `${val.toFixed(0)}%`;
      svg.appendChild(lbl);
    }

    for (let i = 0; i <= 4; i++) {
      const gx = pad.l + (pw * i) / 4;
      const line = document.createElementNS(ns, "line");
      line.setAttribute("x1", gx);
      line.setAttribute("x2", gx);
      line.setAttribute("y1", pad.t);
      line.setAttribute("y2", pad.t + ph);
      line.setAttribute("stroke", "#e2e8f0");
      line.setAttribute("stroke-dasharray", "3 3");
      svg.appendChild(line);

      const val = minX + ((maxX - minX) * i) / 4;
      const lbl = document.createElementNS(ns, "text");
      lbl.setAttribute("x", gx);
      lbl.setAttribute("y", pad.t + ph + 16);
      lbl.setAttribute("text-anchor", "middle");
      lbl.setAttribute("font-size", "10");
      lbl.setAttribute("fill", "#64748b");
      lbl.textContent = val.toFixed(3);
      svg.appendChild(lbl);
    }

    // Axes lines
    const axis = document.createElementNS(ns, "path");
    axis.setAttribute("d", `M ${pad.l} ${pad.t} V ${pad.t + ph} H ${W - pad.r}`);
    axis.setAttribute("fill", "none");
    axis.setAttribute("stroke", "#94a3b8");
    svg.appendChild(axis);

    const xCap = document.createElementNS(ns, "text");
    xCap.setAttribute("x", pad.l + pw / 2);
    xCap.setAttribute("y", H - 6);
    xCap.setAttribute("text-anchor", "middle");
    xCap.setAttribute("font-size", "11");
    xCap.setAttribute("font-weight", "600");
    xCap.setAttribute("fill", "#334155");
    xCap.textContent = "Predicted hERG Cardiotoxicity Hazard (Lower is Safer)";
    svg.appendChild(xCap);

    const yCap = document.createElementNS(ns, "text");
    yCap.setAttribute("transform", `translate(14, ${pad.t + ph / 2}) rotate(-90)`);
    yCap.setAttribute("text-anchor", "middle");
    yCap.setAttribute("font-size", "11");
    yCap.setAttribute("font-weight", "600");
    yCap.setAttribute("fill", "#334155");
    yCap.textContent = "Bioavailability %";
    svg.appendChild(yCap);

    // Front-1 polyline
    const f1Pts = candidates
      .filter((c) => c.pareto_rank === 1)
      .sort((a, b) => a.herg - b.herg);
    if (f1Pts.length > 1) {
      const poly = document.createElementNS(ns, "polyline");
      poly.setAttribute("points", f1Pts.map((c) => `${sx(c.herg)},${sy(c.bioavailability)}`).join(" "));
      poly.setAttribute("fill", "none");
      poly.setAttribute("stroke", "#1d4ed8");
      poly.setAttribute("stroke-width", "1.5");
      poly.setAttribute("stroke-dasharray", "4 3");
      poly.setAttribute("opacity", "0.75");
      svg.appendChild(poly);
    }

    // Points: dominated -> front1 -> seed
    const ordered = [
      ...candidates.filter((c) => !c.is_seed && c.pareto_rank !== 1),
      ...candidates.filter((c) => !c.is_seed && c.pareto_rank === 1),
      ...candidates.filter((c) => c.is_seed),
    ];

    for (const c of ordered) {
      const cx = sx(c.herg);
      const cy = sy(c.bioavailability);
      const dot = document.createElementNS(ns, "circle");
      dot.setAttribute("cx", cx);
      dot.setAttribute("cy", cy);
      if (c.is_seed) {
        dot.setAttribute("r", "7");
        dot.setAttribute("fill", "#b91c1c");
        dot.setAttribute("stroke", "#0f172a");
        dot.setAttribute("stroke-width", "1.4");
      } else if (c.pareto_rank === 1) {
        dot.setAttribute("r", "5.5");
        dot.setAttribute("fill", "#1d4ed8");
        dot.setAttribute("stroke", "#ffffff");
        dot.setAttribute("stroke-width", "1");
      } else {
        dot.setAttribute("r", "4");
        dot.setAttribute("fill", "#94a3b8");
        dot.setAttribute("opacity", "0.75");
      }
      dot.style.cursor = "pointer";
      dot.addEventListener("click", () => inspectCandidate(c));
      dot.addEventListener("mouseenter", () => inspectCandidate(c));
      svg.appendChild(dot);
    }
  }

  async function runOptimization() {
    const alertEl = document.getElementById("opt-alert");
    const btn = document.getElementById("opt-btn");
    alertEl.hidden = true;
    btn.disabled = true;
    btn.textContent = "Evolving...";

    try {
      const payload = {
        smiles: document.getElementById("opt-smiles-input").value,
        mode: document.getElementById("opt-mode").value,
        generations: Number(document.getElementById("opt-gens").value),
        max_analogues: Number(document.getElementById("opt-pop").value),
        min_similarity: Number(document.getElementById("opt-sim").value),
      };
      const data = await apiJson("/api/optimize", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload),
      });

      hasRunOpt = true;
      currentSmiles = data.seed_smiles;
      document.getElementById("opt-smiles-input").value = currentSmiles;

      const bestCand = data.candidates[0];
      document.getElementById("kpi-total").textContent = String(data.total_candidates);
      document.getElementById("kpi-front1").textContent = String(data.pareto_front1_count);
      document.getElementById("kpi-best").textContent = bestCand ? bestCand.desirability.toFixed(3) : "--";
      document.getElementById("kpi-time").textContent = `${data.latency_ms} ms`;

      renderParetoSvg(data.candidates);
      if (bestCand) inspectCandidate(bestCand);

      // Top 6 Structure Cards
      const cardsWrap = document.getElementById("opt-top-cards");
      clearEl(cardsWrap);
      data.top_cards.forEach((c, idx) => {
        const card = document.createElement("div");
        card.className = "analogue-card";

        const head = document.createElement("div");
        head.className = "analogue-card-head";
        const tag = document.createElement("span");
        setBadge(
          tag,
          c.is_seed ? "Seed Lead" : `Top #${idx} · F${c.pareto_rank}`,
          c.is_seed ? "danger" : c.pareto_rank === 1 ? "accent" : "neutral"
        );
        const des = document.createElement("strong");
        des.style.fontSize = "0.8rem";
        des.textContent = `Desir ${c.desirability.toFixed(3)}`;
        head.append(tag, des);

        const img = document.createElement("img");
        img.src = c.image_data_uri;
        img.alt = c.is_seed ? "Seed lead molecule structure" : `Analogue ${idx} structure`;

        const smiCode = document.createElement("code");
        smiCode.className = "mono-truncate";
        smiCode.title = c.smiles;
        smiCode.textContent = c.smiles;

        const stats = document.createElement("div");
        stats.className = "analogue-stats";
        stats.textContent = `Bio ${c.bioavailability.toFixed(1)}% · hERG ${c.herg.toFixed(3)} · QED ${c.qed_rdkit.toFixed(2)} · SA ${c.sascore_rdkit.toFixed(2)}`;

        const xaiBtn = document.createElement("button");
        xaiBtn.type = "button";
        xaiBtn.className = "btn-secondary btn-sm";
        xaiBtn.textContent = "Inspect in XAI";
        xaiBtn.addEventListener("click", () => {
          currentSmiles = c.smiles;
          document.getElementById("xai-smiles-input").value = c.smiles;
          hasRunXai = false;
          selectTab(document.getElementById("tab-saliency"), false);
        });

        card.append(head, img, smiCode, stats, xaiBtn);
        cardsWrap.appendChild(card);
      });

      // Top 15 Results Table
      const tbody = document.getElementById("opt-table-body");
      clearEl(tbody);
      for (const c of data.candidates.slice(0, 15)) {
        const tr = document.createElement("tr");
        if (c.is_seed) tr.classList.add("row-highlight");
        const th = makeCell("th", c.smiles);
        th.scope = "row";
        th.className = "mono";
        th.style.maxWidth = "220px";
        th.style.overflow = "hidden";
        th.style.textOverflow = "ellipsis";
        th.title = c.smiles;

        tr.appendChild(th);
        tr.appendChild(makeCell("td", c.is_seed ? `Seed (F${c.pareto_rank})` : `F${c.pareto_rank}`));
        tr.appendChild(makeCell("td", `${c.bioavailability.toFixed(1)}%`));
        tr.appendChild(makeCell("td", c.herg.toFixed(3)));
        tr.appendChild(makeCell("td", c.bbb.toFixed(3)));
        tr.appendChild(makeCell("td", `${c.qed.toFixed(2)} / ${c.qed_rdkit.toFixed(2)}`));
        tr.appendChild(makeCell("td", `${c.sascore.toFixed(2)} / ${c.sascore_rdkit.toFixed(2)}`));
        tr.appendChild(makeCell("td", c.tanimoto_sim.toFixed(2)));
        tr.appendChild(makeCell("td", c.desirability.toFixed(3)));

        const actTd = document.createElement("td");
        const actBtn = document.createElement("button");
        actBtn.type = "button";
        actBtn.className = "btn-secondary btn-sm";
        actBtn.textContent = "Profile";
        actBtn.addEventListener("click", () => {
          currentSmiles = c.smiles;
          document.getElementById("smiles-input").value = c.smiles;
          selectTab(document.getElementById("tab-profiler"), false);
          runPrediction(c.smiles);
        });
        actTd.appendChild(actBtn);
        tr.appendChild(actTd);
        tbody.appendChild(tr);
      }
    } catch (err) {
      alertEl.textContent = `Error: ${err.message}`;
      alertEl.hidden = false;
    } finally {
      btn.disabled = false;
      btn.textContent = "Run Evolution";
    }
  }

  document.getElementById("optimize-form").addEventListener("submit", (e) => {
    e.preventDefault();
    if (!validateFormBeforeFetch(e.target)) return;
    runOptimization();
  });

  // =========================================================================
  // 7. Tab 4 — Benchmarks & Ablation
  // =========================================================================
  async function loadBenchmarks() {
    const data = await apiJson("/api/benchmarks");
    hasLoadedBenchmarks = true;

    const ds = data.dataset_stats;
    const ms = data.metrics_summary;
    const ours = ms["SSL-GraphGPS (Ours)"];
    const scratch = ms["GraphGPS-Scratch"];

    document.getElementById("bm-total").textContent = `${(ds.unlabeled_ssl_count + ds.admet_total_count).toLocaleString()} Mols`;
    document.getElementById("bm-scaffolds").textContent = ds.unique_scaffolds.toLocaleString();
    document.getElementById("bm-macro-auc").textContent = ours.aggregate.macro_roc_auc.toFixed(4);
    const gainPp = (ours.aggregate.macro_roc_auc - scratch.aggregate.macro_roc_auc) * 100;
    document.getElementById("bm-ssl-gain").textContent = `+${gainPp.toFixed(2)} pp`;

    const tbody = document.getElementById("bm-table-body");
    clearEl(tbody);
    const order = ["RF-ECFP4", "Standard-GCN", "GraphGPS-Scratch", "SSL-GraphGPS (Ours)"];
    for (const modelName of order) {
      const m = ms[modelName];
      if (!m) continue;
      const tr = document.createElement("tr");
      if (modelName === "SSL-GraphGPS (Ours)") tr.classList.add("row-highlight");
      const th = makeCell("th", modelName);
      th.scope = "row";
      tr.appendChild(th);
      tr.appendChild(makeCell("td", m.bioavailability.roc_auc.toFixed(4)));
      tr.appendChild(makeCell("td", m.bbb.roc_auc.toFixed(4)));
      tr.appendChild(makeCell("td", m.herg.roc_auc.toFixed(4)));
      tr.appendChild(makeCell("td", m.aggregate.macro_roc_auc.toFixed(4)));
      tr.appendChild(makeCell("td", `${m.qed.r2.toFixed(4)} (${m.qed.mae.toFixed(3)})`));
      tr.appendChild(makeCell("td", `${m.sascore.r2.toFixed(4)} (${m.sascore.mae.toFixed(3)})`));
      tr.appendChild(makeCell("td", m.aggregate.composite_val_score.toFixed(4)));
      tbody.appendChild(tr);
    }

    const gallery = document.getElementById("benchmark-gallery");
    clearEl(gallery);
    for (const fig of data.figures) {
      const card = document.createElement("article");
      card.className = "figure-card";
      const h3 = document.createElement("h3");
      h3.textContent = fig.title;
      const p = document.createElement("p");
      p.textContent = fig.caption;
      const img = document.createElement("img");
      img.src = fig.url;
      img.alt = fig.title;
      img.loading = "lazy";
      card.append(h3, p, img);
      gallery.appendChild(card);
    }
  }

  // =========================================================================
  // 8. Bootstrap Workbench
  // =========================================================================
  async function init() {
    try {
      const [health, presetsData] = await Promise.all([
        apiJson("/api/health"),
        apiJson("/api/presets"),
      ]);
      document.getElementById("health-text").textContent =
        `SSL-GraphGPS · ${health.macro_roc_auc} Macro AUC · ${(health.param_count / 1000).toFixed(0)}k params · ${health.device.toUpperCase()}`;

      presetList = presetsData.presets;
      renderPresetBar("profiler-presets", (sm) => {
        document.getElementById("smiles-input").value = sm;
        runPrediction(sm);
      });
      renderPresetBar("xai-presets", (sm) => {
        document.getElementById("xai-smiles-input").value = sm;
        runSaliency(sm, document.getElementById("xai-task").value);
      });
      renderPresetBar("opt-presets", (sm) => {
        document.getElementById("opt-smiles-input").value = sm;
        runOptimization();
      });

      await runPrediction(currentSmiles);

      const hash = window.location.hash.replace("#", "").toLowerCase();
      const hashMap = {
        profiler: "tab-profiler",
        saliency: "tab-saliency",
        optimizer: "tab-optimizer",
        benchmarks: "tab-benchmarks",
      };
      if (hashMap[hash]) {
        selectTab(document.getElementById(hashMap[hash]), false);
      }
    } catch (err) {
      document.getElementById("health-dot").style.background = "var(--danger)";
      document.getElementById("health-text").textContent = `Backend offline: ${err.message}`;
    }
  }

  init();
})();
