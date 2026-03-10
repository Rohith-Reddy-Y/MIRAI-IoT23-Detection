// ── IoT-23 MIRAI Dashboard — Main JS ──
// Static dashboard rendering + Live Detection with two-stage variant display

// ============== VARIANT DATA ==============
const VARIANT_ICONS = {
    "C&C Communication": "📡", "C&C Heartbeat": "💓", "C&C File Download": "📥",
    "C&C Heartbeat Attack": "💔", "Mirai C&C": "🤖", "Torii C&C": "🌀",
    "DDoS Attack": "💥", "Horizontal Port Scan": "🔎", "Port Scan Attack": "⚡",
    "Okiru Malware": "🐛", "Okiru Attack": "💣", "Attack": "⚔️",
    "File Download": "📥", "Benign": "✅", "Unknown Malware": "❓"
};

const VARIANT_DESCRIPTIONS = {
    "C&C Communication": "Command & Control server communication — botnet coordination traffic",
    "C&C Heartbeat": "Periodic C&C keepalive signals — maintains botnet control channel",
    "C&C Heartbeat Attack": "C&C heartbeat combined with active attack payload",
    "C&C File Download": "Malicious payload download from C&C server",
    "Mirai C&C": "Mirai botnet-specific command & control traffic",
    "Torii C&C": "Torii botnet C&C — advanced IoT malware with persistence",
    "DDoS Attack": "Distributed Denial of Service — flood attack traffic",
    "Horizontal Port Scan": "Network reconnaissance — scanning for vulnerable ports",
    "Port Scan Attack": "Aggressive port scanning — precursor to exploitation",
    "Okiru Malware": "Okiru/Satori botnet — Mirai variant targeting IoT devices",
    "Okiru Attack": "Active attack traffic from Okiru/Satori botnet",
    "Attack": "Generic malicious attack traffic",
    "File Download": "Suspicious file download activity"
};

// ============== STATIC DASHBOARD ==============
document.addEventListener('DOMContentLoaded', () => {
    renderHeroStats();
    renderPipeline();
    renderSummaryStats();
    renderPerformanceTable();
    renderCharts();
    renderVariants();
    renderFeatures();
    setupUploadZone();
});

function renderHeroStats() {
    const binarySelected = SELECTED_BINARY.length;
    const multiSelected = SELECTED_MULTI.length;
    // Count unique variants across SELECTED multiclass models only
    const allVariants = new Set();
    if (typeof MULTICLASS_STATS !== 'undefined') {
        MULTICLASS_STATS.filter(m => SELECTED_MULTI.includes(m.id))
            .forEach(m => m.variants.filter(v => v !== '-').forEach(v => allVariants.add(v)));
    }
    const totalRows = DATASETS.reduce((s, d) => s + (d.trainRows || 0) + (d.valRows || 0) + (d.testRows || 0), 0);

    document.getElementById('h-models').textContent = binarySelected;
    document.getElementById('h-multi').textContent = multiSelected;
    document.getElementById('h-variants').textContent = allVariants.size;
    document.getElementById('h-rows').textContent = (totalRows / 1e6).toFixed(0) + 'M';
}

function renderPipeline() {
    const el = document.getElementById('pipeline');
    el.innerHTML = PIPELINE_STEPS.map(s =>
        `<div class="pipeline-step"><span class="step-icon">${s.icon}</span><span class="step-title">${s.title}</span><span class="step-desc">${s.desc}</span></div>`
    ).join('');
}

function renderSummaryStats() {
    const trained = DATASETS.filter(d => d.status === 'trained');
    // Avg AUC from SELECTED binary models only (meaningful metric)
    const selectedAucs = trained.filter(d => SELECTED_BINARY.includes(d.id) && d.test?.rocAuc)
        .map(d => d.test.rocAuc);
    const avgAuc = selectedAucs.length ? selectedAucs.reduce((a, b) => a + b, 0) / selectedAucs.length : 0;
    const bestAcc = trained.reduce((best, d) => Math.max(best, d.test?.acc || 0), 0);
    const totalRows = DATASETS.reduce((s, d) => s + (d.trainRows || 0) + (d.valRows || 0) + (d.testRows || 0), 0);

    document.getElementById('s-trained').textContent = trained.length + '/23';
    document.getElementById('s-avgauc').textContent = avgAuc.toFixed(4);
    document.getElementById('s-bestacc').textContent = (bestAcc * 100).toFixed(2) + '%';
    document.getElementById('s-rows').textContent = (totalRows / 1e6).toFixed(0) + 'M';
}
// Models selected for detection (ROC-AUC > 0.85 for binary; ALL trained for multiclass to cover every variant)
const SELECTED_BINARY = ['dataset4','dataset5','dataset8','dataset9','dataset10','dataset17','dataset19','dataset20'];
const SELECTED_MULTI = ['dataset4','dataset5','dataset6','dataset7','dataset8','dataset9','dataset10','dataset11','dataset12','dataset13','dataset14','dataset15','dataset16','dataset17','dataset18','dataset19','dataset20','dataset21','dataset22','dataset23'];

function renderPerformanceTable() {
    const thead = document.getElementById('perfTHead');
    const tbody = document.getElementById('perfTBody');
    thead.innerHTML = '<tr><th></th><th>Dataset</th><th>Status</th><th>Train Rows</th><th>Class Distribution</th><th>Binary AUC</th><th>Binary Acc</th><th>Multi Acc</th><th>Multi F1</th><th>Used</th><th>Variants</th></tr>';

    tbody.innerHTML = DATASETS.map((d, idx) => {
        const multiStat = typeof MULTICLASS_STATS !== 'undefined' ? MULTICLASS_STATS.find(m => m.id === d.id) : null;
        const statusBadge = d.status === 'trained' ? '<span class="badge badge-green">Trained</span>' :
            d.status === 'single_class_train' ? '<span class="badge badge-purple">Single Class</span>' :
            '<span class="badge badge-red">Error</span>';
        const binaryAuc = d.test?.rocAuc ? d.test.rocAuc.toFixed(4) : '—';
        const binaryAcc = d.test?.acc ? (d.test.acc * 100).toFixed(2) + '%' : '—';
        const multiAcc = multiStat ? (multiStat.accuracy * 100).toFixed(2) + '%' : '—';
        const multiF1 = multiStat && multiStat.f1_weighted != null ? (multiStat.f1_weighted * 100).toFixed(2) + '%' : '—';
        const variants = multiStat ? multiStat.variants.filter(v => v !== '-').map(v =>
            `<span class="badge badge-purple" style="margin:1px;font-size:10px">${VARIANT_NAMES[v] || v}</span>`
        ).join(' ') : '—';

        // Class distribution
        let classDist = '—';
        if (d.classTrain) {
            const benign = (d.classTrain[0] || 0).toLocaleString();
            const malicious = (d.classTrain[1] || 0).toLocaleString();
            const malPct = d.classTrain[1] ? ((d.classTrain[1] / (d.classTrain[0] + d.classTrain[1])) * 100).toFixed(1) : '0';
            classDist = `<span style="color:var(--green);font-size:11px">✅ ${benign}</span> / <span style="color:var(--red);font-size:11px">❌ ${malicious}</span><br><span style="color:var(--text-dim);font-size:10px">${malPct}% malicious</span>`;
        }

        // Used indicator
        const usedBinary = SELECTED_BINARY.includes(d.id);
        const usedMulti = SELECTED_MULTI.includes(d.id);
        let usedBadge = '';
        if (usedBinary && usedMulti) {
            usedBadge = '<span class="badge badge-green" style="font-size:10px">✅ Both</span>';
        } else if (usedBinary) {
            usedBadge = '<span class="badge badge-green" style="font-size:10px">✅ Binary</span>';
        } else if (usedMulti) {
            usedBadge = '<span class="badge badge-purple" style="font-size:10px">✅ Multi</span>';
        } else {
            usedBadge = '<span class="badge badge-red" style="font-size:10px;opacity:0.6">❌ Excluded</span>';
        }

        // Build detail row content
        let detailContent = '<div class="detail-grid">';

        // Binary metrics detail
        if (d.test) {
            const auc = d.test.rocAuc || 0;
            const aucPct = (auc * 100).toFixed(1);
            const aucColor = auc >= 0.85 ? 'var(--green)' : auc > 0.5 ? 'var(--amber)' : 'var(--red)';
            detailContent += `<div class="detail-card">
                <h4>🎯 Binary Detection</h4>
                <div class="metric-bar"><span>ROC-AUC</span><div class="bar-track"><div class="bar-fill" style="width:${aucPct}%;background:${aucColor}"></div></div><span>${auc ? auc.toFixed(4) : 'N/A'}</span></div>
                <div class="metric-bar"><span>PR-AUC</span><div class="bar-track"><div class="bar-fill" style="width:${((d.test.prAuc||0)*100).toFixed(0)}%;background:var(--purple)"></div></div><span>${d.test.prAuc ? d.test.prAuc.toFixed(4) : 'N/A'}</span></div>
                <div class="metric-bar"><span>Accuracy</span><div class="bar-track"><div class="bar-fill" style="width:${((d.test.acc||0)*100).toFixed(0)}%;background:var(--cyan)"></div></div><span>${d.test.acc ? (d.test.acc*100).toFixed(2)+'%' : 'N/A'}</span></div>`;
            if (d.test.cm) {
                const cm = d.test.cm;
                const precision = cm.tp / (cm.tp + cm.fp || 1);
                const recall = cm.tp / (cm.tp + cm.fn || 1);
                const f1 = 2 * precision * recall / (precision + recall || 1);
                detailContent += `<div style="margin-top:12px"><strong style="color:var(--text)">Confusion Matrix</strong></div>
                <div class="cm-grid">
                    <div class="cm-cell cm-tn">TN<br>${cm.tn.toLocaleString()}</div>
                    <div class="cm-cell cm-fp">FP<br>${cm.fp.toLocaleString()}</div>
                    <div class="cm-cell cm-fn">FN<br>${cm.fn.toLocaleString()}</div>
                    <div class="cm-cell cm-tp">TP<br>${cm.tp.toLocaleString()}</div>
                </div>
                <div style="font-size:12px;color:var(--text-dim);margin-top:8px">
                    Precision: <strong style="color:var(--cyan)">${(precision*100).toFixed(2)}%</strong> · 
                    Recall: <strong style="color:var(--green)">${(recall*100).toFixed(2)}%</strong> · 
                    F1: <strong style="color:var(--purple)">${(f1*100).toFixed(2)}%</strong>
                </div>`;
            }
            detailContent += '</div>';
        }

        // Multiclass metrics detail
        if (multiStat) {
            const mAcc = (multiStat.accuracy * 100).toFixed(2);
            const mAccColor = multiStat.accuracy >= 0.9 ? 'var(--green)' : multiStat.accuracy >= 0.5 ? 'var(--amber)' : 'var(--red)';
            detailContent += `<div class="detail-card">
                <h4>🧬 Multiclass Variant Detection</h4>
                <div class="metric-bar"><span>Accuracy</span><div class="bar-track"><div class="bar-fill" style="width:${mAcc}%;background:${mAccColor}"></div></div><span>${mAcc}%</span></div>`;
            if (multiStat.f1_weighted != null) {
                detailContent += `<div class="metric-bar"><span>F1-Weighted</span><div class="bar-track"><div class="bar-fill" style="width:${(multiStat.f1_weighted*100).toFixed(1)}%;background:var(--purple)"></div></div><span>${(multiStat.f1_weighted*100).toFixed(2)}%</span></div>`;
            }
            if (multiStat.f1_macro != null) {
                detailContent += `<div class="metric-bar"><span>F1-Macro</span><div class="bar-track"><div class="bar-fill" style="width:${(multiStat.f1_macro*100).toFixed(1)}%;background:var(--cyan)"></div></div><span>${(multiStat.f1_macro*100).toFixed(2)}%</span></div>`;
            }
            detailContent += `<div style="margin-top:12px;font-size:12px;color:var(--text-dim)">
                Train Classes: <strong style="color:var(--purple)">${multiStat.classes}</strong> · 
                Test Classes: <strong style="color:var(--cyan)">${multiStat.testClasses || '—'}</strong>
            </div>
            <div style="margin-top:8px;display:flex;flex-wrap:wrap;gap:4px">${multiStat.variants.filter(v => v !== '-').map(v =>
                `<span class="badge badge-purple" style="font-size:10px">${VARIANT_ICONS[VARIANT_NAMES[v]||v]||'🔴'} ${VARIANT_NAMES[v]||v}</span>`).join('')}
            </div></div>`;
        }

        // Dataset info
        const totalRows = (d.trainRows||0) + (d.valRows||0) + (d.testRows||0);
        detailContent += `<div class="detail-card">
            <h4>📋 Dataset Info</h4>
            <div style="font-size:13px;color:var(--text-dim);line-height:2">
                <div>Total Rows: <strong style="color:var(--text)">${totalRows.toLocaleString()}</strong></div>
                <div>Train: <strong>${(d.trainRows||0).toLocaleString()}</strong> · Val: <strong>${(d.valRows||0).toLocaleString()}</strong> · Test: <strong>${(d.testRows||0).toLocaleString()}</strong></div>
                <div>Features: <strong style="color:var(--cyan)">${d.features || '—'}</strong></div>
                ${d.scalePos ? `<div>Scale Pos Weight: <strong>${d.scalePos}</strong></div>` : ''}
                ${d.threshold ? `<div>Threshold: <strong>${d.threshold}</strong></div>` : ''}
            </div></div>`;

        detailContent += '</div>';

        const mainRow = `<tr class="perf-row" onclick="toggleDetail('detail-${d.id}')" style="cursor:pointer;${!usedBinary && !usedMulti ? 'opacity:0.5' : ''}">
            <td style="font-size:16px;text-align:center">▸</td>
            <td style="font-weight:600;color:var(--cyan)">${d.id}</td>
            <td>${statusBadge}</td>
            <td>${d.trainRows ? d.trainRows.toLocaleString() : '—'}</td>
            <td>${classDist}</td>
            <td style="font-family:'JetBrains Mono',monospace;${d.test?.rocAuc >= 0.85 ? 'color:var(--green)' : d.test?.rocAuc ? 'color:var(--red)' : ''}">${binaryAuc}</td>
            <td style="font-family:'JetBrains Mono',monospace">${binaryAcc}</td>
            <td style="font-family:'JetBrains Mono',monospace;${multiStat?.accuracy >= 0.9 ? 'color:var(--green)' : multiStat?.accuracy ? 'color:var(--amber)' : ''}">${multiAcc}</td>
            <td style="font-family:'JetBrains Mono',monospace;color:var(--purple)">${multiF1}</td>
            <td>${usedBadge}</td>
            <td>${variants}</td>
        </tr>`;

        const detailRow = `<tr id="detail-${d.id}" class="detail-row" style="display:none">
            <td colspan="11" style="padding:0">${detailContent}</td>
        </tr>`;

        return mainRow + detailRow;
    }).join('');
}

function toggleDetail(id) {
    const row = document.getElementById(id);
    const arrow = row.previousElementSibling.querySelector('td');
    if (row.style.display === 'none') {
        row.style.display = 'table-row';
        arrow.textContent = '▾';
    } else {
        row.style.display = 'none';
        arrow.textContent = '▸';
    }
}

function renderCharts() {
    // Binary AUC chart
    const trainedDS = DATASETS.filter(d => d.test && d.test.rocAuc);
    new Chart(document.getElementById('aucChart'), {
        type: 'bar',
        data: {
            labels: trainedDS.map(d => d.id.replace('dataset', 'D')),
            datasets: [{
                label: 'ROC-AUC', data: trainedDS.map(d => d.test.rocAuc),
                backgroundColor: trainedDS.map(d => SELECTED_BINARY.includes(d.id) ?
                    (d.test.rocAuc > 0.99 ? 'rgba(34,197,94,0.7)' : 'rgba(6,182,212,0.7)') :
                    'rgba(239,68,68,0.3)'),
                borderRadius: 6
            }]
        },
        options: {
            responsive: true, plugins: { legend: { display: false } },
            scales: {
                y: { min: 0, max: 1, grid: { color: 'rgba(42,58,92,0.4)' }, ticks: { color: '#94a3b8' } },
                x: { grid: { display: false }, ticks: { color: '#94a3b8', font: { size: 10 } } }
            }
        }
    });

    // Multiclass Accuracy chart
    if (typeof MULTICLASS_STATS !== 'undefined') {
        const ms = [...MULTICLASS_STATS].sort((a, b) => b.accuracy - a.accuracy);
        new Chart(document.getElementById('multiAccChart'), {
            type: 'bar',
            data: {
                labels: ms.map(m => m.id.replace('dataset', 'D')),
                datasets: [{
                    label: 'Accuracy', data: ms.map(m => m.accuracy),
                    backgroundColor: ms.map(m => SELECTED_MULTI.includes(m.id) ?
                        (m.accuracy > 0.99 ? 'rgba(168,85,247,0.7)' : 'rgba(99,102,241,0.7)') :
                        'rgba(245,158,11,0.3)'),
                    borderRadius: 6
                }]
            },
            options: {
                responsive: true, plugins: { legend: { display: false } },
                scales: {
                    y: { min: 0, max: 1, grid: { color: 'rgba(42,58,92,0.4)' }, ticks: { color: '#94a3b8' } },
                    x: { grid: { display: false }, ticks: { color: '#94a3b8', font: { size: 10 } } }
                }
            }
        });
    }
}

function renderVariants() {
    const grid = document.getElementById('variantsGrid');
    if (typeof VARIANT_NAMES === 'undefined') return;

    const uniqueVariants = new Set();
    if (typeof MULTICLASS_STATS !== 'undefined') {
        MULTICLASS_STATS.filter(m => SELECTED_MULTI.includes(m.id))
            .forEach(m => m.variants.filter(v => v !== '-').forEach(v => uniqueVariants.add(v)));
    }

    const variantHTML = [...uniqueVariants].map(v => {
        const displayName = VARIANT_NAMES[v] || v;
        const icon = VARIANT_ICONS[displayName] || '🔴';
        const desc = VARIANT_DESCRIPTIONS[displayName] || 'Malicious network activity pattern';
        const modelCount = MULTICLASS_STATS.filter(m => SELECTED_MULTI.includes(m.id) && m.variants.includes(v)).length;
        return `<div class="variant-card">
            <div class="variant-icon">${icon}</div>
            <div class="variant-name">${displayName}</div>
            <div class="variant-desc">${desc}</div>
            <span class="variant-badge">Detected by ${modelCount} model${modelCount > 1 ? 's' : ''}</span>
        </div>`;
    }).join('');

    grid.innerHTML = variantHTML;
}

function renderFeatures() {
    document.getElementById('features').innerHTML = FEATURES.map(f =>
        `<div class="feature-chip">${f}</div>`
    ).join('');
}

// ============== LIVE DETECTION ==============
let _selectedFile = null, _detectResults = null, _detectCharts = {};

function setupUploadZone() {
    const zone = document.getElementById('uploadZone');
    const fileInput = document.getElementById('detectFile');
    zone.addEventListener('click', () => fileInput.click());
    zone.addEventListener('dragover', e => { e.preventDefault(); zone.classList.add('drag-over'); });
    zone.addEventListener('dragleave', () => zone.classList.remove('drag-over'));
    zone.addEventListener('drop', e => {
        e.preventDefault(); zone.classList.remove('drag-over');
        if (e.dataTransfer.files.length) handleFile(e.dataTransfer.files[0]);
    });
    fileInput.addEventListener('change', e => { if (e.target.files.length) handleFile(e.target.files[0]); });
}

function handleFile(file) {
    _selectedFile = file;
    document.getElementById('uploadZone').style.display = 'none';
    document.getElementById('detectControls').style.display = 'block';
    document.getElementById('detectFileName').textContent = `📄 ${file.name} (${(file.size / 1024).toFixed(1)} KB)`;
    document.getElementById('detectResults').style.display = 'none';
}

function clearDetection() {
    _selectedFile = null; _detectResults = null;
    document.getElementById('uploadZone').style.display = 'block';
    document.getElementById('detectControls').style.display = 'none';
    document.getElementById('detectResults').style.display = 'none';
    document.getElementById('detectProgress').style.display = 'none';
    document.getElementById('detectFile').value = '';
}

async function runDetection() {
    if (!_selectedFile) return;
    const btn = document.getElementById('btnDetect');
    const prog = document.getElementById('detectProgress');
    const fill = document.getElementById('progressFill');
    const txt = document.getElementById('progressText');
    btn.disabled = true; btn.textContent = '⏳ Detecting...';
    prog.style.display = 'block'; fill.style.width = '0%'; fill.style.background = 'var(--gradient-1)';

    const form = new FormData();
    form.append('file', _selectedFile);
    form.append('threshold', document.getElementById('detectThreshold').value);
    form.append('combine', document.getElementById('detectCombine').value);

    try {
        fill.style.width = '25%'; txt.textContent = '🔍 Stage 1: Running binary ensemble detection...';
        const resp = await fetch('/api/predict', { method: 'POST', body: form });
        if (!resp.ok) throw new Error('Server error: ' + resp.status);
        fill.style.width = '55%'; txt.textContent = '🧬 Stage 2: Identifying attack variants...';
        const data = await resp.json();
        if (data.error) throw new Error(data.error);
        fill.style.width = '100%'; txt.textContent = '✅ Two-stage detection complete!';
        _detectResults = data;
        setTimeout(() => { prog.style.display = 'none'; renderDetectResults(data); }, 600);
    } catch (err) {
        fill.style.width = '100%'; fill.style.background = 'var(--red)';
        txt.textContent = '❌ Error: ' + err.message;
    }
    btn.disabled = false; btn.textContent = '🔍 Run Detection';
}

function renderDetectResults(data) {
    document.getElementById('detectResults').style.display = 'block';
    document.getElementById('det-total').textContent = data.total.toLocaleString();
    document.getElementById('det-benign').textContent = data.benign.toLocaleString();
    document.getElementById('det-malicious').textContent = data.malicious.toLocaleString();
    const tl = data.threat_pct;
    const det_threat = document.getElementById('det-threat');
    det_threat.textContent = tl.toFixed(1) + '%';
    det_threat.className = 'value ' + (tl > 50 ? 'detect-malicious' : tl > 20 ? '' : 'detect-benign');
    document.getElementById('tab-ben-count').textContent = data.benign.toLocaleString();
    document.getElementById('tab-mal-count').textContent = data.malicious.toLocaleString();

    // Destroy old charts
    Object.values(_detectCharts).forEach(c => c.destroy());
    _detectCharts = {};

    // Pie chart
    _detectCharts.pie = new Chart(document.getElementById('detectPieChart'), {
        type: 'doughnut',
        data: {
            labels: ['Benign (Allowed)', 'Malicious (Dropped)'],
            datasets: [{
                data: [data.benign, data.malicious], backgroundColor: ['rgba(34,197,94,0.7)', 'rgba(239,68,68,0.7)'],
                borderColor: ['#22c55e', '#ef4444'], borderWidth: 2, hoverOffset: 8
            }]
        },
        options: { responsive: true, plugins: { legend: { labels: { color: '#94a3b8' } } } }
    });

    // Histogram
    const bins = 20;
    const benHist = histBins(data.prob_distribution.benign, bins);
    const malHist = histBins(data.prob_distribution.malicious, bins);
    _detectCharts.hist = new Chart(document.getElementById('detectHistChart'), {
        type: 'bar',
        data: {
            labels: benHist.labels,
            datasets: [
                { label: 'Benign', data: benHist.counts, backgroundColor: 'rgba(34,197,94,0.5)', borderColor: '#22c55e', borderWidth: 1, borderRadius: 3 },
                { label: 'Malicious', data: malHist.counts, backgroundColor: 'rgba(239,68,68,0.5)', borderColor: '#ef4444', borderWidth: 1, borderRadius: 3 }
            ]
        },
        options: {
            responsive: true, plugins: { legend: { labels: { color: '#94a3b8' } } },
            scales: {
                x: { grid: { display: false }, ticks: { color: '#94a3b8', font: { size: 9 } } },
                y: { grid: { color: 'rgba(42,58,92,0.5)' }, ticks: { color: '#94a3b8' } }
            }
        }
    });

    // Variant chart (Stage 2 results)
    const vc = data.variant_counts || {};
    const variantLabels = Object.keys(vc).filter(k => k !== 'N/A' && k !== 'Unknown Malware');
    const variantData = variantLabels.map(k => vc[k]);
    const variantColors = [
        'rgba(168,85,247,0.7)', 'rgba(239,68,68,0.7)', 'rgba(245,158,11,0.7)',
        'rgba(6,182,212,0.7)', 'rgba(34,197,94,0.7)', 'rgba(236,72,153,0.7)',
        'rgba(99,102,241,0.7)', 'rgba(249,115,22,0.7)', 'rgba(20,184,166,0.7)'
    ];
    if (variantLabels.length > 0) {
        _detectCharts.variant = new Chart(document.getElementById('detectVariantChart'), {
            type: 'doughnut',
            data: {
                labels: variantLabels,
                datasets: [{
                    data: variantData,
                    backgroundColor: variantColors.slice(0, variantLabels.length),
                    borderColor: variantColors.slice(0, variantLabels.length).map(c => c.replace('0.7', '1')),
                    borderWidth: 2, hoverOffset: 8
                }]
            },
            options: { responsive: true, plugins: { legend: { labels: { color: '#94a3b8', font: { size: 11 } } } } }
        });
    } else {
        const ctx = document.getElementById('detectVariantChart').getContext('2d');
        ctx.fillStyle = '#94a3b8'; ctx.font = '14px Inter, sans-serif'; ctx.textAlign = 'center';
        ctx.fillText('No variants detected (all benign)', ctx.canvas.width / 2, ctx.canvas.height / 2);
    }

    // Model chart
    const ms = data.model_stats.filter(m => m.flagged > 0).sort((a, b) => b.flagged - a.flagged);
    _detectCharts.model = new Chart(document.getElementById('detectModelChart'), {
        type: 'bar',
        data: {
            labels: ms.map(m => m.name.replace('dataset', 'D')),
            datasets: [{
                label: 'Packets Flagged', data: ms.map(m => m.flagged),
                backgroundColor: ms.map((_, i) => `rgba(6,182,212,${0.4 + i * 0.04})`),
                borderColor: '#06b6d4', borderWidth: 1, borderRadius: 6
            }]
        },
        options: {
            indexAxis: 'y', responsive: true, plugins: { legend: { display: false } },
            scales: {
                x: { grid: { color: 'rgba(42,58,92,0.5)' }, ticks: { color: '#94a3b8' } },
                y: { grid: { display: false }, ticks: { color: '#94a3b8', font: { size: 10 } } }
            }
        }
    });

    switchTab('dropped');  // Show dropped first so user sees variants
    document.getElementById('detectResults').scrollIntoView({ behavior: 'smooth', block: 'start' });
}

function histBins(arr, n) {
    const labels = [], counts = new Array(n).fill(0);
    for (let i = 0; i < n; i++) labels.push((i / n).toFixed(2));
    arr.forEach(v => { const idx = Math.min(Math.floor(v * n), n - 1); counts[idx]++; });
    return { labels, counts };
}

function switchTab(tab) {
    document.querySelectorAll('.tab-btn').forEach((b, i) => b.classList.toggle('active', i === (tab === 'allowed' ? 0 : 1)));
    if (!_detectResults) return;
    const rows = _detectResults.results.filter(r => tab === 'allowed' ? r.verdict === 'BENIGN' : r.verdict === 'MALICIOUS');
    const thead = document.getElementById('detectTHead');
    const tbody = document.getElementById('detectTBody');
    if (tab === 'dropped') {
        thead.innerHTML = '<tr><th>#</th><th>Verdict</th><th>Probability</th><th>🧬 Attack Variant</th><th>Variant Conf.</th><th>Top Model</th></tr>';
        tbody.innerHTML = rows.slice(0, 200).map(r => {
            const vIcon = VARIANT_ICONS[r.variant] || '🔴';
            return `<tr><td>${r.index}</td><td><span class="badge badge-red">❌ DROPPED</span></td>` +
                `<td style="font-family:'JetBrains Mono',monospace">${(r.probability * 100).toFixed(2)}%</td>` +
                `<td><span class="badge badge-purple">${vIcon} ${r.variant || 'Unknown'}</span></td>` +
                `<td style="font-family:'JetBrains Mono',monospace">${r.variant_confidence ? (r.variant_confidence * 100).toFixed(1) + '%' : '—'}</td>` +
                `<td style="color:var(--cyan)">${r.top_model.replace('dataset', 'D')}</td></tr>`;
        }).join('');
    } else {
        thead.innerHTML = '<tr><th>#</th><th>Verdict</th><th>Probability</th><th>Top Model</th><th>Confidence</th></tr>';
        tbody.innerHTML = rows.slice(0, 200).map(r =>
            `<tr><td>${r.index}</td><td><span class="badge badge-green">✅ ALLOWED</span></td>` +
            `<td style="font-family:'JetBrains Mono',monospace">${(r.probability * 100).toFixed(2)}%</td>` +
            `<td style="color:var(--cyan)">${r.top_model.replace('dataset', 'D')}</td>` +
            `<td style="font-family:'JetBrains Mono',monospace">${(r.top_model_conf * 100).toFixed(2)}%</td></tr>`
        ).join('');
    }
}

function downloadResults(type) {
    if (!_detectResults) return;
    const rows = _detectResults.results.filter(r => type === 'allowed' ? r.verdict === 'BENIGN' : r.verdict === 'MALICIOUS');
    const header = type === 'dropped' ?
        'index,verdict,probability,variant,variant_confidence,top_model\n' :
        'index,verdict,probability,top_model,confidence\n';
    const csv = header + rows.map(r => type === 'dropped' ?
        `${r.index},${r.verdict},${r.probability},${r.variant || 'N/A'},${r.variant_confidence || 0},${r.top_model}` :
        `${r.index},${r.verdict},${r.probability},${r.top_model},${r.top_model_conf}`
    ).join('\n');
    const blob = new Blob([csv], { type: 'text/csv' });
    const a = document.createElement('a');
    a.href = URL.createObjectURL(blob);
    a.download = `${type}_packets.csv`;
    a.click();
}
