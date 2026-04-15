// Global variables
window.ortSession = null;
window.currentInferenceInput = null;
window.resultsStale = false;

// Global ORT Environment setup for GitHub Pages
ort.env.wasm.wasmPaths = 'https://cdn.jsdelivr.net/npm/onnxruntime-web/dist/';

// The exact bounds from Python make_pretty logic
// AV: (0, 24)
// TSR: (>70)   (Using index 1 from perfvar_list) Wait, app.py uses 0..100.
// rut5k: (0, 0.5)
// rut20k: (0, 0.5)
// Cantabro: (0, 20)
const boundsOk = {
    lower: [0, 70, 0, 0, 0], // Note TSR lower bound is 70 according to data.py bounds check "x>=70"
    upper: [24, Infinity, 0.5, 0.5, 20]
};

const perfVarList = [
    "Air Void (%)",
    "TSR (%)",
    "Rut Depth at 5,000 passes (in.)",
    "Rut Depth at 20,000 passes (in.)",
    "Cantabro Loss - Unaged (%)"
];

// Gradation Configuration
const gradations = [
    { id: '19', label: '19mm', default: 100 },
    { id: '12p5', label: '12.5mm', default: 88 },
    { id: '9p5', label: '9.5mm', default: 69 },
    { id: '4p75', label: '4.75mm', default: 16 },
    { id: '2p36', label: '2.36mm', default: 6 },
    { id: '1p18', label: '1.18mm', default: 5 },
    { id: '0p60', label: '0.60mm', default: 5 },
    { id: '0p30', label: '0.30mm', default: 4 },
    { id: '0p15', label: '0.15mm', default: 4 },
    { id: '0p075', label: '0.075mm', default: 3.1 }
];

function initUI() {
    const container = document.getElementById('grad-sliders-container');
    gradations.forEach((g, idx) => {
        // Enforce physical constraints: smaller sieve cannot have higher passing %
        // e.g. 19mm must be >= 12.5mm
        const html = `
            <div class="space-y-1">
                <div class="flex justify-between">
                    <label class="text-xs font-semibold text-slate-600">${g.label}</label>
                    <span id="val-${g.id}" class="text-xs font-mono text-slate-500">${g.default.toFixed(1)}%</span>
                </div>
                <input type="range" id="inp-${g.id}" data-idx="${idx}" min="0" max="100" step="0.1" value="${g.default}" class="w-full h-1 bg-slate-200 rounded-lg appearance-none cursor-pointer accent-indigo-500">
            </div>
        `;
        container.insertAdjacentHTML('beforeend', html);
    });

    // Add event listeners with cascade logic
    gradations.forEach((g, idx) => {
        const el = document.getElementById(`inp-${g.id}`);
        el.addEventListener('input', (e) => {
            const val = parseFloat(e.target.value);
            document.getElementById(`val-${g.id}`).textContent = val.toFixed(1) + '%';
            
            // Cascade down (smaller sieves can't be bigger)
            for(let i = idx + 1; i < gradations.length; i++) {
                const child = document.getElementById(`inp-${gradations[i].id}`);
                let childVal = parseFloat(child.value);
                if (childVal > val) {
                    child.value = val;
                    document.getElementById(`val-${gradations[i].id}`).textContent = val.toFixed(1) + '%';
                }
            }
            
            // Cascade up (larger sieves can't be smaller)
            for(let i = idx - 1; i >= 0; i--) {
                const parent = document.getElementById(`inp-${gradations[i].id}`);
                let parentVal = parseFloat(parent.value);
                if (parentVal < val) {
                    parent.value = val;
                    document.getElementById(`val-${gradations[i].id}`).textContent = val.toFixed(1) + '%';
                }
            }
        });
    });

    document.getElementById('btn-predict').addEventListener('click', runInference);

    // Attach "Stale" listeners to all inputs to clear plots if values change
    const allInputs = document.querySelectorAll('#prediction-controls input, #prediction-controls select');
    allInputs.forEach(input => {
        input.addEventListener('change', markResultsStale);
        if (input.type === 'range') {
            input.addEventListener('input', markResultsStale);
        }
    });
}

function markResultsStale() {
    if (!resultsStale) {
        resultsStale = true;
        const msg = `
            <div class="flex flex-col items-center justify-center py-12 text-slate-400 animate-pulse">
                <svg xmlns="http://www.w3.org/2000/svg" class="h-12 w-12 mb-3 opacity-20" fill="none" viewBox="0 0 24 24" stroke="currentColor"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M4 4v5h.582m15.356 2A8.001 8.001 0 004.582 9m0 0H9m11 11v-5h-.581m0 0a8.003 8.003 0 01-15.357-2m15.357 2H15" /></svg>
                <p class="text-sm font-medium">Inputs changed</p>
                <p class="text-xs">Click "Run Inference" to refresh results</p>
            </div>
        `;
        document.getElementById('prediction-results-plots').innerHTML = msg;
        document.getElementById('prediction-results-table').innerHTML = `<tr><td colspan="4" class="py-10 text-center text-slate-400 italic">Results out of date. Re-run inference.</td></tr>`;
    }
}

function resetGradation() {
    gradations.forEach((g) => {
        const el = document.getElementById(`inp-${g.id}`);
        el.value = g.default;
        document.getElementById(`val-${g.id}`).textContent = g.default.toFixed(1) + '%';
    });
}

function switchTab(tabId) {
    const tabs = ['prediction', 'xai', 'optimizer'];
    tabs.forEach(t => {
        const section = document.getElementById(`section-${t}`);
        const btn = document.getElementById(`tab-${t}`);
        if (section) section.classList.add('hidden');
        if (btn) {
            btn.classList.remove('border-indigo-600', 'text-indigo-600', 'border-emerald-600', 'text-emerald-600');
            btn.classList.add('border-transparent', 'text-slate-500');
        }
    });

    const activeSection = document.getElementById(`section-${tabId}`);
    const activeBtn = document.getElementById(`tab-${tabId}`);
    
    if (activeSection) activeSection.classList.remove('hidden');
    if (activeBtn) {
        const color = tabId === 'optimizer' ? 'emerald' : 'indigo';
        activeBtn.classList.add(`border-${color}-600`, `text-${color}-600`);
        activeBtn.classList.remove('border-transparent', 'text-slate-500');
    }

    if (tabId === 'xai' && typeof renderShapleyPlots === 'function') {
        const tensorParams = prepareTensorInputs();
        renderShapleyPlots(tensorParams.flatValues);
    }
    
    // Fix Plotly resize issues when switching back to already-rendered plots
    if (tabId === 'prediction' && !resultsStale) {
        window.dispatchEvent(new Event('resize'));
    }
}

function toggleResultsView(viewId) {
    if (viewId === 'visual') {
        document.getElementById('results-visual').classList.remove('hidden');
        document.getElementById('results-visual').classList.add('flex');
        document.getElementById('results-data').classList.add('hidden');
        
        document.getElementById('btn-view-visual').className = "px-3 py-1 text-sm font-semibold rounded-md bg-white text-indigo-700 shadow-sm transition";
        document.getElementById('btn-view-data').className = "px-3 py-1 text-sm font-medium rounded-md text-slate-500 hover:text-slate-700 transition";
        
        // Re-trigger layout calculation in case plots were drawn while hidden
        setTimeout(() => {
            window.dispatchEvent(new Event('resize'));
        }, 10);
    } else {
        document.getElementById('results-visual').classList.add('hidden');
        document.getElementById('results-visual').classList.remove('flex');
        document.getElementById('results-data').classList.remove('hidden');
        
        document.getElementById('btn-view-data').className = "px-3 py-1 text-sm font-semibold rounded-md bg-white text-indigo-700 shadow-sm transition";
        document.getElementById('btn-view-visual').className = "px-3 py-1 text-sm font-medium rounded-md text-slate-500 hover:text-slate-700 transition";
    }
}

// AES decryption using CryptoJS
async function decryptModel(buffer) {
    const keyStr = "OGFC_SecretKey_1234567890123456X"; // Obfuscated Decoy
    const ivStr = "1234567890123456";
    const key = CryptoJS.enc.Utf8.parse(keyStr);
    const iv = CryptoJS.enc.Utf8.parse(ivStr);

    const wordBuf = CryptoJS.lib.WordArray.create(buffer);
    const decrypted = CryptoJS.AES.decrypt({ ciphertext: wordBuf }, key, {
        iv: iv,
        mode: CryptoJS.mode.CBC,
        padding: CryptoJS.pad.Pkcs7
    });
    
    // convert back to arraybuffer
    const typedArray = new Uint8Array(decrypted.sigBytes);
    for (let i = 0; i < decrypted.sigBytes; i++) {
        typedArray[i] = (decrypted.words[i >>> 2] >>> (24 - (i % 4) * 8)) & 0xff;
    }
    return typedArray.buffer;
}

// Prepare model tensor (equivalent to Python build_tensor logic)
function prepareTensorInputs() {
    const addVal = parseInt(document.getElementById('inp-additive').value);
    const fillerVal = parseInt(document.getElementById('inp-filler').value);
    const aggVal = parseInt(document.getElementById('inp-aggregate').value); // radio: 0-4
    const nmasVal = parseInt(document.getElementById('inp-nmas').value);
    const acVal = parseFloat(document.getElementById('inp-ac').value) / 100.0;
    
    // Special handling for aggregate vs cr to match model.py routing
    let agg_mod = aggVal;
    let cr_mod = 0;
    if (aggVal === 4) {
        agg_mod = 0;
        cr_mod = 1;
    }

    const sList = gradations.map(g => parseFloat(document.getElementById(`inp-${g.id}`).value) / 100.0);

    // Dictionaries from data.py
    const addDict = {
        0: [0,0,0,1],
        1: [1,0,0,0],
        2: [0,0,1,0],
        3: [0,1,0,0]
    };
    const fillerDict = {
        0: [0,0,0,0,0,1],
        1: [1,0,0,0,0,0],
        2: [0,1,0,0,0,0],
        4: [0,0,1,0,0,0],
        5: [0,0,0,1,0,0],
        3: [0,0,0,0,1,0]
    };
    const crDict = {
        0: [1,0],
        1: [0,1]
    };
    const nmasDict = {
        0: [0,1],
        1: [1,0]
    };
    const aggDict = {
        0: [0,0,0,1],
        1: [1,0,0,0],
        2: [0,1,0,0],
        3: [0,0,1,0]
    };

    const flatValues = [
        ...addDict[addVal],
        ...fillerDict[fillerVal],
        ...crDict[cr_mod],
        ...nmasDict[nmasVal],
        ...aggDict[agg_mod],
        acVal,
        ...sList
    ];
    // Length should be 4 + 6 + 2 + 2 + 4 + 1 + 10 = 29
    return { flatValues };
}

async function runInference() {
    const loadingOverlay = document.getElementById('loading-overlay');
    loadingOverlay.classList.remove('hidden');
    loadingOverlay.classList.add('flex');
    
    try {
        // Build tensor
        const { flatValues } = prepareTensorInputs();
        currentInferenceInput = flatValues;

        if (!ortSession) {
            console.log("Fetching and decrypting model...");
            const response = await fetch('Materials/encrypted_model.bin');
            const arrayBuffer = await response.arrayBuffer();
            const decryptedBuffer = await decryptModel(arrayBuffer);
            ortSession = await ort.InferenceSession.create(decryptedBuffer);
            console.log("Model loaded into ORT.");
        }

        const tensor = new ort.Tensor('float32', Float32Array.from(flatValues), [1, 29]);
        const results = await ortSession.run({ input: tensor });
        const output = results.output.data; 

        // Output shape [1, 5, 3] flattened to [15]. Structure: Var1(LB, Median, UB), Var2(LB, Median, UB), etc.
        const multiplier = [100, 100, 1, 1, 100];
        document.getElementById('prediction-results-plots').innerHTML = ''; // clear plots
        const rowsHtml = []; // clear table
        resultsStale = false;

        window.currentPredictions = {
            "1": output[0*3 + 1] * multiplier[0],
            "4": output[1*3 + 1] * multiplier[1],
            "7": output[2*3 + 1] * multiplier[2],
            "10": output[3*3 + 1] * multiplier[3],
            "13": output[4*3 + 1] * multiplier[4]
        };

        for (let i = 0; i < 5; i++) {
            // Retrieve raw array values
            const val1 = output[i * 3 + 0] * multiplier[i];
            const median = output[i * 3 + 1] * multiplier[i];
            const val2 = output[i * 3 + 2] * multiplier[i];

            // Handle potential quantile crossing (unconstrained NN outputs)
            let lb = Math.min(val1, val2);
            let ub = Math.max(val1, val2);

            // Ensure median is bounded logically in the UI if crossing occurred
            lb = Math.min(lb, median);
            ub = Math.max(ub, median);

            // create div for this bullet plot with header on top
            const divId = `bullet-chart-${i}`;
            const divHtml = `
                <div class="mt-4 first:mt-0">
                    <div class="text-[11px] font-bold text-slate-500 uppercase tracking-wide px-1 mb-1">${perfVarList[i]}</div>
                    <div id="${divId}" class="w-full h-[55px] border-b border-slate-50 pb-1"></div>
                </div>
            `;
            document.getElementById('prediction-results-plots').insertAdjacentHTML('beforeend', divHtml);

            let axisMax = 100;
            let greenRange = [0, 100];
            let redRangeLower = null;
            let redRangeUpper = null;
            if(i === 0) { // Air Void: 0 <= x <= 24
                 axisMax = Math.max(30, ub + 5); 
                 greenRange = [0, 24];
                 redRangeUpper = [24, axisMax];
            } else if(i===1) { // TSR: x >= 70
                 axisMax = Math.max(100, ub + 5); 
                 greenRange = [70, 100];
                 redRangeLower = [0, 70];
            } else if(i===2) { // Rut 5k: 0 <= x <= 0.5
                 axisMax = Math.max(1.0, ub + 0.2); 
                 greenRange = [0, 0.5];
                 redRangeUpper = [0.5, axisMax];
            } else if(i===3) { // Rut 20k: 0 <= x <= 0.5
                 axisMax = Math.max(1.0, ub + 0.2); 
                 greenRange = [0, 0.5];
                 redRangeUpper = [0.5, axisMax];
            } else if(i===4) { // Cantabro: 0 <= x <= 20
                 axisMax = Math.max(40, ub + 5); 
                 greenRange = [0, 20];
                 redRangeUpper = [20, axisMax];
            }

            const inSpec = (median >= greenRange[0] && median <= greenRange[1]);

            let plotSteps = [ { range: greenRange, color: "#dcfce7" } ]; // green-100
            if (redRangeLower) plotSteps.push({ range: redRangeLower, color: "#ffe4e6" }); // rose-100
            if (redRangeUpper) plotSteps.push({ range: redRangeUpper, color: "#ffe4e6" });
            
            // CI layer as translucent gray
            plotSteps.push({ range: [lb, ub], color: "rgba(148, 163, 184, 0.35)" });

            const plotData = [
              {
                type: "indicator",
                mode: "number+gauge",
                value: median,
                number: { valueformat: ".2f", font: { size: 18, color: inSpec ? "#4338ca" : "#e11d48" } },
                domain: { x: [0, 1], y: [0, 1] },
                gauge: {
                  shape: "bullet",
                  axis: { range: [0, axisMax], tickcolor: "#94a3b8" },
                  steps: plotSteps,
                  bar: { color: inSpec ? "#4f46e5" : "#f43f5e", thickness: 0.4 } 
                }
              }
            ];

            const layout = { margin: { t: 5, b: 20, l: 30, r: 20 }, paper_bgcolor: "rgba(0,0,0,0)", font: {family: "ui-sans-serif, system-ui, sans-serif"} };
            
            // Wrap in setTimeout to ensure DOM container width is painted before Plotly extracts sizes
            setTimeout(() => {
                Plotly.newPlot(divId, plotData, layout, {displayModeBar: false, responsive: true});
            }, 50);

            // Populate Data Table view
            const buildCellHtml = (val) => {
                const ok = (val >= greenRange[0] && val <= greenRange[1]);
                const c = ok ? 'text-emerald-700 bg-emerald-50/50' : 'text-rose-700 bg-rose-50/50';
                return `<td class="py-4 px-4 text-right font-mono ${c}">${val.toFixed(2)}</td>`;
            };

            rowsHtml.push(`
                <tr class="border-b border-slate-50 transition-colors hover:bg-slate-100">
                    <td class="py-4 px-4 font-bold text-slate-600">${perfVarList[i]}</td>
                    ${buildCellHtml(lb)}
                    ${buildCellHtml(median)}
                    ${buildCellHtml(ub)}
                </tr>
            `);
        }

        document.getElementById('prediction-results-table').innerHTML = rowsHtml.join('');

    } catch (e) {
        console.error(e);
        alert("An error occurred during inference. See console.");
    } finally {
        loadingOverlay.classList.remove('flex');
        loadingOverlay.classList.add('hidden');
    }
}

// Initial setup
window.addEventListener('DOMContentLoaded', initUI);

async function runSystemValidation() {
    try {
        const response = await fetch('Materials/model_validation_tests.json');
        const tests = await response.json();

        if (!ortSession) {
            console.log("Fetching and decrypting model for validation...");
            const modelResp = await fetch('Materials/encrypted_model.bin');
            const arrayBuffer = await modelResp.arrayBuffer();
            const decryptedBuffer = await decryptModel(arrayBuffer);
            ortSession = await ort.InferenceSession.create(decryptedBuffer);
        }

        const multiplier = [100, 100, 1, 1, 100];
        const names = ["Air Void", "TSR", "Rut 5k", "Rut 20k", "Cantabro"];
        const resultsTable = [];

        for (let t = 0; t < tests.length; t++) {
            const input_vector = tests[t].input_vector;
            const expected_output = tests[t].expected_output;
            
            const tensor = new ort.Tensor('float32', Float32Array.from(input_vector), [1, 29]);
            const results = await ortSession.run({ input: tensor });
            const output = results.output.data;

            for (let i = 0; i < 5; i++) {
                // Expected from Python: [LB, Median, UB]
                const ex_lb = expected_output[i][0] * multiplier[i];
                const ex_med = expected_output[i][1] * multiplier[i];
                const ex_ub = expected_output[i][2] * multiplier[i];

                // Actual from JS
                const ac_lb = output[i*3 + 0] * multiplier[i];
                const ac_med = output[i*3 + 1] * multiplier[i];
                const ac_ub = output[i*3 + 2] * multiplier[i];

                // Compute deltas
                const d_lb = Math.abs(ex_lb - ac_lb);
                const d_med = Math.abs(ex_med - ac_med);
                const d_ub = Math.abs(ex_ub - ac_ub);

                // Check formatting
                if (d_lb > 1e-4) resultsTable.push({"Case": t, "Var": names[i], "Type": "LB", "Exp": ex_lb, "Act": ac_lb, "Delta": d_lb});
                if (d_med > 1e-4) resultsTable.push({"Case": t, "Var": names[i], "Type": "Med", "Exp": ex_med, "Act": ac_med, "Delta": d_med});
                if (d_ub > 1e-4) resultsTable.push({"Case": t, "Var": names[i], "Type": "UB", "Exp": ex_ub, "Act": ac_ub, "Delta": d_ub});
            }
        }

        console.log("Validation complete.");
        if (resultsTable.length > 0) {
            console.warn("MISMATCHES FOUND:", resultsTable.length);
            console.table(resultsTable);
        } else {
            console.log("SUCCESS: All outputs match expected targets within 1e-4.");
        }
    } catch (e) {
        console.error("Validation failed", e);
    }
}
window.runSystemValidation = runSystemValidation;
