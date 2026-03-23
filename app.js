// Global variables
let ortSession = null;
let currentInferenceInput = null;

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
}

function resetGradation() {
    gradations.forEach((g) => {
        const el = document.getElementById(`inp-${g.id}`);
        el.value = g.default;
        document.getElementById(`val-${g.id}`).textContent = g.default.toFixed(1) + '%';
    });
}

function switchTab(tabId) {
    document.getElementById('section-prediction').classList.add('hidden');
    document.getElementById('section-xai').classList.add('hidden');
    document.getElementById('tab-prediction').classList.remove('border-indigo-600', 'text-indigo-600');
    document.getElementById('tab-prediction').classList.add('border-transparent', 'text-slate-500');
    document.getElementById('tab-xai').classList.remove('border-indigo-600', 'text-indigo-600');
    document.getElementById('tab-xai').classList.add('border-transparent', 'text-slate-500');

    document.getElementById(`section-${tabId}`).classList.remove('hidden');
    document.getElementById(`tab-${tabId}`).classList.add('border-indigo-600', 'text-indigo-600');
    document.getElementById(`tab-${tabId}`).classList.remove('border-transparent', 'text-slate-500');

    if (tabId === 'xai' && typeof renderShapleyPlots === 'function') {
        const tensorParams = prepareTensorInputs();
        // Notify XAI tab that we opened it so it re-renders
        renderShapleyPlots(tensorParams.flatValues);
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
        document.getElementById('prediction-results-body').innerHTML = ''; // clear loading state

        for (let i = 0; i < 5; i++) {
            const lb = output[i * 3 + 0] * multiplier[i];
            const median = output[i * 3 + 1] * multiplier[i];
            const ub = output[i * 3 + 2] * multiplier[i];

            const inSpec = (median >= boundsOk.lower[i] && median <= boundsOk.upper[i]);

            // create div for this bullet plot
            const divId = `bullet-chart-${i}`;
            const divHtml = `<div id="${divId}" class="w-full h-[60px] pb-2"></div>`;
            document.getElementById('prediction-results-body').insertAdjacentHTML('beforeend', divHtml);

            let specThreshold = 0;
            let axisMax = 100;
            if(i === 0) { specThreshold = boundsOk.upper[0]; axisMax = Math.max(30, ub + 5); } // AV
            else if(i===1) { specThreshold = boundsOk.lower[1]; axisMax = Math.max(100, ub + 5); } // TSR
            else if(i===2) { specThreshold = boundsOk.upper[2]; axisMax = Math.max(1.0, ub + 0.2); } // Rut 5k
            else if(i===3) { specThreshold = boundsOk.upper[3]; axisMax = Math.max(1.0, ub + 0.2); } // Rut 20k
            else if(i===4) { specThreshold = boundsOk.upper[4]; axisMax = Math.max(40, ub + 5); } // Cantabro

            var plotData = [
              {
                type: "indicator",
                mode: "number+gauge",
                value: median,
                number: { valueformat: ".2f", font: { size: 24, color: inSpec ? "#4338ca" : "#e11d48" } },
                domain: { x: [0.35, 1], y: [0.2, 0.8] },
                title: { text: `<span style="font-size:0.8em;color:#475569;font-weight:600">${perfVarList[i]}</span>` },
                gauge: {
                  shape: "bullet",
                  axis: { range: [0, axisMax], tickcolor: "#94a3b8" },
                  threshold: {
                    line: { color: "#e11d48", width: 3 },
                    thickness: 0.8,
                    value: specThreshold
                  },
                  steps: [
                    { range: [lb, ub], color: "#e2e8f0" } 
                  ],
                  bar: { color: inSpec ? "#4f46e5" : "#f43f5e", thickness: 0.4 } 
                }
              }
            ];

            var layout = { margin: { t: 10, b: 20, l: 10, r: 40 }, paper_bgcolor: "rgba(0,0,0,0)", font: {family: "ui-sans-serif, system-ui, sans-serif"} };
            Plotly.newPlot(divId, plotData, layout, {displayModeBar: false, responsive: true});
        }

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
