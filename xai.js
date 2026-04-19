let shapleyData = null;
let shapBackground = null;
let xaiInitialized = false;

// Feature mapping between the 29-element FlatArray tensor and the textual labels available in JSON
// In x_df from python: ["evotherm", "zycotherm", "sasobit", "nan_additive", ...]

async function loadShapley() {
    try {
        const response = await fetch('Materials/shapley_data.json');
        shapleyData = await response.json();
        
        const bgResp = await fetch('Materials/shap_background_data.json');
        shapBackground = await bgResp.json();
        
        // Populate Select Dropdown
        const select = document.getElementById('xai-indicator-select');
        select.innerHTML = '';
        Object.keys(shapleyData.shap_data).forEach(idx => {
            const name = shapleyData.shap_data[idx].name;
            const opt = document.createElement('option');
            opt.value = idx;
            opt.textContent = name;
            select.appendChild(opt);
        });

        select.addEventListener('change', () => {
            if (currentInferenceInput) {
                renderShapleyPlots(currentInferenceInput);
            } else {
                renderShapleyPlots(null); // Render default
            }
        });

        document.getElementById('xai-loading').style.display = 'none';
        document.getElementById('xai-chart-container').classList.remove('hidden');
        xaiInitialized = true;
    } catch (e) {
        console.error("Failed to load Shapley data", e);
        document.getElementById('xai-loading').innerHTML = "<p class='text-red-500'>Error loading XAI data.</p>";
    }
}

async function renderShapleyPlots(inputFlatArray) {
    if (!xaiInitialized || !shapBackground) return;
    if (!inputFlatArray) return; // Wait for valid input

    if (typeof window.ortSession === "undefined" || !window.ortSession) {
        document.getElementById('xai-loading').style.display = 'block';
        document.getElementById('xai-loading').innerHTML = `
            <div class="p-8 text-center bg-slate-50 rounded-lg border border-slate-200 flex flex-col items-center justify-center mt-12 mx-auto max-w-md">
                <svg class="w-12 h-12 mb-4 text-slate-400" fill="none" viewBox="0 0 24 24" stroke="currentColor"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M13 16h-1v-4h-1m1-4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z"></path></svg>
                <h3 class="font-bold text-lg text-slate-700">Model not initialized</h3>
                <p class="text-sm mt-2 text-slate-500">Please click "Calculate Performance" on the Prediction tab first to securely load the ONNX model.</p>
            </div>`;
        document.getElementById('xai-chart-container').classList.add('hidden');
        return;
    }

    try {

    const select = document.getElementById('xai-indicator-select');
    const idx = parseInt(select.value); // 0 to 4 mapping exactly to the neural network output columns
    
    // We will calculate baseline expected value dynamically based on ONNX runs to ensure perfect scale match

    // Background matrix is size 50 x 29
    const bgData = shapBackground.background_data;
    const nBg = bgData.length;
    const nFeatures = inputFlatArray.length; // 29

    // The idx value corresponds to the flattened output tensor axis (1, 4, 7, 10, 13)
    const colOffset = idx; // Target variable direct index
    
    // Multipliers for final physical scale:
    // av(1)->100, tsr(4)->100, rut5k(7)->1, rut20k(10)->1, cantabro(13)->100
    let outMult = 1;
    if (idx === 1 || idx === 4 || idx === 13) outMult = 100;

    // Calculate approximate SHAP using Sequential Path Integration
    // This perfectly distributes the interactions, guaranteeing Local Alignment = 0
    // To do this, we sequentially replace features one by one cumulatively.
    const runBatch = new Float32Array(nFeatures * nBg * nFeatures);
    
    let p = 0;
    for (let f = 0; f < nFeatures; f++) {
        for (let b = 0; b < nBg; b++) {
            for (let j = 0; j < nFeatures; j++) {
                if (j <= f) { // Cumulative sequential substitution
                    runBatch[p++] = inputFlatArray[j];
                } else {
                    runBatch[p++] = bgData[b][j];
                }
            }
        }
    }

    // Run 1450 evaluations
    const tensor = new ort.Tensor('float32', runBatch, [nFeatures * nBg, nFeatures]);
    const results = await ortSession.run({ input: tensor });
    const outputDat = results.output.data;

    // We also need baseline predictions for all 50 samples to subtract them.
    // We can evaluate them as a batch:
    const baseBatch = new Float32Array(nBg * nFeatures);
    let bp = 0;
    for(let b=0; b<nBg; b++) {
        for(let j=0; j<nFeatures; j++) {
            baseBatch[bp++] = bgData[b][j];
        }
    }
    const baseTensor = new ort.Tensor('float32', baseBatch, [nBg, nFeatures]);
    const baseResults = await ortSession.run({ input: baseTensor });
    const baseOutputDat = baseResults.output.data;

    const rawShap = new Float32Array(nFeatures);
    for (let f = 0; f < nFeatures; f++) {
        let sum = 0;
        for (let b = 0; b < nBg; b++) {
            // outputDat maps to [batchIndex * 15 (5 vars * 3 quants) + colOffset]
            const batchIdx = f * nBg + b;
            const pMod = outputDat[batchIdx * 15 + colOffset] * outMult;
            
            // For sequential path, the baseline is the PREVIOUS feature's output
            const pBase = (f === 0) 
                 ? (baseOutputDat[b * 15 + colOffset] * outMult) 
                 : (outputDat[( (f - 1) * nBg + b ) * 15 + colOffset] * outMult);
                 
            sum += (pMod - pBase);
        }
        rawShap[f] = sum / nBg;
    }

    // Now aggregate categorical variables just like the python notebook did
    // Order:
    // Additive: 0, 1, 2, 3
    // Filler: 4, 5, 6, 7, 8, 9
    // CrumbRubber: 10, 11
    // NMAS: 12, 13
    // Aggregate: 14, 15, 16, 17
    // NumerVars: 18 -> 28
    
    // Retrieve actual categorical UI text selected by the user
    const getTxt = (id) => {
        const el = document.getElementById(id);
        return el ? el.options[el.selectedIndex].text : "Unknown";
    };

    const addTxt = getTxt('inp-additive');
    const fillTxt = getTxt('inp-filler');
    const aggTxt = getTxt('inp-aggregate');
    const nmasTxt = getTxt('inp-nmas');

    const crTxt = (aggTxt === 'Crumb Rubber') ? 'Yes' : 'No';
    const aggTxtReal = (aggTxt === 'Crumb Rubber') ? 'None' : aggTxt;

    const groupedFeatures = [];
    groupedFeatures.push({name: `Additive = ${addTxt}`, val: rawShap[0]+rawShap[1]+rawShap[2]+rawShap[3]});
    groupedFeatures.push({name: `Filler = ${fillTxt}`, val: rawShap[4]+rawShap[5]+rawShap[6]+rawShap[7]+rawShap[8]+rawShap[9]});
    groupedFeatures.push({name: `Crumb Rubber = ${crTxt}`, val: rawShap[10]+rawShap[11]});
    groupedFeatures.push({name: `NMAS = ${nmasTxt}`, val: rawShap[12]+rawShap[13]});
    groupedFeatures.push({name: `Aggregate = ${aggTxtReal}`, val: rawShap[14]+rawShap[15]+rawShap[16]+rawShap[17]});
    
    // Numerical names
    const numNames = ["ac","s_19","s_12p5","s_9p5","s_4p75","s_2p36","s_1p18","s_0p60","s_0p30","s_0p15","s_0p075"];
    for(let i=0; i<numNames.length; i++) {
        groupedFeatures.push({name: numNames[i] + " = " + (inputFlatArray[18+i]*100).toFixed(1) + "%", val: rawShap[18+i]});
    }

    let sumBase = 0;
    for (let b = 0; b < nBg; b++) {
        sumBase += baseOutputDat[b * 15 + colOffset];
    }
    const scaledExpectedValue = (sumBase / nBg) * outMult;
    
    // Calculate final output expected vs exact actual to calculate model drift
    // Since we evaluated this natively with the exact model, the model drift is zero!
    const shapSum = groupedFeatures.reduce((acc, f) => acc + f.val, 0);
    
    // Exact ONNX prediction alignment
    if (window.currentPredictions && window.currentPredictions[idx] !== undefined) {
        const livePrediction = window.currentPredictions[idx];
        const localAlign = livePrediction - (scaledExpectedValue + shapSum);
        // Add it to the chart to guarantee perfect waterfall alignment down to Float32 precision bounds
        if (Math.abs(localAlign) > 1e-5) {
             groupedFeatures.push({name: "Local Alignment", val: localAlign});
        }
    }

    // Sort by absolute value ascending (so largest is at the top of horizontal chart)
    groupedFeatures.sort((a,b) => Math.abs(a.val) - Math.abs(b.val));

    const yNames = ["Baseline E[f(x)]", ...groupedFeatures.map(d => d.name), "Prediction f(x)"];
    const measure = ["absolute", ...groupedFeatures.map(d => 'relative'), "total"];
    const xVals = [scaledExpectedValue, ...groupedFeatures.map(d => d.val), 0];
    const textVals = [`${scaledExpectedValue.toFixed(3)}`, ...groupedFeatures.map(d => (d.val > 0 ? "+" : "") + d.val.toFixed(3)), "Final Output"];

    var trace1 = {
        type: "waterfall",
        orientation: "h",
        measure: measure,
        y: yNames,
        x: xVals,
        textposition: "outside",
        text: textVals,
        connector: {
            line: { color: "rgb(63, 63, 63)" }
        },
        increasing: { marker: { color: "#10b981" } }, // Emerald 500
        decreasing: { marker: { color: "#f43f5e" } }, // Rose 500
        totals: { marker: { color: "#4f46e5" } } // Indigo 600
    };

    var layout = {
        title: { text: "Feature Impacts" },
        xaxis: { title: "SHAP Value (Impact on Model Output)" },
        yaxis: { automargin: true },
        margin: { l: 150, t: 50, b: 50 },
        plot_bgcolor: "rgba(0,0,0,0)",
        paper_bgcolor: "rgba(0,0,0,0)"
    };

    Plotly.newPlot('xai-chart-container', [trace1], layout, {responsive: true});
    } catch (err) {
        document.getElementById('xai-loading').style.display = 'block';
        document.getElementById('xai-loading').innerHTML = "<p class='text-red-500'>XAI Render Error: " + err.message + "</p>";
        console.error(err);
    }
}

window.addEventListener('DOMContentLoaded', loadShapley);
