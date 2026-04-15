// Mix Optimizer Logic

const OPTION_LABELS = {
    agg: ["None", "RAP", "RWP", "Steel Slag", "Crumb Rubber"],
    add: ["None", "Evotherm", "Zycotherm", "Sasobit"],
    fill: ["Limestone & Sandstone", "Cement", "Fly Ash", "Traprock", "Granite", "Hydrated Lime"],
    nmas: ["12.5 mm", "9.5 mm"]
};

// Extracted structurally from app.js data.py simulation
const dictCache = {
    add: { 0:[0,0,0,1], 1:[1,0,0,0], 2:[0,0,1,0], 3:[0,1,0,0] },
    filler: { 0:[0,0,0,0,0,1], 1:[1,0,0,0,0,0], 2:[0,1,0,0,0,0], 4:[0,0,1,0,0,0], 5:[0,0,0,1,0,0], 3:[0,0,0,0,1,0] },
    cr: { 0:[1,0], 1:[0,1] },
    nmas: { 0:[0,1], 1:[1,0] },
    agg: { 0:[0,0,0,1], 1:[1,0,0,0], 2:[0,1,0,0], 3:[0,0,1,0] }
};

window.runOptimizer = async function() {
    if (typeof ortSession === "undefined" || !ortSession) {
        alert("Model is not initialized. Please click 'Run Inference' on the Prediction panel first to decrypt and load the ONNX model.");
        return;
    }

    const overlay = document.getElementById('opt-loading-overlay');
    const statusText = document.getElementById('opt-status-text');
    overlay.classList.remove('hidden');
    overlay.classList.add('flex');
    
    document.getElementById('opt-results-count').textContent = "Processing...";

    // 1. Build Search Space
    const space = {
        ac: document.getElementById('opt-lock-ac').checked ? 
            [parseFloat(document.getElementById('inp-ac').value) / 100.0] : 
            [0.05, 0.055, 0.06, 0.065, 0.07, 0.075, 0.08],
        agg: document.getElementById('opt-lock-agg').checked ? 
            [parseInt(document.getElementById('inp-aggregate').value)] : 
            [0, 1, 2, 3, 4],
        add: document.getElementById('opt-lock-add').checked ? 
            [parseInt(document.getElementById('inp-additive').value)] : 
            [0, 1, 2, 3],
        fill: document.getElementById('opt-lock-fill').checked ? 
            [parseInt(document.getElementById('inp-filler').value)] : 
            [0, 1, 2, 3, 4, 5],
        nmas: document.getElementById('opt-lock-nmas').checked ? 
            [parseInt(document.getElementById('inp-nmas').value)] : 
            [0, 1]
    };

    // Grab fixed gradations (always locked)
    const sKeys = ["19", "12p5", "9p5", "4p75", "2p36", "1p18", "0p60", "0p30", "0p15", "0p075"];
    const fixedSList = sKeys.map(k => parseFloat(document.getElementById(`inp-${k}`).value) / 100.0);

    // 2. Generate Permutations
    let permutations = [];
    for(let ac of space.ac) {
        for(let agg of space.agg) {
            for(let add of space.add) {
                for(let fill of space.fill) {
                    for(let nmas of space.nmas) {
                        permutations.push({ac, agg, add, fill, nmas});
                    }
                }
            }
        }
    }

    statusText.textContent = `Evaluating ${permutations.length} permutations...`;
    
    // We process in small chunks to avoid freezing the browser completely
    let validMixes = [];
    const chunkSize = 100;

    for (let i = 0; i < permutations.length; i += chunkSize) {
        const chunk = permutations.slice(i, i + chunkSize);
        
        // Execute chunk sequentially
        for (const p of chunk) {
            let agg_mod = p.agg;
            let cr_mod = 0;
            if (p.agg === 4) {
                agg_mod = 0;
                cr_mod = 1;
            }

            const flatValues = [
                ...dictCache.add[p.add],
                ...dictCache.filler[p.fill],
                ...dictCache.cr[cr_mod],
                ...dictCache.nmas[p.nmas],
                ...dictCache.agg[agg_mod],
                p.ac,
                ...fixedSList
            ];

            const tensor = new ort.Tensor('float32', Float32Array.from(flatValues), [1, 29]);
            const results = await ortSession.run({ input: tensor });
            const output = results.output.data; 

            const AV = output[0*3 + 1] * 100;
            const TSR = output[1*3 + 1] * 100;
            const Rut5k = output[2*3 + 1];
            const Rut20k = output[3*3 + 1];
            const Cantabro = output[4*3 + 1] * 100;

            // Strict Specification Checks
            if (AV >= 0 && AV <= 24 &&
                TSR >= 70 && TSR <= 100 &&
                Rut5k >= 0 && Rut5k <= 0.5 &&
                Rut20k >= 0 && Rut20k <= 0.5 &&
                Cantabro >= 0 && Cantabro <= 20) {
                
                // Score based on distance to ideal centers (Cost function: lower = safer)
                // Ideal AV ~ 12, Ideal TSR ~ 100, Rut ~ 0, Cantabro ~ 0
                const score = (Math.abs(AV - 12)/12) + (Math.abs(100 - TSR)/30) + (Rut5k/0.5) + (Rut20k/0.5) + (Cantabro/20);
                validMixes.push({ p, score, AV, TSR, Rut5k, Rut20k, Cantabro });
            }
        }
        
        // Update UI
        statusText.textContent = `Evaluated ${Math.min(i + chunkSize, permutations.length)} of ${permutations.length}...`;
        await new Promise(r => setTimeout(r, 1)); // Yield thread to browser paint
    }

    validMixes.sort((a,b) => a.score - b.score);
    const topMixes = validMixes.slice(0, 3);
    
    // 3. Render Results
    const container = document.getElementById('optimizer-results-container');
    container.innerHTML = '';
    
    if (topMixes.length === 0) {
        container.innerHTML = `
            <div class="p-8 text-center text-rose-500 bg-rose-50 rounded-lg border border-rose-100 flex flex-col items-center">
                <svg class="w-12 h-12 mb-4 text-rose-400" fill="none" viewBox="0 0 24 24" stroke="currentColor"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M12 9v2m0 4h.01m-6.938 4h13.856c1.54 0 2.502-1.667 1.732-3L13.732 4c-.77-1.333-2.694-1.333-3.464 0L3.34 16c-.77 1.333.192 3 1.732 3z"/></svg>
                <h3 class="font-bold text-lg">No compliant combinations found.</h3>
                <p class="text-sm mt-2">Try relaxing your locked parameters or adjusting your fixed gradation curve.</p>
            </div>`;
        document.getElementById('opt-results-count').textContent = `0 found`;
    } else {
        document.getElementById('opt-results-count').textContent = `${validMixes.length} found`;
        
        topMixes.forEach((mix, idx) => {
            const badge = idx === 0 ? `<span class="bg-indigo-600 text-white text-xs font-bold px-2 py-1 rounded shadow-sm relative -top-3">Recommended</span>` : '';
            const html = `
                <div class="bg-white rounded-lg border ${idx === 0 ? 'border-indigo-400 ring-1 ring-indigo-400 shadow-md' : 'border-slate-200'} p-5 relative">
                    ${badge}
                    <div class="flex justify-between items-start mb-4">
                        <div>
                            <h3 class="font-bold text-slate-800 text-lg">Mix Recommendation #${idx + 1}</h3>
                            <p class="text-slate-500 text-xs mt-1">Safety Cost Score: ${mix.score.toFixed(3)}</p>
                        </div>
                        <button onclick="applyMix(${mix.p.ac}, ${mix.p.agg}, ${mix.p.add}, ${mix.p.fill}, ${mix.p.nmas})" class="text-xs font-semibold bg-emerald-100 text-emerald-800 hover:bg-emerald-200 px-3 py-1.5 rounded transition">Apply to Panel &rarr;</button>
                    </div>
                    
                    <div class="grid grid-cols-2 md:grid-cols-5 gap-3 mb-5 bg-slate-50 p-3 rounded border border-slate-100">
                        <div>
                            <p class="text-[10px] text-slate-400 uppercase font-bold tracking-wider">AC Content</p>
                            <p class="font-mono text-slate-800 font-semibold text-sm">${(mix.p.ac * 100).toFixed(1)}%</p>
                        </div>
                        <div>
                            <p class="text-[10px] text-slate-400 uppercase font-bold tracking-wider">Spec. Material</p>
                            <p class="font-mono text-slate-800 font-semibold text-sm truncate" title="${OPTION_LABELS.agg[mix.p.agg]}">${OPTION_LABELS.agg[mix.p.agg]}</p>
                        </div>
                        <div>
                            <p class="text-[10px] text-slate-400 uppercase font-bold tracking-wider">Additive</p>
                            <p class="font-mono text-slate-800 font-semibold text-sm truncate" title="${OPTION_LABELS.add[mix.p.add]}">${OPTION_LABELS.add[mix.p.add]}</p>
                        </div>
                        <div>
                            <p class="text-[10px] text-slate-400 uppercase font-bold tracking-wider">Filler</p>
                            <p class="font-mono text-slate-800 font-semibold text-sm truncate" title="${OPTION_LABELS.fill[mix.p.fill]}">${OPTION_LABELS.fill[mix.p.fill]}</p>
                        </div>
                        <div>
                            <p class="text-[10px] text-slate-400 uppercase font-bold tracking-wider">NMAS</p>
                            <p class="font-mono text-slate-800 font-semibold text-sm">${OPTION_LABELS.nmas[mix.p.nmas]}</p>
                        </div>
                    </div>

                    <div class="flex gap-4 border-t border-slate-100 pt-3">
                        <div class="flex-1">
                            <p class="text-[10px] text-slate-400 uppercase font-bold tracking-wider">Air Void</p>
                            <p class="text-sm font-bold text-emerald-700">${mix.AV.toFixed(2)}%</p>
                        </div>
                        <div class="flex-1">
                            <p class="text-[10px] text-slate-400 uppercase font-bold tracking-wider">TSR</p>
                            <p class="text-sm font-bold text-emerald-700">${mix.TSR.toFixed(2)}%</p>
                        </div>
                        <div class="flex-1">
                            <p class="text-[10px] text-slate-400 uppercase font-bold tracking-wider">Rut 5k</p>
                            <p class="text-sm font-bold text-emerald-700">${mix.Rut5k.toFixed(3)}</p>
                        </div>
                        <div class="flex-1">
                            <p class="text-[10px] text-slate-400 uppercase font-bold tracking-wider">Rut 20k</p>
                            <p class="text-sm font-bold text-emerald-700">${mix.Rut20k.toFixed(3)}</p>
                        </div>
                        <div class="flex-1 border-l border-slate-200 pl-4">
                            <p class="text-[10px] text-slate-400 uppercase font-bold tracking-wider">Cantabro</p>
                            <p class="text-sm font-bold text-emerald-700">${mix.Cantabro.toFixed(2)}%</p>
                        </div>
                    </div>
                </div>
            `;
            container.insertAdjacentHTML('beforeend', html);
        });
    }

    overlay.classList.remove('flex');
    overlay.classList.add('hidden');
};

// Helper function to apply the mix directly to the dashboard
window.applyMix = function(ac, agg, add, fill, nmas) {
    document.getElementById('inp-ac').value = (ac * 100).toFixed(1);
    document.getElementById('val-ac').textContent = (ac * 100).toFixed(1);
    document.getElementById('inp-aggregate').value = agg;
    document.getElementById('inp-additive').value = add;
    document.getElementById('inp-filler').value = fill;
    document.getElementById('inp-nmas').value = nmas;
    
    // Switch to prediction tab and run
    if(window.switchTab) window.switchTab('prediction');
    if(window.runInference) window.runInference();
};
