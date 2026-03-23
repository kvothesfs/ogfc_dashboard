let shapleyData = null;
let xaiInitialized = false;

// Feature mapping between the 29-element FlatArray tensor and the textual labels available in JSON
// In x_df from python: ["evotherm", "zycotherm", "sasobit", "nan_additive", ...]
// However, shapley charts just need numeric/categorical visual representations, which we will match to the closest sampled point in shap_data.

async function loadShapley() {
    try {
        const response = await fetch('Materials/shapley_data.json');
        shapleyData = await response.json();
        
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

function renderShapleyPlots(inputFlatArray) {
    if (!xaiInitialized) return;
    
    // Select correct indicator
    const select = document.getElementById('xai-indicator-select');
    const idx = select.value;
    const targetData = shapleyData.shap_data[idx];
    const expectedValue = shapleyData.expected_values[idx];

    // For dynamic XAI, because the DeepExplainer outputs SHAP for every training sample locally,
    // we need to approximate the SHAP values using the baseline or use an average.
    // In a fully dynamic local system, ShapExplainer would run on the ONNX model.
    // Because deep explainer requires gradients or heavy torch logic, we visualize the global distribution
    // and highlight the "Closest Profile" based on Euclidean distance to `inputFlatArray`.

    let closestRowIndex = 0;
    if (inputFlatArray) {
        // find row in targetData.x_df that minimizes distance. 
        // Note: x_df in the JSON is a dictionary array, but it aligns with the numerical values we can map to
        // Wait, x_df in the extraction script contains categorical names.
        // It's just a proxy for the Waterfall.
        closestRowIndex = 0; // fallback to 0 for now as true tensor-to-df mapping would require dictionary unrolling
    }

    const shapRow = targetData.shap_df[closestRowIndex];
    const xRow = targetData.x_df[closestRowIndex];
    // xRow holds the names of columns as keys.
    const keys = Object.keys(xRow);
    const contributions = keys.map((k, i) => shapRow[i]);
    
    // Sort for waterfall
    const items = keys.map((k, i) => ({
        name: k,
        val: contributions[i],
        textVal: String(xRow[k])
    })).sort((a,b) => Math.abs(a.val) - Math.abs(b.val));

    const yNames = ["Baseline E[f(x)]", ...items.map(d => d.name), "Prediction f(x)"];
    const measure = ["absolute", ...items.map(d => 'relative'), "total"];
    const xVals = [expectedValue, ...items.map(d => d.val), 0];
    const textVals = [`${expectedValue.toFixed(2)}`, ...items.map(d => d.textVal), "Final Output"];

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
        margin: { l: 150 },
        plot_bgcolor: "rgba(0,0,0,0)",
        paper_bgcolor: "rgba(0,0,0,0)"
    };

    Plotly.newPlot('xai-chart-container', [trace1], layout, {responsive: true});
}

window.addEventListener('DOMContentLoaded', loadShapley);
