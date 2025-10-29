// Configuring URL and API
const getApiBaseUrl = () => {
    if (window.location.hostname === 'localhost' || window.location.hostname === '127.0.0.1') {
        return 'http://localhost:8000';
    } else {
        return 'https://sde-solver-app.onrender.com';
    }
};

const API_BASE_URL = getApiBaseUrl();

// Pastel colors palette for charts
const PASTEL_COLORS = [
    '#FFB6C1', '#87CEEB', '#98FB98', '#DDA0DD', 
    '#FFD700', '#F0E68C', '#E6E6FA', '#B0E0E6',
    '#FFA07A', '#20B2AA', '#DEB887', '#D8BFD8'
];

feather.replace();

// Navigation
document.querySelectorAll('.nav-link').forEach(link => {
    link.addEventListener('click', function(e) {
        e.preventDefault();
        const target = this.getAttribute('data-target');
        showSection(target);
        
        // Update active nav link
        document.querySelectorAll('.nav-link').forEach(l => {
            l.classList.remove('active-tab');
        });
        this.classList.add('active-tab');
    });
});

// Mobile menu
document.getElementById('mobileMenuButton').addEventListener('click', function() {
    const menu = document.getElementById('mobileMenu');
    menu.classList.toggle('hidden');
});

function showSection(sectionId) {
    // Hide all sections
    document.querySelectorAll('.section').forEach(section => {
        section.classList.add('hidden');
        section.classList.remove('active');
    });
    
    // Show target section
    const targetSection = document.getElementById(sectionId);
    targetSection.classList.remove('hidden');
    targetSection.classList.add('active');
}

// Improved LaTeX cleaning function
function cleanLatex(content) {
    if (!content) return '';
    
    let cleaned = content;
    
    // Remove excessive $$ and ensure proper formatting
    cleaned = cleaned
        .replace(/\$\s*\$/g, '') // Remove empty $$
        .replace(/\\\[\s*\\\]/g, '') // Remove empty \[]
        .replace(/\$\s*\\\s*\$/g, '') // Remove $ around LaTeX commands
        .replace(/([a-zA-Z])([A-Z])/g, '$1 $2') // Add space between words
        .trim();
    
    // Remove surrounding $ if they exist
    if (cleaned.startsWith('$') && cleaned.endsWith('$')) {
        cleaned = cleaned.substring(1, cleaned.length - 1);
    }
    
    // Remove surrounding \[ and \] if they exist
    if (cleaned.startsWith('\\[') && cleaned.endsWith('\\]')) {
        cleaned = cleaned.substring(2, cleaned.length - 2);
    }
    
    return cleaned;
}

// Load example buttons
document.querySelectorAll('.load-example').forEach(btn => {
    btn.addEventListener('click', function() {
        const drift = this.getAttribute('data-drift');
        const diffusion = this.getAttribute('data-diffusion');
        const params = this.getAttribute('data-params');
        const processType = this.getAttribute('data-process-type') || 'custom';
        
        document.getElementById('drift').value = drift;
        document.getElementById('diffusion').value = diffusion;
        document.getElementById('problemDescription').value = params;
        document.getElementById('processType').value = processType;
        
        showSection('solver');
    });
});

// Solve SDE function
async function solveSDE() {
    const btn = document.getElementById('solveBtn');
    const thinkingDiv = document.getElementById('aiThinking');
    const stepsContainer = document.getElementById('solutionSteps');
    const errorContainer = document.getElementById('errorContainer');
    const finalAnswer = document.getElementById('finalAnswer');
    const simulationResult = document.getElementById('simulationResult');

    // Reset state
    btn.disabled = true;
    thinkingDiv.classList.remove('hidden');
    stepsContainer.innerHTML = '';
    errorContainer.classList.add('hidden');
    finalAnswer.classList.add('hidden');
    simulationResult.classList.add('hidden');

    try {
        const equationType = document.querySelector('input[name="equationType"]:checked').value;
        const drift = document.getElementById('drift').value.trim();
        const diffusion = document.getElementById('diffusion').value.trim();
        const initialCondition = document.getElementById('initialCondition').value.trim();
        const timeVariable = document.getElementById('timeVariable').value.trim() || 't';
        const paramsText = document.getElementById('problemDescription').value.trim();
        const processType = document.getElementById('processType').value;  // Added as per correction

        if (!drift || !diffusion) {
            throw new Error('Please enter both drift and diffusion coefficients');
        }

        let parameters = {};
        if (paramsText) {
            try {
                parameters = JSON.parse(paramsText);
            } catch (e) {
                throw new Error('Invalid JSON in parameters field');
            }
        }

        const requestData = {
            equation_type: equationType,
            drift: drift,
            diffusion: diffusion,
            initial_condition: initialCondition,
            parameters: parameters,
            process_type: processType  // Added as per correction
        };

        const response = await fetch(`${API_BASE_URL}/solve`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(requestData)
        });

        if (!response.ok) {
            const errorData = await response.json();
            throw new Error(errorData.detail || 'Failed to solve SDE');
        }

        const data = await response.json();

        // Display steps
        data.steps.forEach((step, index) => {
            const stepElement = document.createElement('div');
            stepElement.className = 'solution-step bg-white p-6 rounded-lg shadow-md';
            stepElement.innerHTML = `
                <h4 class="font-bold text-teal-700 mb-2">Step {index + 1}: {step.title}</h4>
                <div class="formula-box">${step.content}$</div> 
            `;
            stepsContainer.appendChild(stepElement);
        });

        // Display final answer
        finalAnswer.innerHTML = `
            <h3 class="font-bold text-xl text-teal-700 mb-4">Final Solution</h3>
            <div class="formula-box">${data.final_solution}$</div>
        `;
        finalAnswer.classList.remove('hidden');

        // Re-render MathJax
        if (typeof MathJax !== 'undefined') {
            MathJax.Hub.Queue(["Typeset", MathJax.Hub, stepsContainer]);
            MathJax.Hub.Queue(["Typeset", MathJax.Hub, finalAnswer]);
        }

    } catch (error) {
        errorContainer.innerHTML = `
            <div class="bg-red-50 border-l-4 border-red-500 p-4 rounded">
                <h4 class="font-bold text-red-700 mb-2">Solving Error</h4>
                <p class="text-red-600">${error.message}</p>
                <p class="text-red-500 text-sm mt-2">Please check your equation syntax and try again.</p>
            </div>
        `;
        errorContainer.classList.remove('hidden');
    } finally {
        btn.disabled = false;
        thinkingDiv.classList.add('hidden');
    }
}

// Simulate SDE function
async function simulateSDE() {
    const btn = document.getElementById('simulateBtn');
    const thinkingDiv = document.getElementById('simulationThinking');
    const simulationResult = document.getElementById('simulationResult');
    const errorContainer = document.getElementById('simulationError');
    const plotContainer = document.getElementById('plotContainer');
    const downloadPlotBtn = document.getElementById('downloadPlotBtn');

    btn.disabled = true;
    thinkingDiv.classList.remove('hidden');
    simulationResult.classList.add('hidden');
    errorContainer.classList.add('hidden');
    plotContainer.innerHTML = '';

    try {
        const equationType = document.querySelector('input[name="equationType"]:checked').value;
        const drift = document.getElementById('drift').value.trim();
        const diffusion = document.getElementById('diffusion').value.trim();
        const initialCondition = document.getElementById('initialCondition').value.trim() || '0';
        const paramsText = document.getElementById('problemDescription').value.trim();
        const processType = document.getElementById('processType').value;
        const timeStart = parseFloat(document.getElementById('timeStart').value) || 0;
        const timeEnd = parseFloat(document.getElementById('timeEnd').value) || 1;
        const numPoints = parseInt(document.getElementById('numPoints').value) || 100;
        const numTrajectories = parseInt(document.getElementById('numTrajectories').value) || 5;

        if (!drift || !diffusion) {
            throw new Error('Please enter both drift and diffusion coefficients');
        }

        let parameters = {};
        if (paramsText) {
            parameters = JSON.parse(paramsText);
        }

        const requestData = {
            equation_type: equationType,
            drift: drift,
            diffusion: diffusion,
            initial_condition: initialCondition,
            parameters: parameters,
            time_span: [timeStart, timeEnd],
            num_points: numPoints,
            num_trajectories: numTrajectories,
            process_type: processType
        };

        const response = await fetch(`${API_BASE_URL}/simulate`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(requestData)
        });

        if (!response.ok) {
            const errorData = await response.json();
            throw new Error(errorData.detail || 'Failed to simulate SDE');
        }

        const data = await response.json();

        // Create Plotly chart
        const traces = data.trajectories.map((traj, i) => ({
            x: data.time_points,
            y: traj,
            mode: 'lines',
            name: `Trajectory ${i+1}`,
            line: { color: data.colors[i] }
        }));

        const layout = {
            title: 'SDE Simulation Trajectories',
            xaxis: { title: 'Time' },
            yaxis: { title: 'X(t)' },
            hovermode: 'closest',
            legend: { orientation: 'h' },
            margin: { t: 40, b: 40, l: 40, r: 40 }
        };

        Plotly.newPlot('plotContainer', traces, layout);

        // Enable download
        downloadPlotBtn.classList.remove('hidden');
        downloadPlotBtn.onclick = () => {
            Plotly.downloadImage('plotContainer', {
                format: 'png',
                width: 1200,
                height: 600,
                filename: 'sde_simulation'
            });
        };

        simulationResult.classList.remove('hidden');

    } catch (error) {
        errorContainer.innerHTML = `
            <div class="bg-red-50 border-l-4 border-red-500 p-4 rounded">
                <h4 class="font-bold text-red-700 mb-2">Simulation Error</h4>
                <p class="text-red-600">${error.message}</p>
            </div>
        `;
        errorContainer.classList.remove('hidden');
    } finally {
        btn.disabled = false;
        thinkingDiv.classList.add('hidden');
    }
}

// Function to get current solution data for export
function getCurrentSolutionData() {
    const equationType = document.querySelector('input[name="equationType"]:checked').value;
    const drift = document.getElementById('drift').value.trim();
    const diffusion = document.getElementById('diffusion').value.trim();
    const paramsText = document.getElementById('problemDescription').value.trim();

    const steps = Array.from(document.querySelectorAll('#solutionSteps .solution-step')).map(step => ({
        title: step.querySelector('h4').textContent.replace(/Step \d+: /, ''),
        content: step.querySelector('.formula-box').textContent
    }));

    const final_solution = document.querySelector('#finalAnswer .formula-box').textContent;

    let parameters = {};
    if (paramsText) {
        parameters = JSON.parse(paramsText);
    }

    return {
        equation_type: equationType,
        drift: drift,
        diffusion: diffusion,
        parameters: parameters,
        steps: steps,
        final_solution: final_solution
    };
}

// Download PDF
async function downloadPDF() {
    try {
        const data = getCurrentSolutionData();
        const response = await fetch(`${API_BASE_URL}/generate_pdf`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(data)
        });

        if (!response.ok) {
            throw new Error('Failed to generate PDF');
        }

        const blob = await response.blob();
        const url = window.URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = 'sde_solution.pdf';
        document.body.appendChild(a);
        a.click();
        document.body.removeChild(a);
        window.URL.revokeObjectURL(url);
    } catch (error) {
        console.error('Error downloading PDF:', error);
        // Show error message to user
    }
}

// Download LaTeX
async function downloadLatex() {
    try {
        const data = getCurrentSolutionData();
        const response = await fetch(`${API_BASE_URL}/generate_latex`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(data)
        });

        if (!response.ok) {
            throw new Error('Failed to generate LaTeX');
        }

        const blob = await response.blob();
        const url = window.URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = 'sde_solution.tex';
        document.body.appendChild(a);
        a.click();
        document.body.removeChild(a);
        window.URL.revokeObjectURL(url);
    } catch (error) {
        console.error('Error downloading LaTeX:', error);
        // Show error message to user
    }
}

// Handle expand buttons for formula boxes
document.addEventListener('click', function(e) {
    if (e.target.classList.contains('expand-btn')) {
        const box = e.target.closest('.formula-box');
        box.classList.toggle('collapsed');
        e.target.textContent = box.classList.contains('collapsed') ? 'Expand' : 'Collapse';
    }
});

// Configure process selector
function setupProcessSelector() {
    const processTypeSelect = document.getElementById('processType');
    processTypeSelect.addEventListener('change', function() {
        const processType = this.value;
        const defaultParams = {
            'custom': '{}',
            'geometric_brownian': '{"mu": 0.1, "sigma": 0.2, "S0": 100}',
            'ornstein_uhlenbeck': '{"theta": 1.0, "mu": 0.0, "sigma": 0.5, "x0": 0}',
            'vasicek': '{"a": 0.1, "b": 0.05, "sigma": 0.02, "r0": 0.05}',
            'cir': '{"a": 0.1, "b": 0.05, "sigma": 0.1, "r0": 0.05}',
            'brownian': '{}',
            'exponential_martingale': '{"mu": 0, "sigma": 0.2}',
            'poisson': '{"lambd": 1.0}',
            'jump_diffusion': '{"mu": 0.1, "sigma": 0.2, "lambd": 0.5, "jump_mean": 0, "jump_std": 0.1, "S0": 100}'
        };

        const defaultDrift = {
            'custom': 'mu*x',
            'geometric_brownian': 'mu*x',
            'ornstein_uhlenbeck': 'theta*(mu - x)',
            'vasicek': 'a*(b - x)',
            'cir': 'a*(b - x)',
            'brownian': '0',
            'exponential_martingale': '0',
            'poisson': '0',
            'jump_diffusion': 'mu*x'
        };

        const defaultDiffusion = {
            'custom': 'sigma*x',
            'geometric_brownian': 'sigma*x',
            'ornstein_uhlenbeck': 'sigma',
            'vasicek': 'sigma',
            'cir': 'sigma*sqrt(x)',
            'brownian': '1',
            'exponential_martingale': 'sigma*x',
            'poisson': '1',
            'jump_diffusion': 'sigma*x'
        };

        if (defaultParams[processType]) {
            document.getElementById('problemDescription').value = defaultParams[processType];
        }
        if (defaultDrift[processType]) {
            document.getElementById('drift').value = defaultDrift[processType];
        }
        if (defaultDiffusion[processType]) {
            document.getElementById('diffusion').value = defaultDiffusion[processType];
        }
    });
}

// Load examples from API
async function loadExamples() {
    try {
        const response = await fetch(`${API_BASE_URL}/examples`);
        const examples = await response.json();
        
        // Update examples in UI
        updateExamplesUI(examples);
    } catch (error) {
        console.error('Error loading examples:', error);
    }
}

function updateExamplesUI(examples) {
    const examplesContainer = document.getElementById('examplesContainer');
    if (!examplesContainer) return;
    
    let examplesHTML = '';
    
    for (const [key, example] of Object.entries(examples)) {
        examplesHTML += `
            <div class="bg-white rounded-lg shadow-md overflow-hidden">
                <div class="bg-teal-600 text-white p-4">
                    <h3 class="font-bold text-lg">${example.name}</h3>
                </div>
                <div class="p-4">
                    <p class="text-gray-700 mb-4">${example.description}</p>
                    <div class="space-y-2 text-sm">
                        <div><strong>Equation:</strong> \\( dX_t = ${example.drift}  dt + ${example.diffusion}  dW_t \\)</div>
                        <div><strong>Parameters:</strong> ${Object.entries(example.parameters).map(([k, v]) => `${k} = ${v}`).join(', ')}</div>
                    </div>
                    <button class="w-full bg-teal-600 text-white py-2 rounded mt-4 hover:bg-teal-700 transition load-example" 
                            data-drift="${example.drift}" data-diffusion="${example.diffusion}" 
                            data-params='${JSON.stringify(example.parameters)}' data-process-type="${example.process_type || 'custom'}">
                        Load Example
                    </button>
                </div>
            </div>
        `;
    }
    
    examplesContainer.innerHTML = examplesHTML;
    
    // Re-attach events to new buttons
    document.querySelectorAll('.load-example').forEach(btn => {
        btn.addEventListener('click', function() {
            const drift = this.getAttribute('data-drift');
            const diffusion = this.getAttribute('data-diffusion');
            const params = this.getAttribute('data-params');
            const processType = this.getAttribute('data-process-type') || 'custom';
            
            document.getElementById('drift').value = drift;
            document.getElementById('diffusion').value = diffusion;
            document.getElementById('problemDescription').value = params;
            document.getElementById('processType').value = processType;
            
            showSection('solver');
        });
    });
    
    // Re-render MathJax
    if (typeof MathJax !== 'undefined') {
        MathJax.Hub.Queue(["Typeset", MathJax.Hub]);
    }
}

// Initialization
document.addEventListener('DOMContentLoaded', function() {
    // Set active section
    showSection('solver');
    
    // Set active nav link
    document.querySelector('.nav-link[data-target="solver"]').classList.add('active-tab');
    
    // Configure process selector
    setupProcessSelector();
    
    // Load examples
    loadExamples();
    
    // Log the API URL for debugging
    console.log('API Base URL:', API_BASE_URL);
    console.log('Frontend URL:', window.location.href);

    // Attach event listeners for buttons
    document.getElementById('solveBtn').addEventListener('click', solveSDE);
    document.getElementById('simulateBtn').addEventListener('click', simulateSDE);
    document.getElementById('downloadPdfBtn').addEventListener('click', downloadPDF);
    document.getElementById('downloadLatexBtn').addEventListener('click', downloadLatex);
});
