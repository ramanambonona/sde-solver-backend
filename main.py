import os
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, FileResponse
from pydantic import BaseModel
from typing import List, Optional, Dict, Any, Union
import sympy as sp
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import io
import base64
import json
from sde_solver import SDESolver
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib import colors
from reportlab.lib.units import inch
import tempfile
import uuid

app = FastAPI(title="SDE Symbolic Solver API", version="1.0.0")

# CORS middleware
origins = [
    "http://localhost:3000",
    "http://localhost:8000",
    "https://sde-solver.netlify.app",
    "https://*.netlify.app",
]

# Netlify URL with production
if os.environ.get("NETLIFY_URL"):
    origins.append(os.environ.get("NETLIFY_URL"))

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class SDEProblem(BaseModel):
    equation_type: str  # "ito" or "stratonovich"
    drift: str
    diffusion: str
    initial_condition: Optional[str] = None
    variables: Optional[Dict[str, str]] = None
    parameters: Optional[Dict[str, float]] = None

class SimulationRequest(BaseModel):
    equation_type: str
    drift: str
    diffusion: str
    initial_condition: str
    parameters: Dict[str, float]
    time_span: List[float] = [0, 1]
    num_points: int = 100
    num_trajectories: int = 5
    process_type: str = "custom"  # New field for process type

class SolutionResponse(BaseModel):
    steps: List[Dict[str, str]]
    final_solution: str
    solution_type: str
    metadata: Dict[str, Any]

class SimulationResponse(BaseModel):
    plot_data: str  # base64 encoded image
    time_points: List[float]
    trajectories: List[List[float]]
    colors: List[str]  # New: colors for each trajectory

class ExtendedStochasticProcesses:
    """Extended class for all types of stochastic processes"""

    def __init__(self):
        self.pastel_colors = [
            '#FFB6C1', '#87CEEB', '#98FB98', '#DDA0DD',
            '#FFD700', '#F0E68C', '#E6E6FA', '#B0E0E6',
            '#FFA07A', '#20B2AA', '#DEB887', '#D8BFD8'
        ]

    def geometric_brownian_motion(self, S0, mu, sigma, T, dt, n_paths=1):
        """Geometric Brownian Motion"""
        t = np.arange(0, T+dt, dt)
        n_steps = len(t)

        paths = np.zeros((n_paths, n_steps))
        paths[:, 0] = S0

        for i in range(1, n_steps):
            dW = np.random.normal(0, np.sqrt(dt), n_paths)
            paths[:, i] = paths[:, i-1] * np.exp(
                (mu - 0.5*sigma**2)*dt + sigma*dW
            )

        return t, paths

    def ornstein_uhlenbeck(self, x0, theta, mu, sigma, T, dt, n_paths=1):
        """Ornstein-Uhlenbeck Process"""
        t = np.arange(0, T+dt, dt)
        n_steps = len(t)

        paths = np.zeros((n_paths, n_steps))
        paths[:, 0] = x0

        for i in range(1, n_steps):
            dW = np.random.normal(0, np.sqrt(dt), n_paths)
            paths[:, i] = paths[:, i-1] + theta*(mu - paths[:, i-1])*dt + sigma*dW

        return t, paths

    def vasicek_model(self, r0, k, theta, sigma, T, dt, n_paths=1):
        """Vasicek Model (interest rate)"""
        return self.ornstein_uhlenbeck(r0, k, theta, sigma, T, dt, n_paths)

    def cir_model(self, r0, k, theta, sigma, T, dt, n_paths=1):
        """CIR Model (Cox-Ingersoll-Ross)"""
        t = np.arange(0, T+dt, dt)
        n_steps = len(t)

        paths = np.zeros((n_paths, n_steps))
        paths[:, 0] = r0

        for i in range(1, n_steps):
            dW = np.random.normal(0, np.sqrt(dt), n_paths)
            drift = k*(theta - paths[:, i-1])*dt
            diffusion = sigma*np.sqrt(np.maximum(paths[:, i-1], 0))*dW
            paths[:, i] = paths[:, i-1] + drift + diffusion

        return t, paths

    def brownian_motion(self, T, dt, n_paths=1):
        """Standard Brownian Motion (martingale)"""
        t = np.arange(0, T+dt, dt)
        n_steps = len(t)

        paths = np.zeros((n_paths, n_steps))

        for i in range(1, n_steps):
            dW = np.random.normal(0, np.sqrt(dt), n_paths)
            paths[:, i] = paths[:, i-1] + dW

        return t, paths

    def exponential_martingale(self, mu, sigma, T, dt, n_paths=1):
        """Exponential Martingale"""
        t, W = self.brownian_motion(T, dt, n_paths)

        paths = np.exp(sigma * W - 0.5 * sigma**2 * t)
        return t, paths

    def poisson_process(self, lambd, T, dt, n_paths=1):
        """Poisson Process"""
        t = np.arange(0, T+dt, dt)
        n_steps = len(t)

        paths = np.zeros((n_paths, n_steps))

        for i in range(1, n_steps):
            jumps = np.random.poisson(lambd * dt, n_paths)
            paths[:, i] = paths[:, i-1] + jumps

        return t, paths

    def jump_diffusion(self, S0, mu, sigma, lambd, jump_mean, jump_std, T, dt, n_paths=1):
        """Jump Diffusion Process (Merton)"""
        t = np.arange(0, T+dt, dt)
        n_steps = len(t)

        paths = np.zeros((n_paths, n_steps))
        paths[:, 0] = S0

        for i in range(1, len(t)):
            dW = np.random.normal(0, np.sqrt(dt), n_paths)
            jump_sizes = np.random.normal(jump_mean, jump_std, n_paths)
            jump_occurrence = np.random.poisson(lambd * dt, n_paths)

            paths[:, i] = paths[:, i-1] * np.exp(
                (mu - 0.5*sigma**2)*dt + sigma*dW
            ) + jump_sizes * jump_occurrence

        return t, paths

    def custom_sde_simulation(self, drift_func, diffusion_func, parameters,
                              time_span, num_points, num_trajectories, initial_condition):
        """Custom SDE simulation with Euler-Maruyama"""
        t = np.linspace(time_span[0], time_span[1], num_points)
        dt = t[1] - t[0]

        paths = np.zeros((num_trajectories, num_points))
        paths[:, 0] = initial_condition

        for i in range(1, num_points):
            dW = np.random.normal(0, np.sqrt(dt), num_trajectories)

            # Evaluate the drift and diffusion functions
            drift_vals = drift_func(paths[:, i-1], t[i-1], **parameters)
            diffusion_vals = diffusion_func(paths[:, i-1], t[i-1], **parameters)

            paths[:, i] = paths[:, i-1] + drift_vals * dt + diffusion_vals * dW

        return t, paths

def generate_stochastic_process(process_type, parameters, time_span, num_points, num_trajectories):
    """Generate stochastic processes with the extended class"""
    sp = ExtendedStochasticProcesses()

    T = time_span[1] - time_span[0]
    dt = T / num_points

    process_map = {
        'geometric_brownian': sp.geometric_brownian_motion,
        'ornstein_uhlenbeck': sp.ornstein_uhlenbeck,
        'vasicek': sp.vasicek_model,
        'cir': sp.cir_model,
        'brownian': sp.brownian_motion,
        'exponential_martingale': sp.exponential_martingale,
        'poisson': sp.poisson_process,
        'jump_diffusion': sp.jump_diffusion
    }

    if process_type in process_map:
        try:
            # Extract process-specific parameters
            common_params = {
                'T': T, 'dt': dt, 'n_paths': num_trajectories
            }

            if process_type == 'geometric_brownian':
                func_params = {'S0': parameters.get('S0', 100),
                              'mu': parameters.get('mu', 0.1),
                              'sigma': parameters.get('sigma', 0.2)}
            elif process_type == 'ornstein_uhlenbeck':
                func_params = {'x0': parameters.get('x0', 0),
                              'theta': parameters.get('theta', 1.0),
                              'mu': parameters.get('mu', 0),
                              'sigma': parameters.get('sigma', 0.5)}
            elif process_type == 'vasicek':
                func_params = {'r0': parameters.get('r0', 0.05),
                              'k': parameters.get('k', 0.1),
                              'theta': parameters.get('theta', 0.05),
                              'sigma': parameters.get('sigma', 0.02)}
            elif process_type == 'cir':
                func_params = {'r0': parameters.get('r0', 0.05),
                              'k': parameters.get('k', 0.1),
                              'theta': parameters.get('theta', 0.05),
                              'sigma': parameters.get('sigma', 0.02)}
            elif process_type == 'brownian':
                func_params = {}
            elif process_type == 'exponential_martingale':
                func_params = {'mu': parameters.get('mu', 0),
                              'sigma': parameters.get('sigma', 0.2)}
            elif process_type == 'poisson':
                func_params = {'lambd': parameters.get('lambd', 1.0)}
            elif process_type == 'jump_diffusion':
                func_params = {'S0': parameters.get('S0', 100),
                              'mu': parameters.get('mu', 0.1),
                              'sigma': parameters.get('sigma', 0.2),
                              'lambd': parameters.get('lambd', 0.5),
                              'jump_mean': parameters.get('jump_mean', 0),
                              'jump_std': parameters.get('jump_std', 0.1)}

            func_params.update(common_params)
            t, paths = process_map[process_type](**func_params)

            return {
                'success': True,
                'time': t.tolist(),
                'paths': paths.tolist(),
                'colors': sp.pastel_colors[:num_trajectories]
            }
        except Exception as e:
            return {'success': False, 'error': str(e)}
    else:
        return {'success': False, 'error': f"Unknown process type: {process_type}"}

@app.get("/")
async def root():
    return {"message": "SDE Symbolic Solver API", "version": "1.0.0"}

@app.post("/solve", response_model=SolutionResponse)
async def solve_sde(problem: SDEProblem):
    try:
        solver = SDESolver()

        # Parse the problem
        result = solver.solve(
            equation_type=problem.equation_type,
            drift=problem.drift,
            diffusion=problem.diffusion,
            initial_condition=problem.initial_condition,
            variables=problem.variables or {},
            parameters=problem.parameters or {}
        )

        return SolutionResponse(
            steps=result["steps"],
            final_solution=result["final_solution"],
            solution_type=result["solution_type"],
            metadata=result["metadata"]
        )

    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error solving SDE: {str(e)}")

@app.post("/simulate")
async def simulate_sde(sim_request: SimulationRequest):
    try:
        # If it's a predefined process, use the extended class
        if sim_request.process_type != "custom":
            result = generate_stochastic_process(
                sim_request.process_type,
                sim_request.parameters,
                sim_request.time_span,
                sim_request.num_points,
                sim_request.num_trajectories
            )

            if not result['success']:
                raise HTTPException(status_code=400, detail=result['error'])

            # Generate the plot with colors
            plot_data = generate_plot_with_colors(
                result['time'],
                result['paths'],
                result['colors'],
                f"Process {sim_request.process_type}"
            )

            return JSONResponse(content={
                "plot_data": plot_data,
                "time_points": result['time'],
                "trajectories": result['paths'],
                "colors": result['colors']
            })
        else:
            # Custom simulation with Euler-Maruyama
            plot_data, time_points, trajectories = generate_simulation_plot(sim_request)

            # Generate pastel colors for trajectories
            pastel_colors = [
                '#FFB6C1', '#87CEEB', '#98FB98', '#DDA0DD', '#FFD700',
                '#F0E68C', '#E6E6FA', '#B0E0E6', '#FFA07A', '#20B2AA'
            ]
            colors = pastel_colors[:len(trajectories)]

            return JSONResponse(content={
                "plot_data": plot_data,
                "time_points": time_points,
                "trajectories": trajectories,
                "colors": colors
            })

    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error simulating SDE: {str(e)}")

def generate_plot_with_colors(time_points, trajectories, colors_list, title):
    """Generate plot with pastel colors"""
    plt.rcParams.update({
        'font.family': 'Palatino',
        'font.serif': 'Palatino',
        'mathtext.fontset': 'cm',
        'axes.edgecolor': 'black',
        'axes.linewidth': 0.8,
        'axes.spines.top': False,
        'axes.spines.right': False
    })

    fig, ax = plt.subplots(figsize=(10, 6), dpi=100)

    for i, trajectory in enumerate(trajectories):
        color = colors_list[i] if i < len(colors_list) else '#349192'
        ax.plot(time_points, trajectory, color=color, linewidth=2, alpha=0.8,
                label=f'Trajectory {i+1}')

    ax.set_xlabel('Time (t)', fontsize=12, fontname='Palatino')
    ax.set_ylabel('X(t)', fontsize=12, fontname='Palatino')
    ax.set_title(title, fontsize=14, fontname='Palatino', fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend()

    # Signature
    ax.text(0.02, 0.98, 'SDE Solver by Ramanambonona Ambinintsoa, PhD',
            transform=ax.transAxes, fontsize=10, fontname='Palatino',
            verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    buffer = io.BytesIO()
    plt.savefig(buffer, format='png', dpi=150, bbox_inches='tight',
                facecolor='white', edgecolor='black')
    buffer.seek(0)
    plot_data = base64.b64encode(buffer.getvalue()).decode()
    plt.close()

    return plot_data

def generate_simulation_plot(sim_request: SimulationRequest):
    """Generate simulation plot for custom SDE"""
    plt.rcParams.update({
        'font.family': 'Palatino',
        'font.serif': 'Palatino',
        'mathtext.fontset': 'cm',
        'axes.edgecolor': 'black',
        'axes.linewidth': 0.8,
        'axes.spines.top': False,
        'axes.spines.right': False
    })

    fig, ax = plt.subplots(figsize=(10, 6), dpi=100)

    # Generate sample data
    t = np.linspace(sim_request.time_span[0], sim_request.time_span[1], sim_request.num_points)

    trajectories = []
    pastel_colors = ['#FFB6C1', '#87CEEB', '#98FB98', '#DDA0DD', '#FFD700']

    for i in range(sim_request.num_trajectories):
        # Simple geometric Brownian motion simulation as example
        mu = sim_request.parameters.get('mu', 0.1)
        sigma = sim_request.parameters.get('sigma', 0.2)
        x0 = float(sim_request.initial_condition) if sim_request.initial_condition else 1.0

        # Euler-Maruyama simulation
        dt = t[1] - t[0]
        x = np.zeros_like(t)
        x[0] = x0

        for j in range(1, len(t)):
            dW = np.random.normal(0, np.sqrt(dt))
            x[j] = x[j-1] + mu * x[j-1] * dt + sigma * x[j-1] * dW

        trajectories.append(x.tolist())
        color = pastel_colors[i % len(pastel_colors)]
        ax.plot(t, x, color=color, linewidth=2, alpha=0.8, label=f'Trajectory {i+1}')

    # Customize plot
    ax.set_xlabel('Time (t)', fontsize=12, fontname='Palatino')
    ax.set_ylabel('X(t)', fontsize=12, fontname='Palatino')
    ax.set_title('SDE Simulation', fontsize=14, fontname='Palatino', fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend()

    # Add signature
    ax.text(0.02, 0.98, 'SDE Solver by Ramanambonona Ambinintsoa, PhD',
            transform=ax.transAxes, fontsize=10, fontname='Palatino',
            verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    # Save to base64
    buffer = io.BytesIO()
    plt.savefig(buffer, format='png', dpi=150, bbox_inches='tight',
                facecolor='white', edgecolor='black')
    buffer.seek(0)
    plot_data = base64.b64encode(buffer.getvalue()).decode()
    plt.close()

    return plot_data, t.tolist(), trajectories

class DownloadRequest(BaseModel):
    steps: List[Dict[str, str]]
    final_solution: str
    equation_type: str
    drift: str
    diffusion: str
    parameters: Dict[str, float]

@app.post("/download-pdf")
async def download_pdf(download_data: DownloadRequest):
    """Generate a PDF with the results"""
    try:
        # Create a temporary file
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.pdf')
        pdf_path = temp_file.name

        # Create the PDF
        doc = SimpleDocTemplate(pdf_path, pagesize=letter)
        styles = getSampleStyleSheet()

        # Custom style
        title_style = ParagraphStyle(
            'CustomTitle',
            parent=styles['Heading1'],
            fontSize=16,
            textColor=colors.HexColor('#349192'),
            spaceAfter=12,
        )

        content = []

        # Title
        content.append(Paragraph("Stochastic Differential Equation Resolution", title_style))
        content.append(Spacer(1, 12))

        # Equation
        eq_style = ParagraphStyle(
            'EquationStyle',
            parent=styles['Normal'],
            fontSize=12,
            textColor=colors.black,
            backColor=colors.HexColor('#f0fdfa'),
            borderPadding=10,
            borderColor=colors.HexColor('#349192'),
            borderWidth=1,
        )

        equation_text = f"Equation: dXₜ = {download_data.drift} dt + {download_data.diffusion} dWₜ"
        content.append(Paragraph(equation_text, eq_style))
        content.append(Spacer(1, 12))

        # Parameters
        params_text = "Parameters: " + ", ".join([f"{k} = {v}" for k, v in download_data.parameters.items()])
        content.append(Paragraph(params_text, styles['Normal']))
        content.append(Spacer(1, 12))

        # Solution Steps
        content.append(Paragraph("Resolution Steps:", styles['Heading2']))
        content.append(Spacer(1, 8))

        for i, step in enumerate(download_data.steps, 1):
            step_title = f"Step {i}: {step['title']}"
            content.append(Paragraph(step_title, styles['Heading3']))
            # Clean up LaTeX content
            clean_content = step['content'].replace('$', '').replace('\\[', '').replace('\\]', '')
            content.append(Paragraph(clean_content, styles['Normal']))
            content.append(Spacer(1, 6))

        # Final Solution
        content.append(Paragraph("Final Solution:", styles['Heading2']))
        content.append(Spacer(1, 8))
        final_solution_clean = download_data.final_solution.replace('$', '').replace('\\[', '').replace('\\]', '')
        content.append(Paragraph(final_solution_clean, eq_style))

        # Signature
        content.append(Spacer(1, 20))
        content.append(Paragraph("Generated by SDE Symbolic Solver", styles['Italic']))
        content.append(Paragraph("Ramanambonona Ambinintsoa, PhD", styles['Italic']))

        doc.build(content)

        return FileResponse(pdf_path, media_type='application/pdf',
                            filename=f"sde_solution_{uuid.uuid4().hex[:8]}.pdf")

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating PDF: {str(e)}")

@app.post("/download-latex")
async def download_latex(download_data: DownloadRequest):
    """Generate a LaTeX file with the results"""
    try:
        latex_content = generate_latex_content(download_data)

        # Create a temporary file
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.tex')
        tex_path = temp_file.name

        with open(tex_path, 'w', encoding='utf-8') as f:
            f.write(latex_content)

        return FileResponse(tex_path, media_type='application/x-tex',
                            filename=f"sde_solution_{uuid.uuid4().hex[:8]}.tex")

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating LaTeX: {str(e)}")

def generate_latex_content(data):
    """Generate the LaTeX content"""
    # L'erreur venait des doubles backslashes qui n'ont pas lieu d'être dans cette partie
    # du code si on utilise le f-string
    return f"""\\documentclass{{article}}
\\usepackage[utf8]{{inputenc}}
\\usepackage{{amsmath}}
\\usepackage{{amssymb}}
\\usepackage{{xcolor}}
\\usepackage{{geometry}}

\\geometry{{a4paper, margin=1in}}

\\title{{Stochastic Differential Equation Solver}}
\\author{{Ramanambonona Ambinintsoa, PhD}}
\\date{{\\today}}

\\begin{{document}}

\\maketitle

\\section*{{Equation}}
\\begin{{flushleft}}
Stochastic Differential Equation of type {data.equation_type}:\\
$dX_t = {data.drift}  dt + {data.diffusion}  dW_t$
\\end{{flushleft}}

\\section*{{Parameters}}
\\begin{{flushleft}}
{', '.join([f'${k} = {v}$' for k, v in data.parameters.items()])}
\\end{{flushleft}}

\\section*{{Resolution Steps}}
{chr(10).join([f'\\subsection*{{Step {i}: {step["title"]}}}' + chr(10) + f'{step["content"].replace("$", "")}' for i, step in enumerate(data.steps, 1)])}

\\section*{{Final Solution}}
\\begin{{equation}}
{data.final_solution.replace("$", "")}
\\end{{equation}}

\\vspace{{2cm}}
\\begin{{flushright}}
\\textit{{Generated by SDE Symbolic Solver}}\\
\\textit{{Ramanambonona Ambinintsoa, PhD}}
\\end{{flushright}}

\\end{{document}}"""
    
@app.get("/health")
async def health_check():
    return {"status": "healthy"}

# Additional endpoints for examples
@app.get("/examples")
async def get_examples():
    examples = {
        "geometric_brownian": {
            "name": "Geometric Brownian Motion",
            "drift": "mu*x",
            "diffusion": "sigma*x",
            "description": "Used in financial mathematics for stock prices",
            "parameters": {"mu": 0.1, "sigma": 0.2, "S0": 100},
            "process_type": "geometric_brownian"
        },
        "ornstein_uhlenbeck": {
            "name": "Ornstein-Uhlenbeck Process",
            "drift": "theta*(mu - x)",
            "diffusion": "sigma",
            "description": "Mean-reverting process used in interest rates modeling",
            "parameters": {"theta": 1.0, "mu": 0.0, "sigma": 0.5, "x0": 0},
            "process_type": "ornstein_uhlenbeck"
        },
        "vasicek_model": {
            "name": "Vasicek Model",
            "drift": "a*(b - x)",
            "diffusion": "sigma",
            "description": "Interest rate model with mean reversion",
            "parameters": {"a": 0.1, "b": 0.05, "sigma": 0.02, "r0": 0.05},
            "process_type": "vasicek"
        },
        "cir_model": {
            "name": "CIR Model",
            "drift": "a*(b - x)",
            "diffusion": "sigma*sqrt(x)",
            "description": "Cox-Ingersoll-Ross interest rate model",
            "parameters": {"a": 0.1, "b": 0.05, "sigma": 0.1, "r0": 0.05},
            "process_type": "cir"
        },
        "brownian_motion": {
            "name": "Standard Brownian Motion",
            "drift": "0",
            "diffusion": "1",
            "description": "Standard Wiener process",
            "parameters": {},
            "process_type": "brownian"
        }
    }
    return examples

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
