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
import re  # Added for LaTeX cleaning in PDF

app = FastAPI(title="SDE Symbolic Solver API", version="1.0.0")

# CORS middleware 
origins = [
    "http://localhost:3000",
    "http://localhost:8000",
    "https://sde-solver.netlify.app",  
    "https://*.netlify.app", 
]

# URL Netlify with production
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
    process_type: Optional[str] = "custom"  # Added as per correction

class SimulationRequest(BaseModel):
    equation_type: str
    drift: str
    diffusion: str
    initial_condition: str
    parameters: Dict[str, float]
    time_span: List[float] = [0, 1]
    num_points: int = 100
    num_trajectories: int = 5
    process_type: str = "custom"  # Already present

class SolutionResponse(BaseModel):
    steps: List[Dict[str, str]]
    final_solution: str
    solution_type: str
    metadata: Dict[str, Any]

class SimulationResponse(BaseModel):
    plot_data: str  # base64 encoded image
    time_points: List[float]
    trajectories: List[List[float]]
    colors: List[str]  # Nouveau: couleurs pour chaque trajectoire

class ExtendedStochasticProcesses:
    """Classe étendue pour tous les types de processus stochastiques"""
    
    def __init__(self):
        self.pastel_colors = [
            '#FFB6C1', '#87CEEB', '#98FB98', '#DDA0DD', 
            '#FFD700', '#F0E68C', '#E6E6FA', '#B0E0E6',
            '#FFA07A', '#20B2AA', '#DEB887', '#D8BFD8'
        ]
    
    def geometric_brownian_motion(self, S0, mu, sigma, T, dt, n_paths=1):
        """Mouvement brownien géométrique"""
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
        """Processus d'Ornstein-Uhlenbeck"""
        t = np.arange(0, T+dt, dt)
        n_steps = len(t)
        
        paths = np.zeros((n_paths, n_steps))
        paths[:, 0] = x0
        
        for i in range(1, n_steps):
            dW = np.random.normal(0, np.sqrt(dt), n_paths)
            paths[:, i] = paths[:, i-1] + theta*(mu - paths[:, i-1])*dt + sigma*dW
        
        return t, paths
    
    def vasicek_model(self, r0, k, theta, sigma, T, dt, n_paths=1):
        """Modèle de Vasicek (taux d'intérêt)"""
        return self.ornstein_uhlenbeck(r0, k, theta, sigma, T, dt, n_paths)
    
    def cir_model(self, r0, k, theta, sigma, T, dt, n_paths=1):
        """Modèle CIR (Cox-Ingersoll-Ross)"""
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
        """Mouvement brownien standard (Wiener process)"""
        t = np.arange(0, T+dt, dt)
        n_steps = len(t)
        
        paths = np.zeros((n_paths, n_steps))
        
        for i in range(1, n_steps):
            dW = np.random.normal(0, np.sqrt(dt), n_paths)
            paths[:, i] = paths[:, i-1] + dW
        
        return t, paths
    
    def exponential_martingale(self, x0, mu, sigma, T, dt, n_paths=1):
        """Martingale exponentielle"""
        t = np.arange(0, T+dt, dt)
        n_steps = len(t)
        
        paths = np.zeros((n_paths, n_steps))
        paths[:, 0] = x0
        
        for i in range(1, n_steps):
            dW = np.random.normal(0, np.sqrt(dt), n_paths)
            paths[:, i] = paths[:, i-1] * np.exp(mu*dt + sigma*dW)
        
        return t, paths
    
    def poisson_process(self, lambd, T, dt, n_paths=1):
        """Processus de Poisson"""
        t = np.arange(0, T+dt, dt)
        n_steps = len(t)
        
        paths = np.zeros((n_paths, n_steps))
        
        for i in range(1, n_steps):
            jumps = np.random.poisson(lambd * dt, n_paths)
            paths[:, i] = paths[:, i-1] + jumps
        
        return t, paths
    
    def jump_diffusion(self, S0, mu, sigma, lambd, jump_mean, jump_std, T, dt, n_paths=1):
        """Processus de diffusion avec sauts (Merton model)"""
        t = np.arange(0, T+dt, dt)
        n_steps = len(t)
        
        paths = np.zeros((n_paths, n_steps))
        paths[:, 0] = S0
        
        for i in range(1, n_steps):
            dW = np.random.normal(0, np.sqrt(dt), n_paths)
            dN = np.random.poisson(lambd * dt, n_paths)
            dJ = np.random.normal(jump_mean, jump_std, n_paths) * dN
            paths[:, i] = paths[:, i-1] * np.exp(
                (mu - 0.5*sigma**2)*dt + sigma*dW + dJ
            )
        
        return t, paths

processes = ExtendedStochasticProcesses()

@app.post("/solve", response_model=SolutionResponse)
async def solve_sde(problem: SDEProblem):
    try:
        solver = SDESolver()
        result = solver.solve(
            problem.equation_type,
            problem.drift,
            problem.diffusion,
            problem.initial_condition,
            problem.variables or {},
            problem.parameters or {},
            problem.process_type
        )
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error solving SDE: {str(e)}")

@app.post("/simulate", response_model=SimulationResponse)
async def simulate_sde(request: SimulationRequest):
    try:
        # Extract parameters with defaults
        params = request.parameters
        T = request.time_span[1] - request.time_span[0]
        dt = T / (request.num_points - 1)
        
        # Get initial value
        x0 = params.get("x0", float(request.initial_condition) if request.initial_condition else 1.0)
        
        # Map process types to simulation functions
        sim_functions = {
            "geometric_brownian": lambda: processes.geometric_brownian_motion(
                params.get("S0", x0),
                params.get("mu", 0.1),
                params.get("sigma", 0.2),
                T, dt, request.num_trajectories
            ),
            "ornstein_uhlenbeck": lambda: processes.ornstein_uhlenbeck(
                params.get("x0", x0),
                params.get("theta", 1.0),
                params.get("mu", 0.0),
                params.get("sigma", 0.5),
                T, dt, request.num_trajectories
            ),
            "vasicek": lambda: processes.vasicek_model(
                params.get("r0", x0),
                params.get("a", 0.1),
                params.get("b", 0.05),
                params.get("sigma", 0.02),
                T, dt, request.num_trajectories
            ),
            "cir": lambda: processes.cir_model(
                params.get("r0", x0),
                params.get("a", 0.1),
                params.get("b", 0.05),
                params.get("sigma", 0.1),
                T, dt, request.num_trajectories
            ),
            "brownian": lambda: processes.brownian_motion(
                T, dt, request.num_trajectories
            ),
            "exponential_martingale": lambda: processes.exponential_martingale(
                params.get("x0", x0),
                params.get("mu", 0.0),
                params.get("sigma", 0.2),
                T, dt, request.num_trajectories
            ),
            "poisson": lambda: processes.poisson_process(
                params.get("lambd", 1.0),
                T, dt, request.num_trajectories
            ),
            "jump_diffusion": lambda: processes.jump_diffusion(
                params.get("S0", x0),
                params.get("mu", 0.1),
                params.get("sigma", 0.2),
                params.get("lambd", 0.5),
                params.get("jump_mean", 0.0),
                params.get("jump_std", 0.1),
                T, dt, request.num_trajectories
            ),
            "custom": lambda: processes.brownian_motion(  # Default to Brownian for custom
                T, dt, request.num_trajectories
            )
        }
        
        sim_func = sim_functions.get(request.process_type, sim_functions["custom"])
        time_points, trajectories = sim_func()
        
        # Generate plot
        fig, ax = plt.subplots(figsize=(10, 6))
        colors = processes.pastel_colors[:request.num_trajectories]
        
        for i, traj in enumerate(trajectories):
            ax.plot(time_points, traj, color=colors[i], alpha=0.8, label=f'Trajectory {i+1}')
        
        ax.set_xlabel('Time')
        ax.set_ylabel('X(t)')
        ax.set_title(f'SDE Simulation: {request.process_type.replace("_", " ").title()}')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Save to base64
        buf = io.BytesIO()
        fig.savefig(buf, format='png', dpi=300, bbox_inches='tight')
        buf.seek(0)
        plot_base64 = base64.b64encode(buf.read()).decode('utf-8')
        plt.close(fig)
        
        return SimulationResponse(
            plot_data=plot_base64,
            time_points=time_points.tolist(),
            trajectories=[traj.tolist() for traj in trajectories],
            colors=colors
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error simulating SDE: {str(e)}")

@app.post("/generate_pdf")
async def generate_pdf(data: Dict):
    try:
        buffer = io.BytesIO()
        doc = SimpleDocTemplate(buffer, pagesize=letter)
        styles = getSampleStyleSheet()
        
        story = []
        
        # Title
        story.append(Paragraph("Stochastic Differential Equation Solution", styles['Title']))
        story.append(Spacer(1, 0.2*inch))
        
        # Equation
        eq_text = f"Equation Type: {data['equation_type']}<br/>Drift: {data['drift']}<br/>Diffusion: {data['diffusion']}"
        story.append(Paragraph(eq_text, styles['Normal']))
        story.append(Spacer(1, 0.2*inch))
        
        # Parameters
        if data.get('parameters'):
            param_data = [[k, str(v)] for k, v in data['parameters'].items()]
            t = Table([['Parameter', 'Value']] + param_data)
            t.setStyle(TableStyle([
                ('BACKGROUND', (0,0), (-1,0), colors.grey),
                ('TEXTCOLOR', (0,0), (-1,0), colors.whitesmoke),
                ('ALIGN', (0,0), (-1,-1), 'CENTER'),
                ('FONTNAME', (0,0), (-1,0), 'Helvetica-Bold'),
                ('BOTTOMPADDING', (0,0), (-1,0), 12),
                ('BACKGROUND', (0,1), (-1,-1), colors.beige),
            ]))
            story.append(t)
            story.append(Spacer(1, 0.2*inch))
        
        # Steps
        story.append(Paragraph("Solution Steps", styles['Heading2']))
        for step in data['steps']:
            story.append(Paragraph(step['title'], styles['Heading3']))
            math_style = ParagraphStyle('math', parent=styles['Normal'], alignment=1)  # Center
            cleaned_content = re.sub(r'\$([^\$]+)\$', r'\\\1', step['content'])  # $math$ → \[math\]
            cleaned_content = re.sub(r'\$\$([^\$]+)\$\$', r'\\[ \1 \\]', cleaned_content)  # $$ → display
            story.append(Paragraph(cleaned_content, math_style))
            story.append(Spacer(1, 0.1*inch))
        
        # Final Solution
        story.append(Paragraph("Final Solution", styles['Heading2']))
        cleaned_final = re.sub(r'\$([^\$]+)\$', r'\\\1', data['final_solution'])
        cleaned_final = re.sub(r'\$\$([^\$]+)\$\$', r'\\[ \1 \\]', cleaned_final)
        story.append(Paragraph(cleaned_final, styles['Normal']))
        
        doc.build(story)
        buffer.seek(0)
        
        return FileResponse(
            buffer,
            media_type='application/pdf',
            filename=f"sde_solution_{uuid.uuid4().hex[:8]}.pdf"
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating PDF: {str(e)}")

@app.post("/generate_latex")
async def generate_latex(data: Dict):
    try:
        latex_content = generate_latex_content(data)
        
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.tex')
        tex_path = temp_file.name
        
        with open(tex_path, 'w', encoding='utf-8') as f:
            f.write(latex_content)
        
        return FileResponse(tex_path, media_type='application/x-tex', 
                          filename=f"sde_solution_{uuid.uuid4().hex[:8]}.tex")
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating LaTeX: {str(e)}")

def generate_latex_content(data):
    """Generate LaTeX content in English - safe version without backslash issues"""
    
    # Build steps content
    steps_content = ""
    for i, step in enumerate(data.steps, 1):
        # Clean the content by removing $ and \[ \]
        # Updated cleaning as per correction
        clean_content = re.sub(r'\$([^\$]+)\$', r'\\\1', step['content'])  # $math$ → \[math\]
        clean_content = re.sub(r'\$\$([^\$]+)\$\$', r'\\[ \1 \\]', clean_content)  # $$ → display
        steps_content += f"\\subsection*{{Step {i}: {step['title']}}}\n"
        steps_content += f"{clean_content}\n\n"
    
    # Build parameters content
    params_content = ", ".join([f"${k} = {v}$" for k, v in data.parameters.items()])
    
    # Clean final solution
    final_solution_clean = re.sub(r'\$([^\$]+)\$', r'\\\1', data.final_solution)  # Updated
    final_solution_clean = re.sub(r'\$\$([^\$]+)\$\$', r'\\[ \1 \\]', final_solution_clean)
    
    # Use string formatting with named placeholders
    latex_template = """\\documentclass{article}
\\usepackage[utf8]{inputenc}
\\usepackage{amsmath}
\\usepackage{amssymb}
\\usepackage{xcolor}
\\usepackage{geometry}

\\geometry{a4paper, margin=1in}

\\title{Stochastic Differential Equation Solution}
\\author{SDE Symbolic Solver}
\\date{\\today}

\\begin{document}

\\maketitle

\\section*{Equation}
\\begin{flushleft}
Stochastic differential equation of type {equation_type}:\\\\
$dX_t = {drift}  dt + {diffusion}  dW_t$
\\end{flushleft}

\\section*{Parameters}
\\begin{flushleft}
{parameters}
\\end{flushleft}

\\section*{Solution Steps}
{steps}

\\section*{Final Solution}
\\begin{equation}
{final_solution}
\\end{equation}

\\vspace{2cm}
\\begin{flushright}
\\textit{Generated by SDE Symbolic Solver}\\\\
\\textit{Ramanambonona Ambinintsoa, PhD}
\\end{flushright}

\\end{document}"""
    
    return latex_template.format(
        equation_type=data.equation_type,
        drift=data.drift,
        diffusion=data.diffusion,
        parameters=params_content,
        steps=steps_content,
        final_solution=final_solution_clean
    )

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
