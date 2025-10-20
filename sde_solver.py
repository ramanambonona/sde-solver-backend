import sympy as sp
from sympy import symbols, Function, exp, integrate, diff, sqrt, pi, oo, Eq, solve
from typing import Dict, List, Any, Optional
import re

class SDESolver:
    """
    Symbolic solver for Stochastic Differential Equations (SDEs)
    Based on the work of Chu-Ching Huang (2011)
    Extended with additional solution methods
    """
    
    def __init__(self):
        self.steps = []
    
    def add_step(self, title: str, content: str):
        """Add a step to the solution process"""
        cleaned_content = self.clean_latex_content(content)
        self.steps.append({"title": title, "content": cleaned_content})
    
    def clean_latex_content(self, content: str) -> str:
        """Clean excessive $ in LaTeX content"""
        if not content:
            return content
            
        cleaned = re.sub(r'\$\s*\\\s*\$', '', content)
        cleaned = re.sub(r'\$\s*\$', '', cleaned)
        cleaned = re.sub(r'\\\[\s*\\\]', '', cleaned)
        cleaned = re.sub(r'^\$|\$$', '', cleaned)
        cleaned = re.sub(r'^\\\[|\\\]$', '', cleaned)
        return cleaned.strip()
    
    def parse_expression(self, expr_str: str, variables: Dict[str, str]) -> sp.Expr:
        """Parse string expression to sympy expression"""
        sym_map = {}
        for var_name, var_type in variables.items():
            if var_type == "variable":
                sym_map[var_name] = symbols(var_name)
            elif var_type == "function":
                sym_map[var_name] = Function(var_name)(symbols('t'))
        
        common_symbols = {
            't': symbols('t'), 'x': symbols('x'), 'y': symbols('y'), 'z': symbols('z'),
            'w': symbols('w'), 'W': symbols('W'), 'dt': symbols('dt'), 'dw': symbols('dw'),
            'dW': symbols('dW'), 'alpha': symbols('alpha'), 'beta': symbols('beta'),
            'sigma': symbols('sigma'), 'mu': symbols('mu'), 'theta': symbols('theta'),
            'a': symbols('a'), 'b': symbols('b'), 'c': symbols('c'), 'd': symbols('d')
        }
        sym_map.update(common_symbols)
        
        try:
            return sp.sympify(expr_str, locals=sym_map)
        except:
            return self._build_expression_manual(expr_str, sym_map)
    
    def _build_expression_manual(self, expr_str: str, sym_map: Dict) -> sp.Expr:
        """Manual expression building for complex cases"""
        try:
            return eval(expr_str, {"__builtins__": None, "sqrt": sqrt, "exp": exp}, sym_map)
        except:
            return sp.sympify(expr_str)
    
    def solve_separable_sde(self, drift: sp.Expr, diffusion: sp.Expr, t0: float = 0, X0: float = 0) -> Dict:
        """Solve SDE using separation of variables method"""
        t, x, w = symbols('t x w')
        
        self.add_step("Separation of Variables Method", "For SDE: $dX_t = \\mu(t,X)dt + \\sigma(t,X)dW_t$")
        
        try:
            if str(drift) == str(-2*x/(1+t)) and str(diffusion) == str(sqrt(t*(1-t))):
                self.add_step("Specific Case Detected", "Solving: $dX_t = -\\frac{2X_t}{1+t}dt + \\sqrt{t(1-t)}dW_t$")
                
                integrating_factor = exp(integrate(2/(1+t), t))
                self.add_step("Integrating Factor", f"Integrating factor: $F(t) = {sp.latex(integrating_factor)}$")
                
                stochastic_integral = integrate(integrating_factor * diffusion, w)
                solution = (X0 * integrating_factor.subs(t, t0) + stochastic_integral) / integrating_factor
                
                self.add_step("Solution via Integrating Factor", f"$X_t = \\frac{{X_0 F(t_0) + \\int F(s)\\sigma(s)dW_s}}{{F(t)}}$")
                
                return {
                    "solution": solution,
                    "method": "integrating_factor",
                    "steps": self.steps
                }
        except Exception as e:
            self.add_step("Separation Method Failed", f"Could not apply separation method: {str(e)}")
        
        return self.solve_reducible_nonlinear(drift, diffusion, t0, X0)
    
    def solve_linear_sde(self, drift: sp.Expr, diffusion: sp.Expr, t0: float = 0, X0: float = 0) -> Dict:
        """Solve linear SDE using the method from Huang's paper"""
        t, s, w, x = symbols('t s w x')
        
        a = diff(drift, x) if x in drift.free_symbols else 0
        b = drift.subs(x, 0) if x in drift.free_symbols else drift
        c = diff(diffusion, x) if x in diffusion.free_symbols else 0
        d = diffusion.subs(x, 0) if x in diffusion.free_symbols else diffusion
        
        self.add_step("Identify Coefficients", f"Drift: $a(t) = {sp.latex(a)}$, $b(t) = {sp.latex(b)}$\n" f"Diffusion: $c(t) = {sp.latex(c)}$, $d(t) = {sp.latex(d)}$")
        
        integrand_a = a - c**2/2
        integral_a = integrate(integrand_a, (s, t0, t))
        integral_c = integrate(c, (w, t0, t))
        
        Phi = sp.exp(integral_a + integral_c)
        Phi_inv = 1/Phi
        
        self.add_step("Integration Factor",f"$\\Phi_{{t_0,t}} = \\exp\\left(\\int_{{t_0}}^t \\left(a(s) - \\frac{{1}}{{2}}c(s)^2\\right)ds + \\int_{{t_0}}^t c(s)dW_s\\right)$")
        
        integrand1 = (b - c*d) * Phi_inv
        I1 = integrate(integrand1, (s, t0, t))
        
        integrand2 = d * Phi_inv
        I2 = integrate(integrand2, (w, t0, t))
        
        solution = Phi * (X0 + I1 + I2)
        
        self.add_step("Solution Components", f"$X_t = \\Phi_{{t_0,t}} \\left( X_0 + \\int_{{t_0}}^t (b(s) - c(s)d(s))\\Phi_{{t_0,s}}^{{-1}} ds + \\int_{{t_0}}^t d(s)\\Phi_{{t_0,s}}^{{-1}} dW_s \\right)$")
        
        return {
            "solution": solution,
            "integration_factor": Phi,
            "steps": self.steps
        }
    
    def solve_kolmogorov_forward(self, drift: sp.Expr, diffusion: sp.Expr) -> Dict:
        """Solve stationary Kolmogorov forward equation using Wright's formula"""
        t, x = symbols('t x')
        
        self.add_step("Kolmogorov Forward Equation", "Stationary solution of Kolmogorov forward equation:\n" "$\\frac{\\partial p}{\\partial t} = -\\frac{\\partial}{\\partial x}[\\mu(x)p] + \\frac{1}{2}\\frac{\\partial^2}{\\partial x^2}[\\sigma^2(x)p]$")
        
        self.add_step("Stationary Solution", "For stationary solution $\\frac{\\partial p}{\\partial t} = 0$, we get:\n" "$p(x) = \\frac{C}{\\sigma^2(x)} \\exp\\left(2\\int^x \\frac{\\mu(s)}{\\sigma^2(s)} ds\\right)$")
        
        try:
            exponent_integral = 2 * integrate(drift / (diffusion**2), x)
            self.add_step("Exponent Integral",
                         f"$2\\int \\frac{{\\mu(x)}}{{\\sigma^2(x)}} dx = {sp.latex(exponent_integral)}$")
            
            unnormalized_density = sp.exp(exponent_integral) / (diffusion**2)
            C = symbols('C')
            stationary_density = C * unnormalized_density
            
            self.add_step("Unnormalized Density", f"$p(x) = C \\cdot \\frac{{1}}{{\\sigma^2(x)}} \\exp\\left(2\\int \\frac{{\\mu(x)}}{{\\sigma^2(x)}} dx\\right)$\n" f"$p(x) = {sp.latex(stationary_density)}$")
            
            return {
                "solution": stationary_density,
                "normalization_constant": C,
                "solution_type": "stationary_distribution",
                "steps": self.steps
            }
        except Exception as e:
            self.add_step("Integration Failed", f"Could not compute the integral: {str(e)}")
            C = symbols('C')
            general_solution = C * sp.exp(2 * integrate(drift / (diffusion**2), x)) / (diffusion**2)
            
            return {
                "solution": general_solution,
                "normalization_constant": C,
                "solution_type": "stationary_distribution_general",
                "steps": self.steps
            }
    
    def solve_reducible_nonlinear(self, drift: sp.Expr, diffusion: sp.Expr, t0: float = 0, X0: float = 0) -> Dict:
    """Solve reducible nonlinear SDE using integrating factor method"""
    t, w, x = symbols('t w x')
    
    b_expr = diffusion / x if x in diffusion.free_symbols else 0
    
    self.add_step("Identify Integrating Factor", f"Diffusion coefficient: $b(t) = {sp.latex(b_expr)}$")
    
    stochastic_integral = integrate(b_expr, w)
    deterministic_integral = integrate(b_expr**2, t) / 2
    F_t = sp.exp(-stochastic_integral + deterministic_integral)
    
    self.add_step("Integrating Factor", f"$F_t = \\exp\\left(-\\int b(s)dW_s + \\frac{{1}}{{2}}\\int b(s)^2 ds\\right)$")
    
    Y = Function('Y')(t)
    X_expr = Y / F_t
    f_expr = drift.subs(x, X_expr)
    ode_rhs = F_t * f_expr
    
    self.add_step("Transformed ODE", f"Let $Y_t = F_t X_t$, then $\\frac{{dY_t}}{{dt}} = F_t \\cdot f(t, F_t^{{-1}} Y_t)$")
    
    try:
        from sympy import dsolve
        ode_solution = dsolve(sp.Derivative(Y, t) - ode_rhs, Y)
        X_solution = ode_solution.rhs / F_t
    except:
        X_solution = X0 * sp.exp(integrate(b_expr, w) - integrate(b_expr**2, t)/2 + integrate(drift, t)
    
    # CORRECTION : Éviter les sauts de ligne dans l'appel de fonction
    solution_text = "Solution of transformed ODE gives:"
    solution_equation = f"$X_t = {sp.latex(X_solution)}$"
    self.add_step("ODE Solution", solution_text + "\n" + solution_equation)
    
    return {
        "solution": X_solution,
        "integrating_factor": F_t,
        "steps": self.steps
    }
    
    def solve(self, equation_type: str, drift: str, diffusion: str, 
              initial_condition: Optional[str] = None,
              variables: Dict[str, str] = None,
              parameters: Dict[str, float] = None) -> Dict:
        """Main solving method with enhanced capabilities"""
        self.steps = []
        
        if variables is None:
            variables = {'t': 'variable', 'x': 'variable', 'W': 'variable'}
        
        drift_expr = self.parse_expression(drift, variables)
        diffusion_expr = self.parse_expression(diffusion, variables)
        
        self.add_step("Problem Setup", f"Solving {'Itô' if equation_type == 'ito' else 'Stratonovich'} SDE:\n" f"$dX_t = {sp.latex(drift_expr)} dt + {sp.latex(diffusion_expr)} dW_t$")
        
        x, t = symbols('x t')
        is_kolmogorov = "kolmogorov" in drift.lower() or "stationary" in drift.lower()
        
        if is_kolmogorov:
            self.add_step("Equation Classification", "Kolmogorov Forward Equation detected")
            result = self.solve_kolmogorov_forward(drift_expr, diffusion_expr)
        else:
            is_linear = (diff(drift_expr, x, 2) == 0) and (diff(diffusion_expr, x, 2) == 0)
            
            if is_linear:
                self.add_step("Equation Classification", "Linear SDE detected")
                result = self.solve_linear_sde(drift_expr, diffusion_expr)
            else:
                self.add_step("Equation Classification", "Nonlinear SDE detected - trying various methods")
                result = self.solve_separable_sde(drift_expr, diffusion_expr)
        
        final_solution = result["solution"]
        if initial_condition and not is_kolmogorov:
            try:
                ic_expr = self.parse_expression(initial_condition, variables)
                t0_val = parameters.get('t0', 0) if parameters else 0
                x0_val = parameters.get('x0', 1) if parameters else 1
                
                if 'C' in [str(s) for s in final_solution.free_symbols]:
                    C = symbols('C')
                    ic_eq = final_solution.subs(t, t0_val) - x0_val
                    C_solution = solve(ic_eq, C)[0]
                    final_solution = final_solution.subs(C, C_solution)
                    self.add_step("Apply Initial Condition", f"Using $X({t0_val}) = {x0_val}$ to determine constant")
            except Exception as e:
                self.add_step("Initial Condition Application", f"Could not apply initial condition: {str(e)}")
        
        final_solution_latex = self.clean_latex_content(sp.latex(final_solution))
        
        return {
            "steps": self.steps,
            "final_solution": final_solution_latex,
            "solution_type": result.get("solution_type", "exact"),
            "metadata": {
                "equation_type": equation_type,
                "is_kolmogorov": is_kolmogorov,
                "variables_used": list(variables.keys())
            }
        }

def convert_ito_stratonovich(drift_ito: sp.Expr, diffusion: sp.Expr, variables: Dict) -> sp.Expr:
    """Convert between Ito and Stratonovich formulations"""
    x = symbols('x')
    drift_stratonovich = drift_ito - 0.5 * diffusion * diff(diffusion, x)
    return drift_stratonovich


