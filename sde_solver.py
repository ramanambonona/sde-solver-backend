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
    
    def add_step(self, title: str, content: str, is_latex: bool = True):
        """Add a step to the solution process, optional LaTeX flag"""
        if is_latex:
            cleaned_content = self.clean_latex_content(content)
            if not re.search(r'\$|\\\[{1,2}', cleaned_content):
                cleaned_content = f"\\[{cleaned_content}\\]"
        else:
            cleaned_content = content  # Plain text
        self.steps.append({"title": title, "content": cleaned_content})
    
    def clean_latex_content(self, content: str) -> str:
        """Clean excessive $ in LaTeX content"""
        if not content:
            return content
            
        # Nettoyage des séquences de $ ou \[\]
        cleaned = re.sub(r'\$\s*\$', '', content)  # Remove empty $$
        cleaned = re.sub(r'\\\[\s*\\\]', '', cleaned)
        cleaned = re.sub(r'\$\$', r'$', cleaned)  # Deduplicate to single $
        cleaned = re.sub(r'\\\$', r'$', cleaned)  # Unescape
        
        # Supprime les délimiteurs de début/fin si l'expression est entre $...$ ou \[...\]
        cleaned = re.sub(r'^\$|\$$', '', cleaned.strip())
        cleaned = re.sub(r'^\\\[|\\\]$', '', cleaned.strip())
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
            'a': symbols('a'), 'b': symbols('b'), 'c': symbols('c'), 'd': symbols('d'),
            'sqrt': sqrt, 'exp': exp, 'integrate': integrate, 'diff': diff
        }
        sym_map.update(common_symbols)
        
        try:
            return sp.sympify(expr_str, locals=sym_map)
        except:
            return self._build_expression_manual(expr_str, sym_map)
    
    def _build_expression_manual(self, expr_str: str, sym_map: Dict) -> sp.Expr:
        """Manual expression building for complex cases"""
        try:
            # Sécurité: utiliser eval pour les fonctions mathématiques communes
            allowed_globals = {"__builtins__": None, "sqrt": sqrt, "exp": exp}
            # Également permettre d'accéder aux symboles définis dans sym_map
            return eval(expr_str, allowed_globals, sym_map)
        except Exception:
            # Fallback to simple sympify if manual eval fails
            return sp.sympify(expr_str)
    
    def solve_separable_sde(self, drift: sp.Expr, diffusion: sp.Expr, t0: float = 0, X0: float = 0) -> Dict:
        """Solve SDE using separation of variables method"""
        t, x, w = symbols('t x w')
        
        self.add_step("Separation of Variables Method", "dX_t = \\mu(t,X) dt + \\sigma(t,X) dW_t")
        
        try:
            # Cas spécifique pour l'équation linéaire (intégration par facteur intégrant)
            # Cette détection est une heuristique et pourrait être améliorée.
            if x in drift.free_symbols and x in diffusion.free_symbols:
                # Vérification si l'équation est linéaire du type dX_t = (a(t)X + b(t))dt + (c(t)X + d(t))dW_t
                if diff(drift, x, 2) == 0 and diff(diffusion, x, 2) == 0:
                    self.add_step("Case Detection", "Equation is linear, using linear solver approach.", is_latex=False)
                    return self.solve_linear_sde(drift, diffusion, t0, X0)

            # Essayer de séparer les variables (pas toujours possible symboliquement)
            g_x = 1 / diffusion.subs(t, 1) if not diffusion.free_symbols.difference({x}) else 0
            f_t = drift.subs(x, 1) if not drift.free_symbols.difference({t}) else 0
            
            if g_x != 0 and f_t != 0:
                self.add_step("Separable Case", "The equation is separable: dX = f(t) dt + g(x) dW (approximately)")
                # Intégration
                int_g = integrate(g_x, x)
                int_f = integrate(f_t, t)
                solution = int_g + int_f + symbols('C')
            else:
                # Méthode plus générale pour non-séparables
                # Utiliser l'approche de transformation d'Itô
                self.add_step("Non-Separable", "Applying Itô formula for transformation")
                # Supposons une transformation Y = F(X)
                # Pour simplifier, on suppose F tel que dY = ... sans terme quadratique
                # Ceci est heuristique
                F = integrate(1/diffusion, x)
                dF_dx = diff(F, x)
                d2F_dx2 = diff(dF_dx, x)
                ito_drift = dF_dx * drift + 0.5 * d2F_dx2 * diffusion**2
                solution = F + integrate(ito_drift, t) + symbols('W_t')  # Approximation
            
            self.add_step("Integrated Form", sp.latex(solution))
            return {"solution": solution, "solution_type": "separable"}
        
        except Exception as e:
            self.add_step("Error in Separation", f"Could not separate variables: {str(e)}", is_latex=False)
            return {"solution": "No closed-form solution found", "solution_type": "none"}

    def solve_linear_sde(self, drift: sp.Expr, diffusion: sp.Expr, t0: float = 0, X0: float = 0) -> Dict:
        """Solve linear SDE using integrating factor"""
        t, x = symbols('t x')
        
        self.add_step("Linear SDE Method", "dX = (a(t)X + b(t))dt + (c(t)X + d(t))dW")
        
        try:
            # Extract coefficients (assume linear in x)
            a_t = diff(drift, x)
            b_t = drift - a_t * x
            c_t = diff(diffusion, x)
            d_t = diffusion - c_t * x
            
            self.add_step("Coefficients", f"a(t) = {sp.latex(a_t)}, b(t) = {sp.latex(b_t)}, c(t) = {sp.latex(c_t)}, d(t) = {sp.latex(d_t)}")
            
            # Integrating factor mu(t) = exp(-int a(t) dt + int c(t) dW - 1/2 int c(t)^2 dt)  (stochastic)
            int_a = integrate(a_t, t)
            int_c2 = integrate(c_t**2, t)
            mu = exp(-int_a + integrate(c_t, symbols('W')) - 0.5 * int_c2)
            
            self.add_step("Integrating Factor", sp.latex(mu))
            
            # Solution X(t) = mu^{-1} (X0 + int mu (b dt + d dW))
            inv_mu = 1 / mu
            int_term = integrate(mu * b_t, t) + integrate(mu * d_t, symbols('W'))
            solution = inv_mu * (X0 + int_term)
            
            self.add_step("General Solution", sp.latex(solution))
            
            return {"solution": solution, "solution_type": "linear"}
        
        except Exception as e:
            self.add_step("Error in Linear Solve", f"Could not solve linear SDE: {str(e)}", is_latex=False)
            return {"solution": "No closed-form solution found", "solution_type": "none"}

    def solve_kolmogorov_forward(self, drift: sp.Expr, diffusion: sp.Expr) -> Dict:
        """Solve for stationary distribution using Kolmogorov forward equation"""
        x = symbols('x')
        
        self.add_step("Kolmogorov Forward Equation", "-\\frac{d}{dx} [\\mu(x) p(x)] + \\frac{1}{2} \\frac{d^2}{dx^2} [\\sigma^2(x) p(x)] = 0")
        
        try:
            # Assume stationary: integrate the ODE for p(x)
            sigma2 = diffusion**2
            int_factor = exp(-2 * integrate(drift / sigma2, x))
            p_x = symbols('C') * int_factor / sigma2
            
            self.add_step("Stationary Density", sp.latex(p_x))
            
            # Normalize if possible (over R)
            norm_const = integrate(p_x.subs(symbols('C'), 1), (x, -oo, oo))
            if norm_const != oo:
                p_x = p_x / norm_const
                self.add_step("Normalized Density", sp.latex(p_x))
            
            return {"solution": p_x, "solution_type": "stationary_distribution"}
        
        except Exception as e:
            self.add_step("Error in KFE", f"Could not solve KFE: {str(e)}", is_latex=False)
            return {"solution": "No stationary distribution found", "solution_type": "none"}

    def solve(self, equation_type: str, drift_str: str, diffusion_str: str, 
              initial_condition: Optional[str] = None, variables: Dict[str, str] = {}, 
              parameters: Dict[str, float] = {}, process_type: str = "custom") -> Dict:
        """Main solve method with process_type support"""
        t, x = symbols('t x')
        
        drift_expr = self.parse_expression(drift_str, variables)
        diffusion_expr = self.parse_expression(diffusion_str, variables)
        
        self.add_step("Parsed Equation", f"dX_t = {sp.latex(drift_expr)} dt + {sp.latex(diffusion_expr)} dW_t")
        
        # Standardized IC parsing as per correction
        t0_val = 0.0
        x0_val = 1.0  # Default
        if initial_condition:
            ic_match = re.search(r'X\s*\(\s*(\d*\.?\d*)\s*\)\s*=\s*(\d*\.?\d*)', initial_condition)
            if ic_match:
                t0_val = float(ic_match.group(1))
                x0_val = float(ic_match.group(2))
            else:
                # Assume just x0 (common user input)
                try:
                    x0_val = float(initial_condition)
                except ValueError:
                    pass  # Keep default
        
        # Override with params if process_type known
        if process_type != "custom":
            param_map = {
                "geometric_brownian": "S0",
                "ornstein_uhlenbeck": "x0",
                "vasicek": "r0",
                "cir": "r0",
                "brownian": "x0"
            }
            key = param_map.get(process_type, "x0")
            if key in parameters:
                x0_val = parameters[key]
        
        # Détection de cas stationnaire et résolution
        is_kolmogorov = False  # or condition for stationary
        if is_kolmogorov or not initial_condition: 
            self.add_step("Equation Classification", "Kolmogorov Forward Equation (Stationary Density) approach detected or no initial condition given.", is_latex=False)
            result = self.solve_kolmogorov_forward(drift_expr, diffusion_expr)
            solution_type = "stationary_distribution"
        else:
            # Convertir Stratonovich en Itô si nécessaire (simplification: on suppose la formule Itô)
            if equation_type == 'stratonovich':
                self.add_step("Itô-Stratonovich Conversion", "\\mu_{Ito} = \\mu_{Strat} + \\frac{1}{2}\\sigma(x)\\sigma'(x)")
                drift_expr = drift_expr + 0.5 * diffusion_expr * diff(diffusion_expr, x)
                self.add_step("Itô Form", f"dX_t = {sp.latex(drift_expr)} dt + {sp.latex(diffusion_expr)} dW_t")

            # Test de linéarité
            is_linear = (diff(drift_expr, x, 2) == 0) and (diff(diffusion_expr, x, 2) == 0)
            
            if is_linear:
                self.add_step("Equation Classification", "Linear SDE detected.", is_latex=False)
                result = self.solve_linear_sde(drift_expr, diffusion_expr, t0_val, x0_val)  # Updated to pass parsed vals
                solution_type = "linear_general_formula"
            else:
                self.add_step("Equation Classification", "Nonlinear SDE detected - trying various methods.", is_latex=False)
                # On utilise solve_separable_sde qui contient des heuristiques
                result = self.solve_separable_sde(drift_expr, diffusion_expr, t0_val, x0_val)  # Updated
                solution_type = result.get("solution_type", "exact_approx")
        
        final_solution = result["solution"]
        
        # 3. Application de la Condition Initiale (pour les SDEs)
        if initial_condition and not solution_type.startswith("stationary"):
            try:
                # Symbole constant d'intégration (si présent)
                C_sym = symbols('C1') # Utilisé dans la fonction de log
                
                if C_sym in final_solution.free_symbols:
                    # Pour les cas où la constante C1 est encore là
                    ic_eq = Eq(final_solution.subs(t, t0_val), x0_val)
                    C_solution = solve(ic_eq, C_sym)
                    if C_solution:
                        final_solution = final_solution.subs(C_sym, C_solution[0])
                        self.add_step("Apply Initial Condition", f"Using X({t0_val}) = {x0_val} to determine constant C_1")
                    else:
                         self.add_step("Initial Condition Application", "Could not solve for integration constant C1.", is_latex=False)

            except Exception as e:
                self.add_step("Initial Condition Application", f"Could not apply initial condition: {str(e)}", is_latex=False)
        
        final_solution_latex = self.clean_latex_content(sp.latex(final_solution))
        
        return {
            "steps": self.steps,
            "final_solution": final_solution_latex,
            "solution_type": solution_type,
            "metadata": {
                "equation_type": equation_type,
                "is_kolmogorov": solution_type.startswith("stationary"),
                "variables_used": list(variables.keys())
            }
        }

def convert_ito_stratonovich(drift_ito: sp.Expr, diffusion: sp.Expr, variables: Dict) -> sp.Expr:
    """Convert between Ito and Stratonovich formulations (only drift)"""
    x = symbols('x')
    drift_stratonovich = drift_ito - 0.5 * diffusion * diff(diffusion, x)
    return drift_stratonovich
