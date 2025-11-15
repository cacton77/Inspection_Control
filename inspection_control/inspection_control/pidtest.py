import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
import control
from scipy.optimize import fsolve


def analyze_plant_controllability(m, k, c):
    """
    Analyze how plant parameters affect controllability
    """
    # Natural frequency and damping
    wn = np.sqrt(k/m)
    zeta = c/(2*np.sqrt(k*m))
    
    # Time constant (how fast the system responds naturally)
    if zeta >= 1:
        # Overdamped
        tau = 1/(zeta*wn)
    else:
        # Underdamped
        tau = 1/(zeta*wn)
    
    # Control authority metric (how much force needed per unit acceleration)
    control_authority = m
    
    # Resonance peak (amplification at natural frequency)
    if zeta < 0.707:
        resonance_peak = 1/(2*zeta*np.sqrt(1-zeta**2))
    else:
        resonance_peak = 1
    
    return {
        'wn': wn,
        'zeta': zeta,
        'tau': tau,
        'control_authority': control_authority,
        'resonance_peak': resonance_peak
    }

# Example analysis
params = analyze_plant_controllability(m=1.0, k=10.0, c=2.0)
print("Plant Analysis:")
for key, value in params.items():
    print(f"  {key}: {value:.3f}")

def optimize_plant_parameters(constraints, objectives):
    """
    Find optimal m, k, c values for best controllability
    
    constraints: dict with min/max values for m, k, c
    objectives: dict with desired specifications
    """
    
    def cost_function(x):
        m, k, c = x
        
        # Natural frequency and damping
        wn = np.sqrt(k/m)
        zeta = c/(2*np.sqrt(k*m))
        
        # Penalize deviation from ideal damping (0.7 is often good)
        damping_cost = (zeta - 0.7)**2
        
        # Penalize high mass (harder to control)
        mass_cost = m / constraints['m_max']
        
        # Penalize very high or very low natural frequency
        wn_ideal = objectives.get('wn_ideal', 5.0)
        frequency_cost = ((wn - wn_ideal) / wn_ideal)**2
        
        # Penalize high spring constant (cost/stress)
        spring_cost = k / constraints['k_max']
        
        # Total weighted cost
        weights = objectives.get('weights', {
            'damping': 1.0,
            'mass': 0.5,
            'frequency': 0.3,
            'spring': 0.2
        })
        
        total_cost = (weights['damping'] * damping_cost +
                     weights['mass'] * mass_cost +
                     weights['frequency'] * frequency_cost +
                     weights['spring'] * spring_cost)
        
        return total_cost
    
    # Initial guess
    x0 = [constraints['m_max']/2, 
          constraints['k_max']/2, 
          constraints['c_max']/2]
    
    # Bounds
    bounds = [(constraints['m_min'], constraints['m_max']),
              (constraints['k_min'], constraints['k_max']),
              (constraints['c_min'], constraints['c_max'])]
    
    # Optimize
    result = minimize(cost_function, x0, bounds=bounds, method='L-BFGS-B')
    
    return result.x

# Example optimization
constraints = {
    'm_min': 0.1, 'm_max': 10.0,  # kg
    'k_min': 1.0, 'k_max': 100.0,  # N/m
    'c_min': 0.1, 'c_max': 20.0   # N·s/m
}

objectives = {
    'wn_ideal': 5.0,  # Desired natural frequency
    'weights': {
        'damping': 2.0,    # Most important
        'mass': 1.0,       # Important
        'frequency': 0.5,  # Less important
        'spring': 0.3      # Least important
    }
}

m_opt, k_opt, c_opt = optimize_plant_parameters(constraints, objectives)
print(f"\nOptimized plant: m={m_opt:.3f}, k={k_opt:.3f}, c={c_opt:.3f}")


I = 10.0  # Moment of inertia (kg·m²)
k = 20.0  # spring constant (N/m)
c = 1.0  # damping coefficient (N·s/m)
I = m_opt
k = k_opt
c = c_opt
#Kp = 350.0  # Proportional gain
#Ki = 300.0  # Integral gain
#Kd = 50.0   # Derivative gain

# Natural Frequency and Damping Ratio
wn = (k/I)**0.5  # Natural frequency (rad/s)
zeta = c/(2*wn*I)  # Damping ratio

# Calculate PID gains for well-damped response


# Plant transfer function: G(s) = 1/(Is² + cs + k)
G = control.TransferFunction([1], [I, c, k]) 


# PID controller: C(s) = Kp + Ki/s + Kd*s
#C = control.TransferFunction([Kd, Kp, Ki], [1, 0])

# Closed-loop transfer function: T(s) = GC/(1 + GC)
#T = control.feedback(G * C)

# Open-loop with PID
#L = G * C 


# --- Root locus of plant G(s) alone ---
plt.figure(figsize=(8, 6))
control.rlocus(G, plot=True, grid=True)
plt.title('Root Locus of Plant G(s)')
plt.xlabel('Real Axis (s⁻¹)')
plt.ylabel('Imaginary Axis (s⁻¹)')
plt.axvline(0, color='k', linewidth=0.8)  # imaginary axis
plt.grid(True)
plt.show()

# Poles and Zeros
#plt.figure(figsize=(10, 8))
#poles, zeros = control.pzmap(T, plot=True, grid=True)
#plt.title('Pole-Zero Map of Closed-Loop System')
#plt.xlabel('Real Axis (seconds⁻¹)')
#plt.ylabel('Imaginary Axis (seconds⁻¹)')
#plt.show()

# Simulate impulse response
#t, y = control.impulse_response(T)

# figure for x-axis response
#plt.figure()
#plt.plot(t, y, label='Angular Displacement (rad)')
#plt.title('Closed-Loop Impulse Response (X-axis)')
#plt.xlabel('Time (s)')
#plt.ylabel('Displacement (rad)')
#plt.grid()
#plt.legend()
#plt.show()

import numpy as np
import control
import matplotlib.pyplot as plt

def place_poles_from_specs(ts_desired, Mp_percent, criterion='2%'):
    """
    Calculate pole locations from desired specifications
    
    Parameters:
    ts_desired: desired settling time [s]
    Mp_percent: desired maximum overshoot [%]
    criterion: '2%' or '5%' settling time criterion
    """
    
    # Calculate damping ratio from overshoot
    if Mp_percent == 0:
        zeta = 1.0  # Critically damped
    else:
        Mp = Mp_percent / 100
        zeta = np.sqrt(np.log(Mp)**2 / (np.pi**2 + np.log(Mp)**2))
    
    # Calculate natural frequency from settling time
    if criterion == '2%':
        wn = 4 / (zeta * ts_desired)
    else:  # 5% criterion
        wn = 3 / (zeta * ts_desired)
    
    # Calculate pole locations
    sigma = -zeta * wn  # Real part
    wd = wn * np.sqrt(1 - zeta**2)  # Imaginary part (damped frequency)
    
    # Dominant poles (complex conjugate pair)
    pole1 = sigma + 1j * wd
    pole2 = sigma - 1j * wd
    
    return pole1, pole2, zeta, wn

# Example: Design for specific requirements
ts_desired = 1.0   # 1 second settling time
Mp_desired = 10    # 10% overshoot

# Get dominant pole locations
p1, p2, zeta, wn = place_poles_from_specs(ts_desired, Mp_desired)

print(f"Desired Specifications:")
print(f"  Settling time: {ts_desired} s")
print(f"  Overshoot: {Mp_desired}%")
print(f"\nCalculated Parameters:")
print(f"  Damping ratio ζ: {zeta:.3f}")
print(f"  Natural frequency ωn: {wn:.3f} rad/s")
print(f"  Dominant poles: {p1:.3f}, {p2:.3f}")

# equations to solve for [Kd, Kp, Ki, p3]
def equations(vars):
    Kd, Kp, Ki, p3 = vars

    # characteristic polynomial from PID + plant
    a2 = (c + Kd)/I
    a1 = (k + Kp)/I
    a0 = Ki/I

    # desired polynomial coefficients
    A2 = -(p1 + p2 + p3)
    A1 = (p1*p2 + p1*p3 + p2*p3)
    A0 = -(p1*p2*p3)

    # real equations only
    eq1 = a2 - A2.real
    eq2 = a1 - A1.real
    eq3 = a0 - A0.real
    eq4 = A2.imag  # imaginary part must be zero → ensures p3 is real

    return [eq1, eq2, eq3, eq4]

# initial guess
guess = [5, 50, 50, -20]
Kd, Kp, Ki, p3 = fsolve(equations, guess)

print("\nAccurate PID (no assumed pole):")
print("  Kd =", Kd)
print("  Kp =", Kp)
print("  Ki =", Ki)
print("  solved p3 =", p3)

# verify by computing closed-loop poles
s = control.TransferFunction.s
G = 1/(I*s**2 + c*s + k)
C = Kd*s + Kp + Ki/s
T = control.feedback(C*G)

#def design_pid_from_specs(m, k, c, ts_desired, Mp_percent, third_pole_factor=5):
  #  """
 #   Design PID controller from desired specifications
    
 #   Parameters:
  #  m, k, c: plant parameters
  #  ts_desired: desired settling time
  #  Mp_percent: desired overshoot percentage
  #  third_pole_factor: how much farther left to place the third pole
 #   """
    
    # Get dominant poles from specifications
   # p1, p2, zeta, wn = place_poles_from_specs(ts_desired, Mp_percent)
    
    # Place third pole (from integral action)
    # Should be far enough left not to affect transient response
   # p3 = third_pole_factor * p1.real  # 5-10x farther left
    
    # All three closed-loop poles
    #desired_poles = [p1, p2, p3]
    
    # Form desired characteristic polynomial
    # (s-p1)(s-p2)(s-p3) = s³ + a₂s² + a₁s + a₀
   # poly_coeffs = np.poly(desired_poles)
   # a2 = poly_coeffs[1].real
   # a1 = poly_coeffs[2].real
  #  a0 = poly_coeffs[3].real
    
    # Match with closed-loop characteristic equation
    # ms³ + (c+Kd)s² + (k+Kp)s + Ki = ms³ + ma₂s² + ma₁s + ma₀
    
  #  Kd = m * a2 - c
  #  Kp = m * a1 - k
  #  Ki = m * a0
    
  #  return Kp, Ki, Kd, desired_poles


# Design controller
#Kp, Ki, Kd, poles = design_pid_from_specs(I, k, c, 
                                        #  ts_desired=1.0, 
                                        #  Mp_percent=10)

print(f"\nPID Gains:")
print(f"  Kp = {Kp:.3f}")
print(f"  Ki = {Ki:.3f}")
print(f"  Kd = {Kd:.3f}")

# PID controller: C(s) = Kp + Ki/s + Kd*s
#C = control.TransferFunction([Kd, Kp, Ki], [1, 0])

# Closed-loop transfer function: T(s) = GC/(1 + GC)
#T = control.feedback(G * C)

# Open-loop with PID
L = G * C 

# --- Root locus of PID-compensated open-loop L(s) = G(s)C(s) ---
plt.figure(figsize=(8, 6))
control.rlocus(L, plot=True, grid=True)
plt.title('Root Locus of PID-Compensated Open-Loop L(s) = G(s)C(s)')
plt.xlabel('Real Axis (s⁻¹)')
plt.ylabel('Imaginary Axis (s⁻¹)')
plt.axvline(0, color='k', linewidth=0.8)
plt.grid(True)
plt.show()

# Poles and Zeros
plt.figure(figsize=(10, 8))
poles, zeros = control.pzmap(T, plot=True, grid=True)
plt.title('Pole-Zero Map of Closed-Loop System')
plt.xlabel('Real Axis (seconds⁻¹)')
plt.ylabel('Imaginary Axis (seconds⁻¹)')
plt.show()

# Simulate step response
t, y = control.step_response(T,T=np.linspace(0, 2*ts_desired, 1000))

# figure for x-axis response
plt.figure()
plt.plot(t, y, label='Angular Displacement (rad)')
plt.title('Closed-Loop Step Response (X-axis)')
plt.xlabel('Time (s)')
plt.ylabel('Displacement (rad)')
plt.grid()
plt.legend()
plt.show()

# Print performance summary
info = control.step_info(T)
print(f"\nActual Performance:")
print(f"  Settling time: {info['SettlingTime']:.3f} s")
print(f"  Overshoot: {info['Overshoot']:.1f}%")
print(f"  Rise time: {info['RiseTime']:.3f} s")

# -------------------------------------------------------------
#                   PD CONTROLLER DESIGN
# -------------------------------------------------------------

print("\n================ PD Controller Design ================")

# Desired specs (reuse p1, p2 already computed above)
# Desired closed-loop polynomial (2nd order)
desired_poly_PD = np.poly([p1, p2])
a1_PD = desired_poly_PD[1].real   # coefficient of s
a0_PD = desired_poly_PD[2].real   # constant term

# Solve for PD gains
Kd_PD = I*a1_PD - c
Kp_PD = I*a0_PD - k

print(f"PD gains from exact pole placement:")
print(f"  Kd_PD = {Kd_PD:.6f}")
print(f"  Kp_PD = {Kp_PD:.6f}")

# Build PD controller C(s) = Kd*s + Kp
C_PD = Kd_PD*s + Kp_PD

# Closed-loop system with PD
T_PD = control.feedback(C_PD * G, 1)
L_PD = G * C_PD       
# Poles and Zeros
plt.figure(figsize=(10, 8))
poles, zeros = control.pzmap(T_PD, plot=True, grid=True)
plt.title('Pole-Zero Map of Closed-Loop System')
plt.xlabel('Real Axis (seconds⁻¹)')
plt.ylabel('Imaginary Axis (seconds⁻¹)')
plt.show()


# Closed-loop poles
print("PD Closed-loop poles:", control.poles(T_PD))

# Step response
t_PD, y_PD = control.step_response(T_PD, T=np.linspace(0, 2*ts_desired, 1000))

plt.figure()
plt.plot(t_PD, y_PD, label="PD Step Response")
plt.title("Closed-Loop Step Response with PD Control")
plt.xlabel("Time (s)")
plt.ylabel("Displacement (rad)")
plt.grid(True)
plt.legend()
plt.show()

# PD performance
info_PD = control.step_info(T_PD)
print("\nPD Controller Performance:")
print(f"  Settling time: {info_PD['SettlingTime']:.4f} s")
print(f"  Overshoot: {info_PD['Overshoot']:.2f}%")
print(f"  Rise time: {info_PD['RiseTime']:.4f} s")
print(f"  Peak time: {info_PD['PeakTime']:.4f} s")
