### 3.1.2 Integration of Physical Properties

To implement physical constraints that the model can use during learning, the double pendulum was assumed to be a conservative system. Therefore, **energy conservation** and compliance with the **Lagrange equation** were defined as physical constraints.

#### Energy Conservation

To check whether the double pendulum respects energy conservation, the total energy of the system at time $t$ is compared with the energy of the predicted state at time $t+1$.  
The total energy $E_i$ is calculated from the kinetic energy $K$ and the potential energy $V$ of the system:

$$
E_{i} = K + V \tag{16}
$$

If energy conservation holds, the energy difference between each time step should be zero, provided the data do not include dynamics with different initial conditions. Accordingly, the following term applies to the loss function, where $E_{i+1}$ is the energy of the predicted state and $E_i$ is the original energy:

$$
L_{\text{Energy}} = \lvert E_{i+1} - E_i \rvert \tag{17}
$$


### Compliance with the Lagrange Equation

For compliance with the Lagrange equation, no comparison to the previous step is required. Using the model’s predictions during training, the Lagrange equation is calculated and checked for validity.  

Since two angles are present, the Euler-Lagrange equation is applied separately to both pendulums. Thus, the physics-informed loss function of the PINN is given by:

$$
L_{Lagrange} = \sum_{i=1}^{2} \left| \frac{d}{dt}\frac{\partial L}{\partial \dot{\theta}_i} - \frac{\partial L}{\partial \theta_i} \right| \tag{18}
$$

Expanded, this results in:

$$L_{Lagrange} = \left| (m_1+m_2) \ddot{\theta}_1 l_1^2 + m_2 l_1^2 \ddot{\theta}_2 l_2^2 \cos(\theta_1 - \theta_2)
+ m_2 l_1^2 \dot{\theta}_2^2 l_2^2 \sin(\theta_1 - \theta_2)
+ (m_1+m_2) g l_1 \sin(\theta_1) \right|
+ \left| \ddot{\theta}_2 l_2 + \ddot{\theta}_1 l_1^2 l_2 \cos(\theta_1 - \theta_2)
- \ddot{\theta}_1 l_1^2 l_2 \sin(\theta_1 - \theta_2) + g \sin(\theta_2) \right| \tag{19}$$

A drawback of this method is the occurrence of the second derivatives $\ddot{\theta}_1$ and $\ddot{\theta}_2$, which must first be computed, since the model only predicts the first derivatives of the angles. This issue can be solved by numerical differentiation using the difference quotient, where the angular velocity of the previous state $\dot{\theta}_i(t-\Delta t)$, the predicted $\dot{\theta}_i(t)$, and the time step $\Delta t$ are used:

$$
\ddot{\theta}_i = \frac{\dot{\theta}_i(t+\Delta t) - \dot{\theta}_i(t)}{\Delta t} \tag{20}
$$

Alternatively, the derivative can be computed using the `torch.autograd` function from the PyTorch library.

---

## 3.2 Thermal Diffusion

To examine whether the integration of physical information can be transferred to other problems and yield similar results, a standard ML model and two different PINNs were applied to a one-dimensional diffusion problem. The goal is to test whether the use of physical information also improves the learning process here.

### 3.2.1 FDM Solution of the Heat Equation

The heat equation introduced in [11], reduced to a one-dimensional problem with $T$ as local temperature, $\alpha$ as thermal diffusivity, and $x$ as spatial coordinate, is given by:

$$
\frac{\partial T}{\partial t} = \alpha \frac{\partial^2 T}{\partial x^2} \tag{21}
$$

For the numerical solution of this partial differential equation, the Taylor expansion is used to approximate the solution. Thus, the second derivative of the temperature with respect to the node coordinate $x$, with $\Delta x$ as the node spacing, is:

$$
\frac{\partial^2 T}{\partial x^2} = \frac{1}{(\Delta x)^2} \left[ T_{x+\Delta x}^t - 2 T_x^t + T_{x-\Delta x}^t \right] \tag{22}
$$

For the time derivative using the difference quotient:

$$
\frac{\partial T}{\partial t} = \frac{T_x^{t+1} - T_x^t}{\Delta t} \tag{23}
$$

Thus, the numerical solution of the heat equation is:

$$
\frac{T_x^{t+1} - T_x^t}{\Delta t} 
= \frac{\alpha}{(\Delta x)^2} \left[ T_{x+\Delta x}^t - 2 T_x^t + T_{x-\Delta x}^t \right] \tag{24}
$$

Assuming an adiabatic boundary condition, the heat flux is $ \dot{q} = 0 $, since no heat can leave the system. From Fourier’s law, with $\lambda$ as the thermal conductivity:

$$
\dot{q} = \lambda \frac{\partial T}{\partial x} \tag{25}
$$

the boundary conditions at the rod ends [11] are:

$$
\left. \frac{\partial T}{\partial x} \right|_{x=0} = 0 \tag{26}
$$

$$
\left. \frac{\partial T}{\partial x} \right|_{x=L} = 0 \tag{27}
$$

Training data was generated using the numerical solution and these boundary conditions.

### 3.2.2 Integration of Physics

As with the double pendulum, physics can be embedded by checking for compliance with energy conservation during training.

#### Heat Energy Conservation

To incorporate the boundary conditions into the heat equation, the equation is extended by $\Delta x$:

$$
\frac{\partial T}{\partial t} \cdot \Delta x 
= \alpha \frac{\partial^2 T}{\partial x^2} \cdot \Delta x \tag{28}
$$

Integrating over the entire length $L$:

$$
\int_0^L \frac{\partial T}{\partial t} \, dx 
= \left[ \alpha \frac{\partial T}{\partial x} \right]_0^L \tag{29}
$$

Considering the adiabatic boundary conditions yields:

$$
\int_0^L \frac{\partial T}{\partial t} \, dx = 0 \tag{30}
$$

The numerical expression for the loss function, with $n$ as the number of nodes, $t$ as the total number of time steps, $\Delta x$ as the node spacing, and $T$ as the nodal temperature, is:

$$
L_{Energy} = \sum_{i=0}^{t} \sum_{j=1}^{n} (T_{i,j} - T_{i+1,j}) \cdot \Delta x \tag{31}
$$

#### Compliance with the Heat Equation

Another approach is to enforce compliance with the heat equation. To integrate it into the training, the equation can be rearranged to identify deviations. The loss is thus:

$$
L_{1D-Diffusion} = \left| \frac{T_x^{t+1} - T_x^t}{\Delta t} - \frac{\alpha}{(\Delta x)^2} 
\left[ T_{x+\Delta x}^t - 2 T_x^t + T_{x-\Delta x}^t \right] \right| \tag{32}
$$

Here, the initial temperature is compared with the predicted temperature across the nodes and penalized if necessary. It is important to consider the boundary conditions when integrating the physical loss function, as otherwise disproportionately large losses can occur at the boundaries.

