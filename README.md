### Integration of Physical Properties

To implement physical constraints that the model can use during learning, the double pendulum was assumed to be a conservative system. Therefore, **energy conservation** and compliance with the **Lagrange equation** were defined as physical constraints.

#### Energy Conservation

To check whether the double pendulum respects energy conservation, the total energy of the system at time $t$ is compared with the energy of the predicted state at time $t+1$.  
The total energy $E_i$ is calculated from the kinetic energy $K$ and the potential energy $V$ of the system:

<img width="111" height="24" alt="image" src="https://github.com/user-attachments/assets/5bb1bb5f-c2c4-4e67-a06f-e1afa833bca4" />

If energy conservation holds, the energy difference between each time step should be zero, provided the data do not include dynamics with different initial conditions. Accordingly, the following term applies to the loss function, where $E_{i+1}$ is the energy of the predicted state and $E_i$ is the original energy:

<img width="277" height="32" alt="image" src="https://github.com/user-attachments/assets/087fb312-53a0-45b5-99e8-ae377525ed6a" />

### Compliance with the Lagrange Equation

For compliance with the Lagrange equation, no comparison to the previous step is required. Using the model’s predictions during training, the Lagrange equation is calculated and checked for validity.  

Since two angles are present, the Euler-Lagrange equation is applied separately to both pendulums. Thus, the physics-informed loss function of the PINN is given by:

<img width="275" height="75" alt="image" src="https://github.com/user-attachments/assets/deba7423-1b99-4d81-a812-bfd09a67199a" />


Expanded, this results in:

<img width="681" height="115" alt="image" src="https://github.com/user-attachments/assets/8d2db583-796a-49a7-bcce-2db2d0439a6e" />

A drawback of this method is the occurrence of the second derivatives $\ddot{\theta}_1$ and $\ddot{\theta}_2$, which must first be computed, since the model only predicts the first derivatives of the angles. This issue can be solved by numerical differentiation using the difference quotient, where the angular velocity of the previous state $\dot{\theta}_i(t-\Delta t)$, the predicted $\dot{\theta}_i(t)$, and the time step $\Delta t$ are used:

<img width="210" height="66" alt="image" src="https://github.com/user-attachments/assets/7c747592-edf0-4eed-99d1-56b0710facd8" />


Alternatively, the derivative can be computed using the `torch.autograd` function from the PyTorch library. However the analytic method has been used to gain a better understanding of the lossfunction.

---

## 3.2 Thermal Diffusion

To examine whether the integration of physical information can be transferred to other problems and yield similar results, a standard ML model and two different PINNs were applied to a one-dimensional diffusion problem. The goal is to test whether the use of physical information also improves the learning process here.

### 3.2.1 FDM Solution of the Heat Equation

The heat equation introduced in [11], reduced to a one-dimensional problem with $T$ as local temperature, $\alpha$ as thermal diffusivity, and $x$ as spatial coordinate, is given by:

<img width="114" height="56" alt="image" src="https://github.com/user-attachments/assets/8ab1b71a-84b9-4a4c-bf46-799b647706eb" />

For the numerical solution of this partial differential equation, the Taylor expansion is used to approximate the solution. Thus, the second derivative of the temperature with respect to the node coordinate $x$, with $\Delta x$ as the node spacing, is:

<img width="320" height="59" alt="image" src="https://github.com/user-attachments/assets/f003df4a-d249-4ce9-9e08-57fb6903219b" />

For the time derivative using the difference quotient:

<img width="139" height="55" alt="image" src="https://github.com/user-attachments/assets/25aa4458-c984-45f2-9db2-38fb59cf50ff" />

Thus, the numerical solution of the heat equation is:

<img width="367" height="59" alt="image" src="https://github.com/user-attachments/assets/e431c6e9-37d0-46cf-b1e9-85f38c5d83f2" />

Assuming an adiabatic boundary condition, the heat flux is $ \dot{q} = 0 $, since no heat can leave the system. From Fourier’s law, with $\lambda$ as the thermal conductivity:

<img width="94" height="56" alt="image" src="https://github.com/user-attachments/assets/657d7b44-9b9e-4353-a30b-a25bc022d478" />

the boundary conditions at the rod ends [11] are:

<img width="106" height="130" alt="image" src="https://github.com/user-attachments/assets/e71b4606-d3f4-4493-b169-d31b03b4823c" />

Training data was generated using the numerical solution and these boundary conditions.

### 3.2.2 Integration of Physics

As with the double pendulum, physics can be embedded by checking for compliance with energy conservation during training.

#### Heat Energy Conservation

To incorporate the boundary conditions into the heat equation, the equation is extended by $\Delta x$:

<img width="188" height="55" alt="image" src="https://github.com/user-attachments/assets/a9871347-1346-4d21-ab05-c1cce118d0d4" />

Integrating over the entire length $L$:

<img width="195" height="69" alt="image" src="https://github.com/user-attachments/assets/974d42c7-b64b-4b16-b2b6-350fcfc44b26" />

Considering the adiabatic boundary conditions yields:

<img width="130" height="66" alt="image" src="https://github.com/user-attachments/assets/7b2a3af8-cc95-477f-9293-3e9d147416a4" />

The numerical expression for the loss function, with $n$ as the number of nodes, $t$ as the total number of time steps, $\Delta x$ as the node spacing, and $T$ as the nodal temperature, is:

<img width="369" height="74" alt="image" src="https://github.com/user-attachments/assets/ff9d452b-31ce-42a0-b118-d679b73a8fc4" />

#### Compliance with the Heat Equation

Another approach is to enforce compliance with the heat equation. To integrate it into the training, the equation can be rearranged to identify deviations. The loss is thus:

<img width="598" height="61" alt="image" src="https://github.com/user-attachments/assets/23295440-562b-4a7c-b532-8ba738a38fd9" />

Here, the initial temperature is compared with the predicted temperature across the nodes and penalized if necessary. It is important to consider the boundary conditions when integrating the physical loss function, as otherwise disproportionately large losses can occur at the boundaries.

