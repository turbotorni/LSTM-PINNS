****
*Note: The original report was written in German. I translated the most crucial part into English to provide a better understanding of the code. However, the plot titles and labels remain in German. I aimed to make the plots easy to understand despite this.
# 3. Model Systems

In the following, we derive the approaches for physics-informed loss functions using a double pendulum (Figure 6a) and one-dimensional thermal diffusion (Figure 6b). These approaches are later tested and compared with classical ML models in **Chapter 5: Evaluation and Interpretation**.

<img width="813" height="398" alt="image" src="https://github.com/user-attachments/assets/293e373e-7a31-49ca-8641-761110d4a0e8" />

**Figure 6**: Introduction of the model systems  

Both systems enable a targeted investigation of physics-informed loss functions and serve as benchmarks for evaluating machine learning approaches in physical simulations.  
The double pendulum is an example of a nonlinear, chaotic system, as it reacts sensitively to changes in initial conditions, making its motion difficult to predict.  
In contrast, one-dimensional thermal diffusion is described by a linear partial differential equation, which is an advantage for ML models since linear systems are generally easier to learn and predict [10, 11].

## 3.1 The Double Pendulum

For comparing neural networks with physics-informed networks, we used the ML model created by Dennis Gannon for simulating a double pendulum as a **benchmark** [12]. Gannon's paper highlights that while LSTM networks perform well for simple systems like predicting a projectile's trajectory, they quickly reach their limits in more complex applications like the double pendulum, which exhibits chaotic behavior, allowing for precise prediction only for a few seconds [10].

---

### 3.1.1 Dynamics of a Double Pendulum

The dynamics of a double pendulum, as derived in [22], can be analytically determined using the **Lagrange equation**. This yields two solutions: one for the first pendulum and one for the second.

Let $m_1$ and $m_2$ be the **masses**, $l_1$ and $l_2$ the **lengths** of the pendulum segments, $\theta_1$ and $\theta_2$ the **deflection angles** from the vertical axis, and $\dot{\theta}_1$ and $\dot{\theta}_2$ the **angular velocities** of the individual pendulums.

The **kinetic energy** $K$ for the double pendulum is generally given by:

<img width="575" height="63" alt="image" src="https://github.com/user-attachments/assets/b03a7fe6-8302-4db8-98b9-80b98a0cb1b4" />

And for the **potential energy** $V$:

<img width="465" height="30" alt="image" src="https://github.com/user-attachments/assets/e18972e5-4b01-4681-bde7-d952f40cb8f8" />

Using the terms for kinetic energy $K$ and potential energy $V$, the **Lagrangian function** $L$ can be derived:

<img width="114" height="28" alt="image" src="https://github.com/user-attachments/assets/c93abf5f-f1cd-41fc-b69a-b67327fa8713" />

Substituting the previously mentioned variables, this leads to:

<img width="550" height="97" alt="image" src="https://github.com/user-attachments/assets/d106c99c-f8bd-40ba-8910-ec95b9729a87" />

By applying this function to the **Euler-Lagrange equation**:

<img width="175" height="65" alt="image" src="https://github.com/user-attachments/assets/767ef3b6-02fd-49f5-ba49-20ea78c72db4" />

where $\frac{d}{dt}$ is the **time derivative**, $\frac{\partial L}{\partial \theta_i}$ is the **derivative with respect to the deflection angle** of the first or second pendulum, and $\frac{\partial L}{\partial \dot{\theta}_i}$ is the **derivative of the Lagrangian function with respect to the angular velocity** of the first or second pendulum, the angular acceleration for the first pendulum rod, $\ddot{\theta}_1$, follows:

<img width="702" height="82" alt="image" src="https://github.com/user-attachments/assets/6361f7a7-4255-4efa-8236-a72c1a044463" />

And for the second pendulum rod:

<img width="531" height="68" alt="image" src="https://github.com/user-attachments/assets/ca57c3fc-fce0-444e-9005-8badbd7b0b2b" />

By substituting $\theta_1, \dot{\theta}_1, \theta_2$ and $\dot{\theta}_2$ with $u_1, u_2, u_3$ and $u_4$, we get $u$:

<img width="91" height="111" alt="image" src="https://github.com/user-attachments/assets/397afa74-d37a-4dec-be1a-7f9ff93f76fa" />


Taking the time derivative and solving the system of equations consisting of $\ddot{\theta}_1$ and $\ddot{\theta}_2$, we get $\dot{u}$ with $c = \cos(\theta_1-\theta_2)$ and $s = \sin(\theta_1-\theta_2)$:

<img width="616" height="189" alt="image" src="https://github.com/user-attachments/assets/8f063f01-c69a-4ada-981a-5b16e9184818" />

The training data for the pendulum was generated using this approach, where $u$ corresponds to the **initial conditions**.

# Integration of Physical Properties (PINN Integration-function)

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

## Training Methodology

This chapter presents the different training approaches and their practical implementation for the double pendulum and thermal diffusion problems. Particular emphasis is placed on the investigation of different network architectures and the execution of parameter studies to optimize model performance.  
The neural networks were implemented using the Python library **PyTorch**.

---

### Preparation of Training Data

Since the training data originates from simulation data and is therefore ordered, the data within the training batches is called randomly during training. This avoids biased results in which the model detects structures too early in the dataset [14].  
Because our model is trained on more than one variable, the data must also be **scaled** prior to training, as each variable spans a different range between its minimum and maximum values.  
For this purpose, the data is **normalized**, i.e., the dataset’s value range is mapped proportionally from an arbitrary numerical range to either $[0,1]$ or $[-1,1]$ [16].

---

### Network Architecture and Dimensions

In principle, the network architectures for the ML model of the double pendulum and for thermal diffusion are similar; however, the parameters differ.  
- For the **double pendulum**, the input and output layers consist of $\theta_1$, $\dot{\theta}_1$, $\theta_2$, and $\dot{\theta}_2$, combined in the vector $u$. The layer size is therefore 4.  
- For the **one-dimensional diffusion**, the size of the input and output layers corresponds to the number of nodes, which in this case is 60.  

The hidden layers in both cases consist of 100 nodes (or LSTM cells).  
The number of hidden layers was determined depending on the size of the training dataset. In the parameter study, this number is variable (see Table 2).  
In the comparison with the benchmark from [12], the neural network had only one hidden layer.

---

### ML Models of the Parameter Studies

Two parameter studies were conducted:  
1. Training and testing on seven different **initial conditions**.  
2. Training and testing on eight different **double pendulum parameter combinations**.  

Within this study, different neural networks with varying dimensions were created. Their specifications are shown in Table 2.  
The model names are composed of the columns, joined with an underscore.  
For example, the designation for model number 1 is:  
`Lagrange_3_3000x1_InitialCondition`.  
The numbering is for convenience to allow precise referencing of each model.

---

### Table 2: Neural Networks Trained for the Parameter Studies

| No. | Property         | Hidden Layers | Training Epochs × Cycles | Training Variation |
|-----|------------------|---------------|--------------------------|--------------------|
| 1   | Lagrange         | 3             | 3000 × 1                 | Initial Condition  |
| 2   | Energy Conserv.  | 3             | 3000 × 1                 | Initial Condition  |
| 3   | Lagrange         | 7             | 1000 × 7 + 2000 × 1      | Initial Condition  |
| 4   | Energy Conserv.  | 7             | 1000 × 7 + 2000 × 1      | Initial Condition  |
| 5   | Lagrange         | 7             | 1000 × 7                 | Initial Condition  |
| 6   | Energy Conserv.  | 7             | 1000 × 7                 | Initial Condition  |
| 7   | Lagrange         | 3             | 3000 × 1                 | Model Parameters   |
| 8   | Energy Conserv.  | 3             | 3000 × 1                 | Model Parameters   |
| 9   | Lagrange         | 8             | 700 × 8 + 6000 × 1       | Model Parameters   |
| 10  | Energy Conserv.  | 8             | 700 × 8 + 6000 × 1       | Model Parameters   |
| 11  | Lagrange         | 8             | 700 × 8                  | Model Parameters   |
| 12  | Energy Conserv.  | 8             | 700 × 8                  | Model Parameters   |

---

To clarify the fourth column:  
If the number of cycles is greater than one, the models were trained **separately on one condition per cycle**. After each training, one layer was **frozen** (see Figure 7), so that the trained values were not overwritten in the subsequent cycle.  

- Example: In training cycle $x = 1$, no layer is frozen, and all layers are trained.  
- To prevent the loss of all adjusted weights and the risk of the model being trained on only one state, in cycle $x = 2$, the **first layer** of the neural network is excluded from training and no longer updated.  

The purpose of freezing layers as the training cycles progress is to investigate whether **targeted training of layers** influences the model’s performance.

---
<img width="1013" height="497" alt="image" src="https://github.com/user-attachments/assets/0e4c4aed-8a70-463b-ab9d-5c4826d17ab3" />

**Figure 7:** Schematic representation of the training process showing how individual layers are "frozen" or excluded from training.

For models 3, 4, 9, and 10, an additional training step was conducted. After the cycle-based training, the models were further trained with a dataset that combined **all conditions**, without freezing network layers separately.  
This step was designed to strengthen the model’s ability to generalize across different initial conditions.

# Evaluation and Interpretation of Physics-Informed Machine Learning Models

This section details the evaluation and interpretation of the trained Machine Learning models through various tests. We'll start by assessing the investigations on the double pendulum and conclude with the evaluation of one-dimensional thermal diffusion. The term "error" refers to the **deviation between the model's prediction and the simulated test data**. A large error indicates a less accurate prediction and is therefore undesirable.

---

## 5.1 Double Pendulum Investigations

This section presents the results of tests conducted on the double pendulum. We first examine the influence of physical information on the neural network, followed by parameter studies to determine if the networks can generalize further.

### 5.1.1 Comparison of Trained Models

To ensure a consistent comparison with the ML model for double pendulum dynamics developed by Dennis Gannon in [12], we evaluate the different models at the **same energy level** as their training. The pendulum's model parameters are initialized as `m1` kg, `m2` kg, `l1` m, and `l2` m.

Among the trained models, significant differences are immediately apparent. Notably, the model trained solely on data shows large deviations from the simulated pendulum. **Figure 8** illustrates the pendulum's motion at four different time points (25%, 50%, 75%, and 100% of the total time). The simulated motion approximates the real double pendulum movement and serves as the reference for comparing neural network predictions.

* "Actual Movement" (blue) refers to the position predicted by the conventional ML model.
* "Data-based" (green) refers to the position predicted by the conventional ML model.
* "Energy Conservation" (red) refers to the model trained with the physical information of energy conservation.
* "Lagrange" (purple) refers to the model that incorporates the Lagrange formalism.

<img width="688" height="370" alt="image" src="https://github.com/user-attachments/assets/9b5a85df-8d71-4550-9c79-fe35ae70a263" />

**Figure 8: Pendulum motion state at 25%, 50%, 75%, and 100% of the total time**

At the beginning of the simulation, all models remain close to the actual motion. For the physics-informed models, this state is nearly identical, whereas the precision of the data-based model (Dennis Gannon's benchmark) significantly deteriorates over time [12]. To investigate this behavior more closely, we plotted the prediction deviations of the models over time in **Figure 9**. The trend line, calculated from the absolute errors of the predicted angles and angular velocities, clearly shows how the benchmark's predictions, specifically the data-based model, significantly worsen over time.

<img width="896" height="641" alt="image" src="https://github.com/user-attachments/assets/f1a06be7-644a-4085-9876-0391aa97201e" />

**Figure 9: Prediction errors and trend lines of the absolute errors of the model over the simulation period**

Comparing the errors between the two physics-informed models reveals that both the Lagrange-informed and energy conservation-informed models behave similarly. When comparing the trend lines of the absolute errors, the increase in error for the Lagrange model over time is greater than that for energy conservation. This means the model trained with **energy conservation** performs slightly better over time. This could be due to the numerical calculation of the second derivatives of the deflection angles required for evaluating the Lagrange equation (as described in "Compliance with the Lagrange Equation"), which introduces additional deviation into the evaluation.

The boxplot in **Figure 10** shows the median and spread of the deviation between predictions and test data for the entire simulation time. It's clear that the second pendulum arm exhibits larger deviations than the first arm, while the angular velocities show the largest error spread. Comparatively, the model trained with **energy conservation** performed best, though the difference from the Lagrange model is marginal. Across all models, predictions of angular velocities are less accurate.

<img width="827" height="472" alt="image" src="https://github.com/user-attachments/assets/3e6bc0f7-f752-453b-9a94-b52f7f132383" />

**Figure 10: Comparative representation of the prediction errors of the different models**

This changes when the same models are tested on pendulum motion in a higher energy state, also known as a chaotic state [10]. Under this condition, the model errors in **Figure 11** are almost identical.

<img width="827" height="472" alt="image" src="https://github.com/user-attachments/assets/3457c4de-e5ba-43c1-9e95-bf3dc01ba5b8" />

**Figure 11: Comparison of different models tested on chaotic data**

It's interesting that the largest prediction error is observed for `$` (likely a placeholder, please insert the correct variable here), which could be due to the chaotic behavior of the double pendulum.

### 5.1.2 Training Process: MSE and MAE Loss Functions Compared

Training is only meaningfully stopped when progress is marginal or the loss converges. This section examines whether the model is fully trained by comparing the effects of **MAE (Mean Absolute Error)** and **MSE (Mean Squared Error)** loss functions on training, as introduced in Chapter 2.3 "Loss Functions." The benchmark model was trained only with the MSE loss function [12].

Observations showed that the MSE data loss quickly becomes very small during training, as errors are further reduced by squaring. Therefore, training with the MAE loss function is of particular interest. In **Figure 12**, the MSE training loss shows that the model makes little progress after approximately 1000 epochs, as both data loss and physics loss no longer show a significant decrease. Additionally, the model was tested on unseen data during training, with the test loss remaining almost constant throughout. This indicates that the model was fully trained very early, achieving only minimal improvements in the training set without positively impacting test performance.

<img width="819" height="423" alt="image" src="https://github.com/user-attachments/assets/ea12cdce-ee01-4e82-bb31-62a61514c252" />

**Figure 12: Losses and test results for training with Mean Squared Error (MSE) per training epoch over the entire training duration**

Since the test loss is low, it demonstrates that the model generalizes well, and further training progress will have only minor effects on overall performance. The MAE training curve in **Figure 13** behaves similarly; the model quickly learned and made only small progress in training loss.

<img width="861" height="450" alt="image" src="https://github.com/user-attachments/assets/1dfa5119-51ba-408c-b89c-40c3cd54f709" />

**Figure 13: Losses and test results for training with Mean Absolute Error (MAE) per training epoch over the entire training duration**

Notably, the physics losses in both training curves are many times larger than the data losses. As expected, the data losses for the MSE method were marginal compared to the MAE method. The similarity of the energy conservation losses and the test losses from **Figure 12** and **Figure 13** indicates that the choice between MSE and MAE loss functions has little influence on training and final performance. To investigate this influence more precisely, two models were trained without physical losses, using only either the MSE or MAE loss function, and compared in a boxplot in **Figure 14**. This plot provides a comparative view of the performance of both models.

<img width="771" height="466" alt="image" src="https://github.com/user-attachments/assets/299f5b0a-168d-4e68-a0c9-477c65726747" />

**Figure 14: Performance comparison of models trained with MSE or MAE loss functions**

The prediction errors differ minimally. Thus, the choice of the data-based loss function has no significant impact on the models' results. Especially under the influence of physical loss, data losses are relatively small and don't significantly affect the model's total loss.

### 5.1.3 Parameter Study of Different Initial Conditions

Previously, models were trained on a single initial condition. Now, we investigate the effect of training with multiple initial conditions. For this, PINNs were trained and tested on seven different initial conditions. Additionally, the models trained in **Section 5.1.4 "Parameter Study of Different Parameters"** were included in this test.

This study examines various initial conditions and their resulting total kinetic energies. First, we conducted "Tests with known energies" as in **Table 3**, but with different initial conditions than those used during training. This is due to the split of the simulation dataset into training and test sets. As illustrated in **Figure 15**, the simulation begins with the training initial conditions and is split after 80% of the time for training and testing.

<img width="517" height="108" alt="image" src="https://github.com/user-attachments/assets/5b01b1d4-b6a5-4c16-b748-9ea36339643e" />

**Figure 15: Schematic representation of how simulation data was split into training and test datasets**

In the column "Tests with unknown energies," the total kinetic energy, angular displacements, and angular velocities were varied. This test series aims to investigate how the model performs on completely unknown test data.

**Table 3: Initial conditions of the training data and resulting initial conditions of the test data**

| Test | Training ($\theta_1$ in °) | Training ($\theta_2$ in °) | Known Energy Test ($\theta_1$ in °) | Known Energy Test ($\dot{\theta}_1$ in rad/s) | Known Energy Test ($\theta_2$ in °) | Known Energy Test ($\dot{\theta}_2$ in rad/s) |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| 1. | -15 | 30 | -12.5 | 0.54 | 23.6 | -1.33 |
| 2. | -10 | 25 | -7.9 | 0.43 | 19.4 | -1.14 |
| 3. | -5 | 20 | -3.6 | 0.29 | 15.6 | -0.88 |
| 4. | 0 | 15 | 0.6 | 0.13 | 12.2 | -0.57 |
| 5. | 5 | 10 | 4.8 | -0.05 | 9 | -0.2 |
| 6. | 10 | 5 | 8.8 | -0.24 | 6 | 0.18 |
| 7. | 15 | 10 | 13 | -0.41 | 2.7 | 0.54 |

Following these test series, which are close to the training data, another series with initial conditions, as shown in **Table 4**, resulted in an unknown total kinetic energy – i.e., energies that were not trained on.

**Table 4: Test data with completely unknown initial conditions for the trained models**

| Test | Unknown Energy Test ($\theta_1$ in °) | Unknown Energy Test ($\dot{\theta}_1$ in rad/s) | Unknown Energy Test ($\theta_2$ in °) | Unknown Energy Test ($\dot{\theta}_2$ in rad/s) |
| :--- | :--- | :--- | :--- | :--- |
| 1. | -17 | 0.62 | 18.8 | -1.3 |
| 2. | -12.5 | 0.52 | 14.5 | -1.14 |
| 3. | -8.1 | 0.39 | 10.7 | -0.89 |
| 4. | -3.9 | 0.23 | 7.2 | -0.56 |
| 5. | 0.2 | 0.04 | 4 | -0.19 |
| 6. | 4.3 | -0.14 | 0.95 | 0.19 |
| 7. | 8.4 | -0.32 | -2.2 | 0.55 |

For evaluation, we examine the deviations between predicted and actual values of the four state variables (`theta1`, `theta2`, `omega1`, `omega2`) for each test run. These deviations were calculated using the **Mean Absolute Deviation (MAE)**. Since deviations are calculated using unscaled data, we must consider the mean deviation of angles and angular velocities separately due to unit differences.

**Figure 17** summarizes the trained models from `Table 2` (please ensure this table exists and is correctly referenced) by their MAE for `theta1`, `theta2`, `omega1`, and `omega2` across the different test runs in a heatmap. The heatmap represents the data as a color-coded matrix, where MAE values are shown through varying color intensities. This allows for easy visual identification of models with low (better) and high (worse) deviations. Additionally, we've included further thresholds to better visualize the models' functionality in different MAE ranges and to more clearly differentiate their performance within specific value ranges. For better understanding of these thresholds, **Figure 16** provides a representative display of the actual and predicted pendulum motion for models that can be assigned to the existing threshold ranges.

<img width="945" height="303" alt="image" src="https://github.com/user-attachments/assets/35fb898d-03b5-46d7-9c12-2672e084adf2" />

**Figure 16: Pendulum motion over 25 seconds compared with model predictions with various mean absolute deviations**

From the heatmap in **Figure 17**, it's evident that training models by specifically focusing on only one initial condition per cycle (see Models 5 and 6) can yield good results in tests with known energies. Further improvement can be achieved through additional training, as seen in Models 3 and 4, which were trained again with a mix of already trained initial conditions. Despite this, predictions in tests on an unknown system energy remain of limited representativeness due to the large mean absolute deviations of angular velocities.

<img width="831" height="524" alt="image" src="https://github.com/user-attachments/assets/d2e3b378-0100-4dca-bd27-87c46268c794" />

**Figure 17: Heatmap comparing models using mean absolute deviations across corresponding test runs**

Models 1 and 2, trained only on mixed initial conditions, perform significantly worse. This is also true for models trained with different model parameters. Generally, the performance of models trained on different parameters is inferior to those trained on different initial conditions. However, it's noteworthy that models 7-12 perform slightly better in the second test run, likely due to their distinct training methodology.

### 5.1.4 Parameter Study of Different Parameters

Furthermore, models were trained on multiple parameters to assess how well they generalize to predicting different parameter sets. The study procedure mirrors that in **Section 5.1.3 "Parameter Study of Different Initial Conditions,"** but instead of changing initial conditions, we vary the parameters.

**Table 5** lists the model parameter combinations used for training and the first test run. However, this test was conducted with different initial conditions but the same total energy, as further explained in **Figure 15**.

**Table 5: Model parameter combinations used for training and the first test run**

| Test | `m1` in kg | `l1` in m | `m2` in kg | `l2` in m |
| :--- | :--- | :--- | :--- | :--- |
| 1. | 2 | 1.4 | 1 | 1 |
| 2. | 2 | 3 | 1 | 1 |
| 3. | 2 | 1.4 | 2 | 1 |
| 4. | 2 | 3 | 2 | 1 |
| 5. | 3 | 1.4 | 1 | 1 |
| 6. | 3 | 3 | 1 | 1 |
| 7. | 3 | 1.4 | 2 | 1 |
| 8. | 3 | 3 | 2 | 1 |

Since tests with the same parameters are quite close to the training data, we conducted an additional test run where parameter combinations were newly mixed, as shown in **Table 6**.

**Table 6: Mixed model parameter combinations to test the model on completely unknown scenarios**

| Test | `m1` in kg | `l1` in m | `m2` in kg | `l2` in m |
| :--- | :--- | :--- | :--- | :--- |
| 1. | 1 | 1 | 2 | 1.4 |
| 2. | 1 | 1 | 2 | 3 |
| 3. | 2 | 1 | 2 | 1.4 |
| 4. | 2 | 1 | 2 | 3 |
| 5. | 1 | 1 | 3 | 1.4 |
| 6. | 1 | 1 | 3 | 3 |
| 7. | 2 | 1 | 3 | 1.4 |
| 8. | 2 | 1 | 3 | 3 |

The evaluation, as described in Chapter 5.3 (please ensure this chapter exists and is correctly referenced), involves creating a heatmap in **Figure 19** based on the Mean Absolute Deviations (MAE) of the various models across the respective test runs. To simplify interpretation, we used selected thresholds to further characterize the cells, differentiating between usable and unusable models. **Figure 18** provides a representative illustration of these thresholds. **Figure 18a)**, showing the lowest observed mean deviations, accurately depicts the pendulum motion, whereas **Figure 18b)** only moderately represents the simulated data, and **Figure 18c)** is unusable.
<img width="815" height="229" alt="image" src="https://github.com/user-attachments/assets/bd948d3f-3d39-4ce4-935d-f64bd426c1d6" />

**Figure 18: Pendulum motion over 25 seconds compared with model predictions with various mean absolute deviations**

**Figure 19** reveals that all models have greater difficulty predicting different model parameter combinations than different initial conditions. For Models 1, 2, 7, and 8, which were trained only with mixed initial conditions (Models 1 and 2) or mixed parameters (Models 7 and 8), the mean absolute deviation from the tests exceeds the thresholds for both angles and angular velocities. Therefore, this training method is the worst in this study.

Models trained on various initial conditions generally perform worse than those whose training included model parameter combinations. However, models trained on different model parameters also struggle with predicting values. For models 9 to 12, only the test run with known model parameter combinations is representative. The **training method** in combination with the **model's dimensions** proves to be a crucial factor for model performance and should not be overlooked.

<img width="787" height="490" alt="image" src="https://github.com/user-attachments/assets/6f7796cf-f03e-4370-ac2f-fdf6d47badf6" />

**Figure 19: Heatmap comparing models using mean absolute deviations across corresponding test runs**

It's clear that generalizing to different model parameters is significantly more challenging than generalizing to different initial conditions. Although models trained on various model parameters generally perform better than those trained on various initial conditions, the results remain of limited representativeness. In the study from **Section 5.1.3 "Parameter Study of Different Initial Conditions,"** models trained on multiple initial conditions could predict completely unknown states.

---

## 5.2 Investigation of Thermal Diffusion

In addition to the parameter studies, we extended the integration of physical laws to one-dimensional thermal diffusion. The models were trained and tested on a sine distribution. Since thermal diffusion, unlike the double pendulum, is not a chaotic system and thus provides the prerequisite for good generalizability, the number of training epochs for all models was limited to 700. The goal of this limitation was to investigate whether the physics-informed models learn faster than the purely data-based model.

**Figure 20** displays the 3D plots. In this case, the color scaling does not represent temperature but rather the **absolute deviation from the simulation results**. It's clearly visible that the data-based model exhibits the largest deviations, particularly in areas with large gradients and towards the end of the simulation period. In contrast, the physics-informed models provide significantly more precise predictions across the entire simulation range. In this plot, there's hardly any difference between the model informed by heat energy conservation and the model informed by the heat equation. It's also noteworthy that the inflection point in the sine curve's temperature distribution shows very low deviation in all three predictions. Since the error development over the simulation time is not clearly evident from the color coding alone, **Figure 21** provides a clearer illustration. The difference between data-based and physics-informed models becomes even more apparent. Interestingly, the error of the data-based model drops sharply again before increasing.

<img width="327" height="1027" alt="image" src="https://github.com/user-attachments/assets/6fa8c7c2-bd8d-4e77-a711-6188c136c577" />

**Figure 20: 3D Plots of Absolute Deviation from Simulation Results)**

<img width="747" height="290" alt="image" src="https://github.com/user-attachments/assets/27b64acc-b773-4912-825d-e36e8b060dcb" />

**Figure 21: Error development of the different models over the simulation period**

Among themselves, the different variants of the physics-informed networks show only minor differences.

Since all models underwent the same training duration of 700 epochs, it's evident that the physics-informed models learned significantly more efficiently under these conditions than the purely data-driven model. This observation is particularly relevant for applications where training resources are limited.

# Conclusion and Outlook

This investigation into Physics-Informed Neural Networks (PINNs) in the context of the double pendulum and thermal diffusion has yielded several significant findings.

Comparing the data-driven model from [12] (used as a benchmark) with physics-informed models, under the same training conditions, demonstrated a **positive influence of physical information on the prediction accuracy of the models**. Within two parameter studies, we attempted to generalize the PINNs to multiple initial conditions and model parameters. While some models could accurately predict various movement dynamics depending on initial conditions, the results for predicting different dynamics based on varying model parameters were poorer. This highlighted limitations in the models' generalization capabilities.

In comparison to the models trained on thermal diffusion, the **efficiency of PINNs in training was notable**. They learned significantly faster than purely data-driven models within the same training duration of 700 epochs. This underscores the practical utility of such models for applications where training resources are limited. By integrating physical domain knowledge, we could not only create more precise models but also utilize available training data and computational resources more efficiently.

For future research, several promising development opportunities arise:

* A natural extension would be the application of PINNs to thermal diffusion problems with **non-adiabatic boundary conditions**.
* Another possibility is to transfer PINNs to **two-dimensional thermal diffusion problems**, which would make the models useful for more complex real-world scenarios.

Regarding the double pendulum, the results from tests in the chaotic state show an interesting discrepancy. Although PINNs and the benchmark model exhibited clear performance differences in the non-chaotic range, they produced similar results under chaotic conditions, meaning no improvement was gained from PINNs. This observation suggests that the challenge of the chaotic state may not primarily lie in the integration of physical information but could have other causes. Therefore, a promising research direction would be a targeted investigation of the models in **chaotic movement scenarios**.

One approach could be to **increase the temporal resolution of the training data**. Finer time steps would provide the models with more precise input data, which could help in predicting chaotic pendulum movements. Furthermore, an investigation to clarify the discrepancy between the parameter studies—where models learned various model parameter combinations less effectively than initial conditions—would be interesting. Here, examining the **differences in kinetic energy** under various initial conditions and different model parameters could be valuable to rule out that the parameter combinations resulted in excessively large energy differences for the models' predictions. It's possible that the mixing of parameters led to larger energy differences compared to altered initial conditions, which could explain the disparate model performances.

Additionally, as noted in **Section 5.1.2 "Training Process: MSE and MAE Loss Functions Compared,"** the data losses had almost no influence on the total loss. Therefore, investigating the **potential of PINNs for unsupervised learning** would also be interesting, where random input data is used and the model is trained solely through the physical loss functions.

Overall, it's evident that the further development of Physics-Informed Neural Networks holds **potential for modeling physical systems**, but also exhibits limitations, as observed in the parameter studies.

# Used Literature

* [10] R. B. L. a. S. M. Tan, "Double pendulum: An experiment in chaos," Picarro Inc., Santa Clara, California, U.S.A., 1993.
* [11] H. D. Baehr and K. Stephan, *Wärme- und Stoffübertragung* (Heat and Mass Transfer), Stuttgart: Springer Vieweg, 2016.
* [12] D. Gannon, "Deep Learning with Real Data and the Chaotic Double Pendulum," Research Gate, School of Informatics, Computing and Engineering, Indiana University, USA, 2021.
* [14] T. Taulli, *Grundlagen der Künstlichen Intelligenz - Eine nichttechnische Einführung* (Foundations of Artificial Intelligence - A Non-Technical Introduction), Monrovia: Springer, 2022.
* [16] E. Hossain, *Machine Learning Crash Course for Engineers*, Boise, ID, USA: Springer, 2024.
* [22] G. Gonzalez, "Single and Double plane pendulum," Louisiana State University, Baton Rouge, LA, USA.
* [23] S. Alam, B. Gower and N. Mueller, "Numerical modeling of heat transfer mechanism in remote sensing bulb of thermal expansion valves," Elsevier Masson SAS., East Lansing, 48824, MI, USA, 2024.
