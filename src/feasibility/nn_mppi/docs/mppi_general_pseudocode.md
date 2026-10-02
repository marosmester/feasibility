### Algorithm: Model Predictive Path Integral (MPPI) Control

**Given:**

* Current state $x_0$
* Prediction horizon $H$
* Number of sampled trajectories $K$
* Nominal control sequence $\mathbf{u} = \{u_0,\ldots,u_{H-1}\}$
* Noise covariance $\Sigma$
* Temperature parameter $\lambda>0$
* System dynamics $x_{t+1}=f(x_t,u_t)$
* Running cost $\ell(x_t,u_t)$
* Terminal cost $\phi(x_H)$

**Repeat at each control step:**

1. **Sample and roll out** ($k=1,\ldots,K$)

   $$
   u_t^{(k)}=u_t+\epsilon_t^{(k)},\quad \epsilon_t^{(k)}\sim\mathcal{N}(0,\Sigma),\quad x_{t+1}^{(k)}=f\!\left(x_t^{(k)},u_t^{(k)}\right)
   $$

2. **Evaluate costs**

   $$
   J^{(k)}=\phi\!\left(x_H^{(k)}\right)+\sum_{t=0}^{H-1}\ell\!\left(x_t^{(k)},u_t^{(k)}\right)
   $$

3. **Update the nominal controls** ($w_k\propto e^{-J^{(k)}/\lambda}$, normalized)

   $$
   u_t\leftarrow u_t+\sum_{k=1}^{K}w_k\,\epsilon_t^{(k)}
   $$

4. **Apply $u_0$, shift $\mathbf{u}$, observe the new state, repeat.**
