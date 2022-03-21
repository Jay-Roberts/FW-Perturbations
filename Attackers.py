import torch

ch = torch
import numpy as np
from torch.autograd import grad
from torch.nn import CrossEntropyLoss, Module


class FWAttacker:
    """
    Base class for FW attacks. Subclass this and
    switch the LMO for different FW attacks.

    get_adv_perturbation performs FW-Optimization

        d[k+1] = gamma[k] s[k] + (1 - gamma[k]) d[k]
    
    Where gamma[k] = c/(c+k) and s[k] is the solution to

        s* = argmax_{S} <grad[l(x + d[k])] | s>
    
    for the constraint set S.
    """
    def __init__(self, num_steps: int, epsilon: float = 8 / 255.0, c: float = 2.0):
        self.eps = epsilon
        self.N = num_steps
        self.loss = CrossEntropyLoss()
        self.c = c

    def lmo(self, g:ch.Tensor, *args, **kwargs)->ch.tensor:
        pass
    
    def get_adv_perturbation(
                            self,
                            model: Module,
                            x: torch.Tensor,
                            y: torch.Tensor,
                            targeted:bool=False,
                            loss_fn:callable=None
                        ) -> torch.Tensor:
        """
        Performs the FW optimization.

        Parameters
        ----------
            model :
                The embedding function. Should take a tensor
                as input.
            x :
                Initial input to perturb from.
            y :
                A target value or additional tensor. When loss_fn is None
                this is assumed to be a class label.
            targeted :
                When True/False the optimization min/maximizes the loss_fn.
            loss_fn :
                If None then CrossEntropyLoss() is used. Otherwise this is
                the target function to be optimized. Should take (model, x, y)
                as inputs and return a scalar.
        Returns
        -------
            delta :
                The perturbation which approximates the min/max of the
                loss_fn.
        """
        m = -1 if targeted else 1
        delta = torch.zeros_like(x, requires_grad=True)

        for k in range(self.N):
            gamma = self.c / (self.c + k)

            if loss_fn is None:
                output = model(x + delta)
                loss =  self.loss(output, y)
            else:
                loss = loss_fn(model, x + delta, y)

            loss = m * loss
            g = grad(loss, delta)[0]
            s = self.lmo(g)

            delta = gamma * s + (1 - gamma) * delta
        return delta.detach()

class FWLinfAttacker(FWAttacker):
    """
    FW approximation of an Linf adversarial attack.
    """

    def __init__(self, num_steps: int, epsilon: float = 8 / 255.0, c: float = 2.0):
        super().__init__(num_steps, epsilon, c=c)
       

    def lmo(self, g:ch.Tensor)->ch.Tensor:
        """
        The linear maximization oracle for the Linf
        problem.
        """
        return self.eps * g.sign()


class FWL2Attacker(FWAttacker):
    """
    L2 Attack via FW
    """

    def __init__(self, num_steps: int, epsilon: float = 2.0, c: float = 2.0):
        super().__init__(num_steps, epsilon, c=c)
       
    def lmo(self, g:ch.Tensor)->ch.Tensor:
        # LMO for L2 boundary is just
        # L2 normalized gradient.
        shp = g.shape
        B = shp[0]
        
        # Normalize
        g = g.view(B, -1)
        g_norm = ch.norm(g, dim=1, p=2)
        s = g / (g_norm[:, None] + 1e-10)

        s = self.eps * s
        return s.view(shp)
    

class FWL1Attacker(FWAttacker):
    """
    L1 Attack via FW modified to attack the top-q % of pixels with
    largest gradient magnitude. So an attack modifies at most (100-q)%
    of pixels.
    """

    def __init__(self, num_steps: int, q: float = 0.3, pert_q: float = 0.05, **kwargs):
        super().__init__(num_steps, **kwargs)
        self.q = q

        self.q0, self.q1 = q, q
        if pert_q:
            self.q0 = max(0.0, q - pert_q)
            self.q1 = min(1.0, q + pert_q)

    def lmo(self, g:ch.tensor)->ch.Tensor:
        # Perturb q
        q = self.q
        if self.q0 < self.q or self.q1 > self.q:
            q = (self.q1 - self.q0) * np.random.random_sample() + self.q0

        # Convert percent to number of pixels
        shp = g.shape
        nb = shp[0]

        num_pix = np.prod(shp[1:])
        num_q = 1.0 - q
        num_q = max(1, int(num_q * num_pix))

        grad = g.view(nb, -1)

        batch_idx = [[i] * num_q for i in range(nb)]
        batch_idx = list(batch_idx)
        batch_idx = ch.tensor(batch_idx)

        # Find largest q% gradient norms
        _, corners_q = ch.topk(grad.abs(), num_q, dim=1)
        s = ch.zeros_like(grad)
        s[batch_idx, corners_q] = grad.sign()[batch_idx, corners_q]
        
        # Shouldn't matter for true L1 but the L1-q needs to be put
        # back on the boundary        
        s_norm = ch.norm(s, dim=1, p=1)
        s = s / (s_norm[:, None] + 1e-10)
        #s = s.renorm(p=1, dim=0, maxnorm=1.0)

        s *= self.eps
        return s.view(shp)
