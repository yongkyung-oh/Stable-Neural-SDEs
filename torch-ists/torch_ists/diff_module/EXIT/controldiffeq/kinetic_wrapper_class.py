import torch
import torch.nn as nn
#################
kinetic_energy = 1. # int_t ||f||_2^2 <float, None>
jacobian_norm2 = 1. # int_t ||df/dx||_F^2 <float, None> 
#################
# if kinetic_energy and jacobian_norm2:
# 
class KineticWrapper(nn.Module):
    def __init__(self, model, kinetic_energy_coef, jacobian_norm2_coef, div_samples=1):
        super().__init__()
        self.model = model
        self.kinetic_energy_coef, self.jacobian_norm2_coef = kinetic_energy_coef, jacobian_norm2_coef
        self.div_samples = div_samples
    def forward(self, t, z):
        
        with torch.set_grad_enabled(True):
            h0, z0 = z[0], z[1]
            h0.requires_grad_(True)
            t.requires_grad_(True)
            for s_ in z[2:]:
                s_.requires_grad_(True)
            dhdt, last_out = self.model(t,(h0, z0))
            dhdt.requires_grad_(True)
            sqjacnorm = self.jacobian_frobenius_regularization_fn(h0, dhdt)
            quad = self.quadratic_cost(dhdt)
        return (dhdt, last_out, sqjacnorm, quad)
    def jacobian_frobenius_regularization_fn(self, h0, dhdt):
        
        sqjacnorm = []
        
        # torch.manual_seed(2021)

        # torch.cuda.manual_seed(2021)
        # torch.cuda.manual_seed_all(2021)
        # torch.random.manual_seed(2021)
        for e in [torch.randn_like(h0) for k in range(self.div_samples)]:
            e_dhdt_dx = torch.autograd.grad(dhdt, h0, e, create_graph=True)[0]
            n = e_dhdt_dx.view(h0.size(0),-1).pow(2).mean(dim=1, keepdim=True)
            sqjacnorm.append(n)
        return torch.cat(sqjacnorm, dim=1).mean(dim=1)
    def quadratic_cost(self, dx):
        dx = dx.view(dx.shape[0], -1)
        return 0.5*dx.pow(2).mean(dim=-1)
