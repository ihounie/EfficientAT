#pylint: disable=no-member, invalid-name, line-too-long
"""
To be used as an element of torchvision.transforms
"""
import torch
import functools
import math

import torch


class Diffeo(torch.nn.Module):
    """Randomly apply a diffeomorphism to the image(s).
    The image should be a Tensor and it is expected to have [..., n, n] shape,
    where ... means an arbitrary number of leading dimensions.
    
    A random cut is drawn from a discrete Beta distribution of parameters
    alpha and beta such that
        s = alpha + beta (measures how peaked the distribution is)
        r = alpha / beta (measured how biased towards cutmax the distribution is)
        
    Given cut and the allowed* interval of temperatures [Tmin, Tmax], a random T is
    drawn from a Beta distribution with parameters alpha and beta such that:
        s = alpha + beta (measures how peaked the distribution is)
        r = alpha / beta (measured how biased towards T_max the distribution is)
    Beta ~ delta_function for s -> inf. To apply a specific value x \in [0, 1]
    in the allowed interval of T or cut, set
        - s = 1e10
        - r = x / (1 - x)
    *the allowed T interval is defined such as:
        - Tmin corresponds to a typical displacement of 1/2 pixel in the center
          of the image
        - Tmax corresponds to the highest T for which no overhangs are present.
    Args:
        sT (float):  
        rT (float): 
        scut (float):  
        rcut (float): 
        cut_min (int): 
        cut_max (int): 
        
    Returns:
        Tensor: Diffeo version of the input image(s).
    """
    

    def __init__(self, cut, temp, axis = [0]):
        super().__init__()
        
        #self.sT = sT
        #self.rT = rT
        self.cut = cut
        self.temp = temp
        self.axis=axis
        
        #self.betaT = torch.distributions.beta.Beta(sT - sT / (rT + 1), sT / (rT + 1), validate_args=None)
        #self.betacut = torch.distributions.beta.Beta(scut - scut / (rcut + 1), scut / (rcut + 1), validate_args=None)
    
    def forward(self, img):
        """
        Args:
            img (Tensor): Image(s) to be 'diffeomorphed'.
        Returns:
            Tensor: Diffeo image(s).
        """
        
        # image size
        h, w = img.shape[-2:]
        cut = self.cut
        #cut = (self.betacut.sample() * (self.cutmax + 1 - self.cutmin) + self.cutmin).int().item()
        (_, Th), (_, Tw) = temperature_range(h, cut), temperature_range(w, cut)
        #Th, Tw = (self.betaT.sample() * (T2h - T1h) + T1h), (self.betaT.sample() * (T2w - T1w) + T1w)
        
        return deform(img, self.temp*Th, self.temp*Tw, cut, axis=self.axis)
    

    def __repr__(self):
        return self.__class__.__name__ + f'(sT={self.sT}, rT={self.rT}, scut={self.scut}, rcut={self.rcut}, cutmin={self.cutmin}, cutmax={self.cutmax})'


@functools.lru_cache()
def scalar_field_modes(n, m, dtype=torch.float64, device='cpu'):
    """
    sqrt(1 / Energy per mode) and the modes
    """
    x = torch.linspace(0, 1, n, dtype=dtype, device=device)
    k = torch.arange(1, m + 1, dtype=dtype, device=device)
    i, j = torch.meshgrid(k, k)
    r = (i.pow(2) + j.pow(2)).sqrt()
    e = (r < m + 0.5) / r
    s = torch.sin(math.pi * x[:, None] * k[None, :])
    return e, s


def scalar_field(w, h, cut, device='cpu'):
    """
    random scalar field of size wxh made of the first m modes
    """
    #print(f" w:{w}, h:{h}, cut:{cut}")
    e, sw = scalar_field_modes(w, cut, dtype=torch.get_default_dtype(), device=device)
    #print(f"w e: {e.shape}, s {sw.shape}")
    e, sh = scalar_field_modes(h, cut, dtype=torch.get_default_dtype(), device=device)
    #print(f"h e: {e.shape}, s {sh.shape}")
    c = torch.randn(cut, cut, device=device) * e
    #print(f"c {c.shape}")
    return torch.einsum('ij,xi,yj->yx', c, sw, sh)


def deform(image, Th,Tw, cut, interp='linear', axis=[0,1]):
    """
    1. Sample a displacement field tau: R2 -> R2, using tempertature `T` and cutoff `cut`
    2. Apply tau to `image`
    :param img Tensor: square image(s) [..., y, x]
    :param T float: temperature
    :param cut int: high frequency cutoff
    """
    h, w = image.shape[-2:]

    device = image.device.type

    # Sample dx, dy
    # u, v are defined in [0, 1]^2
    # dx, dx are defined in [0, n]^2
    u = scalar_field(w, h, cut, device)  # [n,n]
    #print(f"u {u.shape}")
    v = scalar_field(w, h, cut, device)  # [n,n]
    #print(f"v {v.shape}")
    dx = Tw ** 0.5 * u * w * (0 in axis)
    dy = Th ** 0.5 * v * h * (1 in axis)

    # Apply tau
    return remap(image, dx, dy, interp).contiguous()


def remap(a, dx, dy, interp):
    """
    :param a: Tensor of shape [..., y, x]
    :param dx: Tensor of shape [y, x]
    :param dy: Tensor of shape [y, x]
    :param interp: interpolation method
    """
    n, m = a.shape[-2:]
    #print(a.shape)
    #print(dx.shape)
    #print(dy.shape)
    assert dx.shape == (n, m) and dy.shape == (n, m), 'Image(s) and displacement fields shapes should match.'

    dtype = dx.dtype
    device = dx.device.type
    
    y, x = torch.meshgrid(torch.arange(n, dtype=dtype, device=device), torch.arange(m, dtype=dtype, device=device), indexing='ij')

    xn = (x - dx).clamp(0, m-1)
    yn = (y - dy).clamp(0, n-1)

    if interp == 'linear':
        xf = xn.floor().long()
        yf = yn.floor().long()
        xc = xn.ceil().long()
        yc = yn.ceil().long()

        xv = xn - xf
        yv = yn - yf

        return (1-yv)*(1-xv)*a[..., yf, xf] + (1-yv)*xv*a[..., yf, xc] + yv*(1-xv)*a[..., yc, xf] + yv*xv*a[..., yc, xc]

    if interp == 'gaussian':
        # can be implemented more efficiently by adding a cutoff to the Gaussian
        sigma = 0.4715

        dx = (xn[:, :, None, None] - x)
        dy = (yn[:, :, None, None] - y)

        c = (-dx**2 - dy**2).div(2 * sigma**2).exp()
        c = c / c.sum([2, 3], keepdim=True)

        return (c * a[..., None, None, :, :]).sum([-1, -2])
    
    if interp == 'nearest':
        xn = xn.round().long()
        yn = yn.round().long()
        
        return a[..., yn, xn]


def temperature_range(n, cut):
    """
    Define the range of allowed temperature
    for given image size and cut.
    """
    if isinstance(cut, (float, int)):
        log = math.log(cut)
    else:
        log = cut.log()
    T1 = 1 / (math.pi * n ** 2 * log)
    T2 = 4 / (math.pi ** 3 * cut ** 2 * log)
    return T1, T2


def typical_displacement(T, cut, n):
    if isinstance(cut, (float, int)):
        log = math.log(cut)
    else:
        log = cut.log()
    return n * (math.pi * T * log) ** .5 / 2

def temp_from_displacement(displacement, cut, n):
    if isinstance(cut, (float, int)):
        log = math.log(cut)
    else:
        log = cut.log()
    return 4*(displacement**2) / (math.pi * (n**2) * log) 