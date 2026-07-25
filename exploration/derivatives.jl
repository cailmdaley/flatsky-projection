using SymPy
latex(exp) = clipboard(SymPy.latex(exp))
@vars x y δ₀ α₀ real=true

ρ = sqrt(x^2 + y^2)
c = 2*asin(1/2 * ρ)

δ = asin(cos(c) * sin(δ₀) + y * sin(c)*cos(δ₀) / ρ)
α = α₀ + atan(x * sin(c) / (ρ * cos(δ₀) * cos(c) - y * sin(δ₀ * sin(c))))

dδdx = diff(δ, x)
dδdy = diff(δ, y)
dαdx = diff(α, x)
dαdy = diff(α, y)

@vars phi_0 lambda_0 rho cc
s1=(δ₀ => phi_0, α₀ => lambda_0, sqrt(x^2+y^2) => rho)
s2=(2*asin(0.5*rho) => cc)


latex(dδdx(s1...))
\frac{- \frac{ x \sin{\left (\phi_{0} \right )} \sin{\left (c \right )}}{\rho \sqrt{- 0.25 x^{2} - 0.25 y^{2} + 1}} + \frac{ x y \cos{\left (\phi_{0} \right )} \cos{\left (c \right )}}{\rho^{2} \sqrt{- 0.25 x^{2} - 0.25 y^{2} + 1}} - \frac{x y \sin{\left (c \right )} \cos{\left (\phi_{0} \right )}}{\rho^{3}}}{\sqrt{- \left(\sin{\left (\phi_{0} \right )} \cos{\left (c \right )} + \frac{y \sin{\left (c \right )} \cos{\left (\phi_{0} \right )}}{\rho}\right)^{2} + 1}}

latex(dδdy(s1...))
\frac{- \frac{ y \sin{\left (\phi_{0} \right )} \sin{\left (c \right )}}{\rho \sqrt{- 0.25 x^{2} - 0.25 y^{2} + 1}} + \frac{\sin{\left (c \right )} \cos{\left (\phi_{0} \right )}}{\rho} + \frac{ y^{2} \cos{\left (\phi_{0} \right )} \cos{\left (c \right )}}{\rho^{2} \sqrt{- 0.25 x^{2} - 0.25 y^{2} + 1}} - \frac{y^{2} \sin{\left (c \right )} \cos{\left (\phi_{0} \right )}}{\rho^{3}}}{\sqrt{- \left(\sin{\left (\phi_{0} \right )} \cos{\left (c \right )} + \frac{y \sin{\left (c \right )} \cos{\left (\phi_{0} \right )}}{\rho}\right)^{2} + 1}}

latex(dαdx(s1...))
\frac{\frac{x \left(\frac{ \phi_{0} x y \cos{\left (\phi_{0} \sin{\left (c \right )} \right )} \cos{\left (c \right )}}{\rho \sqrt{- 0.25 x^{2} - 0.25 y^{2} + 1}} + \frac{ x \sin{\left (c \right )} \cos{\left (\phi_{0} \right )}}{\sqrt{- 0.25 x^{2} - 0.25 y^{2} + 1}} - \frac{x \cos{\left (\phi_{0} \right )} \cos{\left (c \right )}}{\rho}\right) \sin{\left (c \right )}}{\left(\rho \cos{\left (\phi_{0} \right )} \cos{\left (c \right )} - y \sin{\left (\phi_{0} \sin{\left (c \right )} \right )}\right)^{2}} + \frac{\sin{\left (c \right )}}{\rho \cos{\left (\phi_{0} \right )} \cos{\left (c \right )} - y \sin{\left (\phi_{0} \sin{\left (c \right )} \right )}} + \frac{ x^{2} \cos{\left (c \right )}}{\rho \left(\rho \cos{\left (\phi_{0} \right )} \cos{\left (c \right )} - y \sin{\left (\phi_{0} \sin{\left (c \right )} \right )}\right) \sqrt{- 0.25 x^{2} - 0.25 y^{2} + 1}}}{\frac{x^{2} \sin^{2}{\left (c \right )}}{\left(\rho \cos{\left (\phi_{0} \right )} \cos{\left (c \right )} - y \sin{\left (\phi_{0} \sin{\left (c \right )} \right )}\right)^{2}} + 1}

latex(factor(dαdy(s1...)))
\frac{\frac{x \left(\frac{ \phi_{0} y^{2} \cos{\left (\phi_{0} \sin{\left (c \right )} \right )} \cos{\left (c \right )}}{\rho \sqrt{- 0.25 x^{2} - 0.25 y^{2} + 1}} + \frac{ y \sin{\left (c \right )} \cos{\left (\phi_{0} \right )}}{\sqrt{- 0.25 x^{2} - 0.25 y^{2} + 1}} + \sin{\left (\phi_{0} \sin{\left (c \right )} \right )} - \frac{y \cos{\left (\phi_{0} \right )} \cos{\left (c \right )}}{\rho}\right) \sin{\left (c \right )}}{\left(\rho \cos{\left (\phi_{0} \right )} \cos{\left (c \right )} - y \sin{\left (\phi_{0} \sin{\left (c \right )} \right )}\right)^{2}} + \frac{ x y \cos{\left (c \right )}}{\rho \left(\rho \cos{\left (\phi_{0} \right )} \cos{\left (c \right )} - y \sin{\left (\phi_{0} \sin{\left (c \right )} \right )}\right) \sqrt{- 0.25 x^{2} - 0.25 y^{2} + 1}}}{\frac{x^{2} \sin^{2}{\left (c \right )}}{\left(\rho \cos{\left (\phi_{0} \right )} \cos{\left (c \right )} - y \sin{\left (\phi_{0} \sin{\left (c \right )} \right )}\right)^{2}} + 1}

simplify(dδdx(s1...))
factor(dδdx(s1...)(s2...))

dδdx.subs(sqrt(x^2 + y^2), ρ)

\operatorname{asin}{\left (\frac{y \sin{\left (2 \operatorname{asin}{\left (0.5 \sqrt{x^{2} + y^{2}} \right )} \right )} \cos{\left (\phi_0 \right )}}{\sqrt{x^{2} + y^{2}}} + \sin{\left (\phi_0 \right )} \cos{\left (2 \operatorname{asin}{\left (0.5 \sqrt{x^{2} + y^{2}} \right )} \right )} \right )}


####
# Automatic Differentiation

function k(δ, α, δ₀, α₀)
	sqrt(2 / (1 + sin(δ₀) * sin(δ) + cos(δ₀) * cos(δ) * cos(α - α₀)))
end
x(δ, α, δ₀, α₀) = k(δ, α, δ₀, α₀) * cos(δ) * sin(α-α₀)
y(δ, α, δ₀, α₀) = k(δ, α, δ₀, α₀) * (sin(δ₀)*cos(δ)*cos(α-α₀) - cos(δ₀)*sin(δ))


ρ(x,y) = sqrt(x^2 + y^2)
c(ρ) = 2*asin(1/2 * ρ)
c(x,y) = 2*asin(1/2 * ρ(x,y))

δ(x, y, δ₀) = asin(cos(c(x,y)) * sin(δ₀) - y * sin(c(x,y)) / ρ(x,y) * cos(δ₀))

α(x, y, δ₀, α₀) = α₀ + atan(x * sin(c(x,y)), (ρ(x,y) * cos(δ₀) * cos(c(x,y)) + y * sin(δ₀ * sin(c(x,y)))))

using PyCall
py"""
from spt3g.coordinateutils import FlatSkyMap, MapProjection
from spt3g.coordinateutils.maputils import get_ra_dec_map
from spt3g.core import G3Units
"""

α₀, δ₀ = deg2rad.([50, -50])
m = py"FlatSkyMap(1000, 1, 2*G3Units.arcmin, alpha_center=$(rad2deg.(α₀)), delta_center=$(rad2deg.(δ₀)), proj=MapProjection.ProjZEA)"o

αs_true, δs_true = [deg2rad.(coord) for coord in py"get_ra_dec_map($m)"]

xs = x.(δs_true, αs_true, δ₀, α₀)
ys = y.(δs_true, αs_true, δ₀, α₀)

δs = δ.(xs, ys, δ₀)
αs = α.(xs, ys, δ₀, α₀);



xys_true = [m.angle_to_xy((m.pixel_to_angle(i))...) for i in 0:size(m)[1]-1]
[(m.pixel_to_angle(i))...) for i in 0:size(m)[1]-1]
xys_true = [m.angle_to_xy((m.pixel_to_angle(i))...) for i in 0:size(m)[1]-1]

δs_calc = @. δ(first(xys_true) * m.res , last(xys_true) * m.res, δ₀)
αs_calc = @. α(first(xys_true) * m.res, last(xys_true) * m.res, δ₀, α₀);


δs_true
αs_true

m
