import numpy as np
from feasibility.heightmap import HeightMapReader

ny = 255    # in cells
nx = 255
x0 = 0      # in meters
y0 = 0
H = np.zeros((ny, nx))  # your elevation grid, [row=y, col=x], cell centers

# Set z:
H[80:120, 60:100] = 0.5   # 0.5 m plateau over that cell range

# Create and save
reader = HeightMapReader(H, origin=(x0, y0), cell=0.05)
reader.save("assets/my_terrain")   # writes my_terrain.png + my_terrain.yaml