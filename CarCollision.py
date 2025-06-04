import numpy as np
import matplotlib.pyplot as plt

def createFootprint(height, width, resolution):
    """
    Creates a rectangular footprint with evenly spaced points.

    Args:
        height (float): height of the rectangle.
        width (float): width of the rectangle.
        resolution (int): number of points per meter

    Returns:
        np.array
    """
    num_points_height = np.ceil(height/resolution) 
    num_points_width = np.ceil(width/resolution) 

    y_points = np.linspace(0, height, int(num_points_height))
    x_points = np.linspace(0, width, int(num_points_width))

    x_grid, y_grid = np.meshgrid(x_points, y_points)

    footprint = np.column_stack((x_grid.flatten(), y_grid.flatten()))
    return np.array(footprint)

def find_nearest_neighbors(points, numNeighbors=4):
    """Find 4 nearest neighbors and return their coordinates, given an array of points

    Args:
        points (np.array): The input array of points.

    Returns:
        np.array: The array of coordinates of the 4 nearest neighbors for each point.
    """
    # Compute pairwise squared Euclidean distances using broadcasting
    dists = np.sum((points[:, np.newaxis, :] - points[np.newaxis, :, :]) ** 2, axis=-1)
    
    # Sort distances along each row, argsort returns sorted indices
    nearest_indices = np.argsort(dists, axis=1)
    
    # Exclude self (index 0), and get the next 4 closest neighbors
    return nearest_indices[:, 1:(numNeighbors+1)]

class Gridworld:
    def __init__(self, maptype):
        self.grid = self.generate_grid(maptype)

    def generate_grid(self, maptype):
        if isinstance(maptype, tuple) and len(maptype) == 2:
            if maptype[0] < 0 or maptype[1] < 0:
                print("Not valid grid")
                return
            # return np.random.randint(0, 2, (maptype[0], maptype[1]))
            return np.random.uniform(0, 1, size=(maptype[0], maptype[1])) < 0.1
        elif isinstance(maptype, np.ndarray):
            return maptype
        elif isinstance(maptype, str):
            #FROM https://github.com/Rishi-V/search-zoo-py/blob/visualizing_basic/v1/users/gridWorld/gridWorldEnv.py
            with open(maptype) as f:
                line = f.readline()  # "type octile"

                line = f.readline()  # "height 32"
                height = int(line.split(' ')[1])

                line = f.readline()  # width 32
                width = int(line.split(' ')[1])

                line = f.readline()  # "map\n"
                assert(line == "map\n")

                mapdata = np.array([list(line.rstrip()) for line in f])

            mapdata.reshape((width,height))
            mapdata[mapdata == '.'] = 0
            mapdata[mapdata == '@'] = 1
            mapdata[mapdata == 'T'] = 1
            mapdata = mapdata.astype(int)
            return mapdata
        else:
            print("Not valid grid")
            return None
    
    def isValid(self, footprint, translation=np.array([[0, 0]]), rotation=np.array([0])):
        """Checks validity

        Args:
            footprint (FT,2): Footprint of robot
            translation (list, optional): Offset. Defaults to [(0, 0]].
            rotation (list, optional): Rotation. Defaults to [0].

        Returns:
            bool list
        """
        if len(translation) != len(rotation): #(K, 2), (K, 1)
            print("Invalid Batch")
            return
        
        rotation = np.deg2rad(rotation) #degrees to radians

        grid = self.grid
        height, width = grid.shape # (H,W)

        cos_values = np.cos(rotation).flatten()  # (K,)
        sin_values = np.sin(rotation).flatten()  # (K,)
        rotation_matrix = np.stack([np.stack([cos_values, -sin_values], axis=-1), 
                                    np.stack([sin_values,  cos_values], axis=-1)], axis=1) #(K, 2, 2)

        #https://numpy.org/doc/2.1/reference/generated/numpy.einsum.html, https://ajcr.net/Basic-guide-to-einsum/
        rotated_points = np.einsum('ij, kjl->kil', footprint, rotation_matrix) #(FT, 2) * (K, 2, 2) = (K, FT, 2)
        points = np.floor(translation[:, np.newaxis, :] + rotated_points).astype(int)  # (K, 1, 2) + (K, FT, 2) = (K, FT, 2)

        in_bounds = np.all((points[:, :, 0] >= 0) & (points[:, :, 0] <= width - 1) &
                       (points[:, :, 1] >= 0) & (points[:, :, 1] <= height - 1), axis=1)  # (K,)
        points = np.clip(points, 0, [width - 1, height - 1])      
        return in_bounds & np.logical_not(np.any(grid[points[:, :, 0], points[:, :, 1]] == 1, axis=1))
    
    def isValidPath(self, footprint, translations, rotations, numNeighbors=4):
        """
        Checks if a path between multiple translation is valid by sampling points along the path.

        Args:
            footprint (FT,2): Footprint of robot
            translation (K, 2): Offset. 
            rotation (K, 1): Rotation. 
            numNeighbors: number of neighbors, int

        Returns:
            numpy bool array (K, numNeighbors)
        """
        if len(translations) != len(rotations):
            print("Invalid Batch")
            return

        nearest_indices = find_nearest_neighbors(translations, numNeighbors)  # (K, m)
        K = translations.shape[0]

        current_pos = translations[:, np.newaxis, :]  # (K, 1, 2)
        neighbor_pos = translations[nearest_indices]  # (K, m, 2)
        distances = np.linalg.norm(neighbor_pos - current_pos, axis=2) # (K, m)
        num_intermediate = np.ceil(distances * 2).astype(int)  # (K, m)
        max_steps = np.max(num_intermediate) 

        intermediate = np.linspace(0, 1, max_steps)  # (N,)
        
        delta_pos = neighbor_pos - current_pos  # (K, m, 2)
        delta_rot = ((rotations[nearest_indices] - rotations[:, np.newaxis] + 180) % 360) - 180  # (K, m)

        interp_pos = (current_pos[:, :, np.newaxis, :] + delta_pos[:, :, np.newaxis, :] 
                      * intermediate[np.newaxis, np.newaxis, :, np.newaxis])  # (K, m, md, 2)
        interp_rot = (rotations[:, np.newaxis, np.newaxis] + delta_rot[:, :, np.newaxis] 
                      * intermediate[np.newaxis, np.newaxis, :])  # (K, m, md)

        flat_interp_pos = interp_pos.reshape(-1, 2)  # (K*m*md, 2)
        flat_interp_rot = np.ravel(interp_rot)         # (K*m*md,)
        valid_flat = self.isValid(footprint, flat_interp_pos, flat_interp_rot)          

        valid_flat = valid_flat.reshape(K, numNeighbors, max_steps)
        return np.all(valid_flat[:, :, :], axis=2)  
    
    def ScaleGrid(self, resolution):
        """returns a fine grain grid

        Args:
            resolution (float): resolution increase by

        Returns:
            numpy bool array
        """
        scale_factor = int(1 / resolution)
        return np.repeat(np.repeat(self.grid, scale_factor, axis=0), 
                    scale_factor, axis=1)
        

    def visualize(self, footprint, translation=np.array([[0, 0]]), rotation=np.array([0]), numNeighbors=4):
        """Visualizes the footprint and its transformation on the grid.

        Args:
            footprint (FT,2): Footprint of robot
            translation (list, optional): Offset. Defaults to [(0, 0)].
            rotation (list, optional): Rotation. Defaults to [0].
        """
        validity = self.isValid(footprint, translation, rotation)
        radrotation = np.deg2rad(rotation)
        grid = self.grid
        fig, ax = plt.subplots()

        # Display the grid
        ax.imshow(grid, cmap='binary', origin='upper', extent=(0, grid.shape[1], grid.shape[0], 0))

        # Calculate points for visualization
        cos_values = np.cos(radrotation).flatten()
        sin_values = np.sin(radrotation).flatten()
        rotation_matrix = np.stack([np.stack([cos_values, -sin_values], axis=-1),
                                    np.stack([sin_values, cos_values], axis=-1)], axis=1)
        rotated_points = np.einsum('ij,kjl->kil', footprint, rotation_matrix)
        points = translation[:, np.newaxis, :] + rotated_points
        reshaped_points = points.reshape(-1, 2)

        # Color points green/red based on validity
        colors = np.where(validity[:, np.newaxis], 'green', 'red').repeat(points.shape[1], axis=0).flatten()
        ax.scatter(reshaped_points[:, 1], reshaped_points[:, 0], c=colors, s=1, zorder=2)

        # Find nearest neighbors and path validity
        nearest_indices = find_nearest_neighbors(translation, numNeighbors)  #Shape: (K, m)
        nearest_points = translation[nearest_indices] #Shape: (K, m)
        path_validity = self.isValidPath(footprint, translation, rotation, numNeighbors) #Shape: (K, m)

        # Plot connections between neighbors 
        color = np.where(path_validity, 'green', 'red')
        for i in range(len(translation)):
            for j in range(numNeighbors):  # Use numNeighbors instead of len(x_values[i])
                x1, y1 = translation[i]
                x2, y2 = nearest_points[i, j]
                ax.plot([y1, y2], [x1, x2], linestyle='-', color=color[i, j])

        # Add grid lines
        ax.set_xticks(np.arange(0, grid.shape[1], 1))
        ax.set_yticks(np.arange(1, grid.shape[0] + 1, 1))
        ax.grid(which='both', color='gray', linestyle='-', linewidth=0.5, alpha=0.5)
        
        plt.show()
    





#Test
footprint = createFootprint(0.5, 0.5, 0.05)
gridworld = Gridworld((10, 10))
num_samples = 25
translation = np.random.sample(size=(num_samples,2))
translation[:,0] *= 9
translation[:,1] *= 9
rotation = np.zeros(shape=num_samples)

gridworld.visualize(footprint, translation, rotation, 3)