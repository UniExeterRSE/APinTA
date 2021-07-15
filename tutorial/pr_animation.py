from typing import List, Sequence, Optional
import matplotlib.animation
from mpl_toolkits.mplot3d.art3d import Line3D
import numpy as np
import matplotlib.pyplot as plt
import mpl_toolkits.mplot3d.axes3d as p3

class PRanimation3d:
    def __init__(self, x_gross, x_fine,
                 var_range : Sequence[Sequence[float]],
                 var_names : Optional[Sequence[str]] = None,
                 max_delay : int = 0) -> None:
        self.fig = plt.figure()
        self.ax = p3.Axes3D(self.fig)
        
        self.x_gross = x_gross
        self.x_fine = x_fine
        
        self.ax.set_xlim3d(var_range[0])
        self.ax.set_ylim3d(var_range[1])
        self.ax.set_zlim3d(var_range[2])
        
        if var_names:
            self.ax.set_xlabel(var_names[0])
            self.ax.set_ylabel(var_names[1])
            self.ax.set_zlabel(var_names[2])
            
        n_vars, self.n_gross, self.n_fine, self.iters = self.x_fine.shape
        assert n_vars == 3
        
        self.current_iter = 0
        self.fine_lines : List[Line3D] = [self.ax.plot([], [], [], 'r')[0] for i in range(self.n_gross)]
        self.fine_line_limit = 1
        self.coarse_dots : List[Line3D] = [self.ax.plot([], [], [], 'bo')[0] for i in range(self.n_gross)]
        self.coarse_dots_end : List[Line3D] = [self.ax.plot([], [], [], 'ko')[0] for i in range(self.n_gross)]
        self.draw_coarse_dots()
        
        self.delay_counter = 0
        self.max_delay = max_delay
        
    def update(self, *_):
        # Have we finished the fine lines
        if self.fine_line_limit == self.n_fine:
            if self.delay_counter == 0:
                self.add_end_coarse_dots()
            if self.delay_counter < self.max_delay:
                self.delay_counter += 1
                return
            print('Ending iteration', self.current_iter)
            self.current_iter += 1
            if self.current_iter >= self.iters - 1:
                print('Stop!')
                raise StopIteration
            self.delay_counter = 0
            self.clear_fine_lines()
            self.draw_coarse_dots()
        else:
            self.extend_fine_lines()
        
    def draw_coarse_dots(self):
        for i in range(self.n_gross):
            self.coarse_dots[i].set_data_3d(self.x_gross[:, i, self.current_iter])
            self.coarse_dots_end[i].set_data_3d([], [], [])
            
    def add_end_coarse_dots(self):
        for i, dot in enumerate(self.coarse_dots_end):
            dot.set_data_3d(self.x_gross[:, i, self.current_iter+1])
    
    def clear_fine_lines(self):
        for line in self.fine_lines:
            line.set_data_3d([], [], [])
        self.fine_line_limit = 1
        
    def extend_fine_lines(self):
        for i, line in enumerate(self.fine_lines):
            line.set_data_3d(self.x_fine[:, i, :self.fine_line_limit, self.current_iter+1])
        self.fine_line_limit += 1
        
    def animate(self, save_name, fps=10):
        num_frames = self.iters * (self.max_delay + self.n_fine)
        print(f'Animating {num_frames} frames')
        try:
            animator = matplotlib.animation.FuncAnimation(self.fig, self.update, num_frames)
        except StopIteration:
            pass
        animator.save(save_name, fps=fps)
        
