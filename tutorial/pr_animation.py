from typing import Generic, List, Sequence, Optional, TypeVar, Union
import numpy as np
import matplotlib.animation
from matplotlib.colors import Colormap, Normalize
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.lines import Line2D
from matplotlib.axes import Axes
from mpl_toolkits.mplot3d.art3d import Line3D
from mpl_toolkits.mplot3d import Axes3D
from abc import abstractmethod, ABC

T = TypeVar('T', bound='_PRanimation')

class _PRanimation(Generic[T], ABC):
    def __init__(self, x_gross, x_fine,
                 delay : int = 0,               # Delay after each iteration
                 coarse_dot_delay : int = 2,    # Delay between adding each new coarse dot
                 end_delay : int = 10,          # Delay at the end of the animation
                 title : Optional[str] = None,
                 line_colour : Union[Colormap, str, None] = None,
                 dot_colour : Union[Colormap, str, None] = None):
        self.edited_artists : List[T]
        self.fine_lines : List[T]
        self.coarse_dots : List[T]
        self.coarse_dots_end : List[T]
        self.ax : Axes
        
        self.x_fine = x_fine        
        self.x_gross = x_gross
        self.n_gross, self.iters, *_ = self.x_fine.shape
        # This will be lower sometimes for adaptive solves
        self.n_fine_max = max(map(len, self.x_fine[:, -1]))
        self.n_vars = len(self.x_fine[-1, -1][-1])
        
        self.current_iter = 1
        self.drawing_fine_lines = False
        self.fine_line_step = 1
        self.n_fine_steps = self.n_fine_max
        self.current_coarse_dot = 0
        self.delay_counter = 0
        self.finished = False
        self.delay = delay
        self.coarse_dot_delay = coarse_dot_delay
        self.end_delay = end_delay
        self.fig = plt.figure()
        self.title = f'{title} - Iteration:' if title else 'Iteration:'
        
        if isinstance(line_colour, Colormap):
            self.line_colour = cm.ScalarMappable(Normalize(0, self.n_gross*1.3), line_colour)
            self.use_line_cmap = True
        else:
            self.line_colour = line_colour if line_colour else 'r'
            self.use_line_cmap = False
        if isinstance(dot_colour, Colormap):
            self.dot_colour = cm.ScalarMappable(Normalize(0, self.n_gross*1.3), dot_colour)
            self.use_dot_cmap = True
        else:
            self.dot_colour = dot_colour if dot_colour else 'b'
            self.use_dot_cmap = False
        
    @staticmethod
    @abstractmethod
    def set_line_data(line: T, data):
        pass
    
    @staticmethod
    @abstractmethod
    def clear_line(line: T):
        pass
    
    @abstractmethod
    def update_title(self, title: str):
        pass
        
    def init_func(self):
        self.current_iter = 0
        self.drawing_fine_lines = False
        self.fine_line_step = 0
        self.current_coarse_dot = 0
        self.delay_counter = 0
        self.finished = False
        # self.draw_coarse_dots()
        self.update_title(f'{self.title} 0')
        return (*self.coarse_dots, *self.coarse_dots_end, *self.fine_lines)
        
    def update(self, frame_num):
        print(frame_num)
        # Pause at the end of the animation
        if self.finished:
            if self.delay_counter < self.end_delay:
                self.delay_counter += 1
                return ()
            print('Done')
            raise StopIteration
        
        self.edited_artists = []
        # Continue extending the fine lines
        if self.drawing_fine_lines:
            self.extend_fine_lines()
        # Draw in the new coarse dots
        elif self.current_coarse_dot < self.n_gross:
            if self.coarse_dot_delay == 0:
                self.add_all_end_coarse_dots()
            if self.delay_counter < self.coarse_dot_delay:
                self.delay_counter += 1
                return ()
            self.add_end_coarse_dot()
            self.delay_counter = 1
        else:
            # Wait at the end of the iteration
            if self.delay_counter < self.delay:
                self.delay_counter += 1
                return ()
            # After the delay start a new iteration/end the animation
            self.iteration_init()
            if self.finished:
                return self.update(frame_num)
            self.update_title(f'{self.title} {self.current_iter}')
            self.clear_fine_lines()
            self.draw_coarse_dots()
        
        return self.edited_artists
    
    def iteration_init(self):
        self.current_iter += 1
        if self.current_iter >= self.iters:
            self.finished = True
            return
        self.drawing_fine_lines = True
        self.delay_counter = 0
        print('Starting iteration', self.current_iter)
        
    def draw_coarse_dots(self):
        for i in range(self.n_gross):
            self.set_line_data(self.coarse_dots[i], self.x_gross[i, self.current_iter-1])
            self.clear_line(self.coarse_dots_end[i])
        self.current_coarse_dot = 0
        self.edited_artists.extend(self.coarse_dots + self.coarse_dots_end)
            
    def add_all_end_coarse_dots(self):
        for i, dot in enumerate(self.coarse_dots_end):
            self.set_line_data(dot, self.x_gross[i, self.current_iter])
        self.current_coarse_dot = self.n_gross
        self.edited_artists.extend(self.coarse_dots_end)
        
    def add_end_coarse_dot(self):
        self.set_line_data(self.coarse_dots_end[self.current_coarse_dot],
                           self.x_gross[self.current_coarse_dot, self.current_iter])
        self.edited_artists.append(self.coarse_dots_end[self.current_coarse_dot])
        self.current_coarse_dot += 1
    
    def clear_fine_lines(self):
        for line in self.fine_lines:
            self.clear_line(line)
        self.fine_line_step = 1
        self.edited_artists.extend(self.fine_lines)
        
    def extend_fine_lines(self):
        for i, line in enumerate(self.fine_lines):
            data = self.x_fine[i, self.current_iter]
            draw_to = int(len(data)*self.fine_line_step/self.n_fine_steps)
            if draw_to == 0:
                draw_to = 1
            self.set_line_data(line, np.array(data[:draw_to]).T)
        self.fine_line_step += 1
        self.edited_artists.extend(self.fine_lines)
        if self.fine_line_step > self.n_fine_steps:
            self.drawing_fine_lines = False
        
    def animate(self, save_name, fps=10):
        num_frames = self.iters * (self.delay + self.n_fine_max + self.n_gross*self.coarse_dot_delay + 1) + self.end_delay - self.n_fine_max
        print(f'Animating up to {num_frames} frames')
        animator = matplotlib.animation.FuncAnimation(self.fig, self.update, num_frames, self.init_func, blit=True)
        try:
            animator.save(save_name, fps=fps)
        except StopIteration:
            print('Animation stopped')
        
class PRanimation3D(_PRanimation[Line3D]):
    def __init__(self, x_gross, x_fine,
                 var_range: Sequence[Sequence[float]],
                 var_names: Optional[Sequence[str]] = None,
                 delay : int = 0,               # Delay after each iteration
                 coarse_dot_delay : int = 2,    # Delay between adding each new coarse dot
                 end_delay : int = 10,          # Delay at the end of the animation
                 title: Optional[str] = None,
                 line_colour : Union[Colormap, str, None] = None,
                 dot_colour : Union[Colormap, str, None] = None):
        super().__init__(x_gross, x_fine, delay=delay, coarse_dot_delay=coarse_dot_delay, end_delay=end_delay,
                         title=title, line_colour=line_colour, dot_colour=dot_colour)
        
        assert self.n_vars == 3
        self.ax = Axes3D(self.fig)
        
        self.ax.set_xlim3d(var_range[0])
        self.ax.set_ylim3d(var_range[1])
        self.ax.set_zlim3d(var_range[2])
        
        if var_names:
            self.ax.set_xlabel(var_names[0])
            self.ax.set_ylabel(var_names[1])
            self.ax.set_zlabel(var_names[2])
        
        self.edited_artists = []
        if self.use_line_cmap:
            self.fine_lines = [self.ax.plot([], [], [], c=self.line_colour.to_rgba(i))[0] for i in range(self.n_gross)]
        else:
            self.fine_lines = [self.ax.plot([], [], [], c=self.line_colour)[0] for i in range(self.n_gross)]
        if self.use_dot_cmap:
            self.coarse_dots = [self.ax.plot([], [], [], 'o', c=self.dot_colour.to_rgba(i))[0] for i in range(self.n_gross)]
        else:
            self.coarse_dots = [self.ax.plot([], [], [], 'o', c=self.dot_colour)[0] for i in range(self.n_gross)]
        self.coarse_dots_end = [self.ax.plot([], [], [], 'kx')[0] for i in range(self.n_gross)]
        self.text = self.ax.text2D(0.5, 0.95, '', transform=self.ax.transAxes, ha='center')
    
    @staticmethod
    def set_line_data(line_3d: Line3D, data):
        line_3d.set_data_3d(data)
        
    @staticmethod
    def clear_line(line_3d: Line3D):
        line_3d.set_data_3d([], [], [])
        
    def update_title(self, title: str):
        self.text.set_text(title)
        
class PRanimation2D(_PRanimation[Line2D]):
    def __init__(self, x_gross, x_fine,
                 var_range: Sequence[Sequence[float]],
                 var_names: Optional[Sequence[str]] = None,
                 delay : int = 0,               # Delay after each iteration
                 coarse_dot_delay : int = 2,    # Delay between adding each new coarse dot
                 end_delay : int = 10,          # Delay at the end of the animation
                 title: Optional[str] = None,
                 line_colour : Union[Colormap, str, None] = None,
                 dot_colour : Union[Colormap, str, None] = None):
        super().__init__(x_gross, x_fine, delay=delay, coarse_dot_delay=coarse_dot_delay, end_delay=end_delay,
                         title=title, line_colour=line_colour, dot_colour=dot_colour)
        
        assert self.n_vars == 2
        self.ax = self.fig.add_subplot(1, 1, 1)
        
        self.ax.set_xlim(var_range[0])
        self.ax.set_ylim(var_range[1])
        
        if var_names:
            self.ax.set_xlabel(var_names[0])
            self.ax.set_ylabel(var_names[1])
        
        self.edited_artists = []
        if self.use_line_cmap:
            self.fine_lines = [self.ax.plot([], [], c=self.line_colour.to_rgba(i))[0] for i in range(self.n_gross)]
        else:
            self.fine_lines = [self.ax.plot([], [], c=self.line_colour)[0] for i in range(self.n_gross)]
        if self.use_dot_cmap:
            self.coarse_dots = [self.ax.plot([], [], 'o', c=self.dot_colour.to_rgba(i))[0] for i in range(self.n_gross)]
        else:
            self.coarse_dots = [self.ax.plot([], [], 'o', c=self.dot_colour)[0] for i in range(self.n_gross)]
        self.coarse_dots_end = [self.ax.plot([], [], 'kx')[0] for i in range(self.n_gross)]
    
    @staticmethod
    def set_line_data(line_2d: Line2D, data):
        line_2d.set_data(data)
        
    @staticmethod
    def clear_line(line_2d: Line2D):
        line_2d.set_data([], [])
        
    def update_title(self, title: str):
        plt.title(title)

class _PRanimationAdaptive(_PRanimation): 
    def iteration_init(self):
        super().iteration_init()
        if self.finished:
            return
        fine_lens = list(map(len, self.x_fine[:, self.current_iter]))
        self.n_fine_steps = round(np.mean(fine_lens))
        print(self.n_fine_steps, fine_lens)
        
class PRanimationAdaptive3D(PRanimation3D, _PRanimationAdaptive):
    pass

class PRanimationAdaptive2D(PRanimation2D, _PRanimationAdaptive):
    pass