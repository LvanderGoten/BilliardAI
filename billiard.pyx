#cython: language_level=3, boundscheck=False, wraparound=False, nonecheck=False
import arcade
from arcade.window_commands import set_viewport
from arcade.window_commands import set_window
import pyglet
import numpy as np
from abc import ABC, abstractmethod
from collections import namedtuple
import math
from bisect import bisect
from time import process_time_ns
from cpython cimport bool

# Named tuples
Step = namedtuple("Step", ["reward", "new_observation", "done"])
Cell = namedtuple("Cell", ["i", "j"])


# Environment constants
MU = 2.
G = 9.81
MIN_VELOCITY = 10
MIN_VELOCITY_SQ = MIN_VELOCITY**2

# Colors
AIR_FORCE_BLUE = (93, 138, 168)
BRITISH_RACING_GREEN = (0, 66, 37)
BRONZE = (205, 127, 50)
RED = (255, 0, 0)
GREEN = (0, 255, 0)
WHITE_SMOKE = (245, 245, 245)


class CushionElement(ABC):
    def __init__(self, color):
        self.color = color

    @abstractmethod
    def collides_with_ball(self, ball):
        pass

    @abstractmethod
    def get_new_ball_direction(self, ball):
        pass


class LinearCushionElement(CushionElement):
    def __init__(self, xa, ya, xb, yb, pocket_radius, color, screen_width, screen_height):
        super().__init__(color=color)

        # Two points on line
        self.xa, self.ya = xa, ya
        self.xb, self.yb = xb, yb

        # Type
        self.orientation = "v" if np.isclose(self.xa, self.xb) else "h"

        # Origin of the line
        self.o_x = xa
        self.o_y = ya

        # Normalized direction vector
        p_x = xb - xa
        p_y = yb - ya
        self.p_norm = math.sqrt(p_x * p_x + p_y * p_y)
        self.p_x = p_x / self.p_norm
        self.p_y = p_y / self.p_norm

        # Filled rectangle
        eps = 1
        if self.orientation == "v":
            self.xf = self.xa - (pocket_radius if self.xa < screen_width/2 else -pocket_radius)/2.
            self.yf = (ya + yb)/2.
            self.wf = pocket_radius
            self.hf = np.abs(yb - ya) + eps
        else:
            self.xf = (xa + xb)/2.
            self.yf = self.ya - (pocket_radius if self.ya < screen_height/2 else -pocket_radius)/2.
            self.wf = np.abs(xb - xa) + eps
            self.hf = pocket_radius

    def on_draw(self):
        pass
        # Functional cushion
        arcade.draw_line(start_x=self.xa, start_y=self.ya,
                         end_x=self.xb, end_y=self.yb,
                         color=self.color)
        # Draw a filled rectangle
        arcade.draw_rectangle_filled(center_x=self.xf, center_y=self.yf, width=self.wf, height=self.hf, color=self.color)

    @staticmethod
    def _get_new_ball_direction(float velocity_x, float velocity_y,
                                float p_x, float p_y):
        cdef float n_x = -p_y
        cdef float n_y = p_x

        cdef float proj = 2 * (velocity_x * n_x + velocity_y * n_y)
        cdef velocity_x_proj = velocity_x - proj * n_x
        cdef velocity_y_proj = velocity_y - proj * n_y
        return velocity_x_proj, velocity_y_proj

    def get_new_ball_direction(self, ball):
        return LinearCushionElement._get_new_ball_direction(ball.velocity_x, ball.velocity_y, self.p_x, self.p_y)

    @staticmethod
    def _collides_with_ball(float o_x, float o_y,
                            float p_x, float p_y,
                            float p_norm,
                            float center_x, float center_y,
                            float radius):

        # Origin (line) to center (ball)
        cdef float v_x = o_x - center_x
        cdef float v_y = o_y - center_y

        # Existence of intersection point(s)
        cdef float discriminant = (p_x * v_x + p_y * v_y)**2 - v_x**2 - v_y**2 + radius**2
        if discriminant < 0:
            return False

        # Check where intersection occurs (could occur outside of line segment)
        cdef float d = -p_x * v_x - p_y * v_y

        return 0 < d < p_norm

    def collides_with_ball(self, ball):
        return LinearCushionElement._collides_with_ball(o_x=self.o_x, o_y=self.o_y,
                                                        p_x=self.p_x, p_y=self.p_y,
                                                        p_norm=self.p_norm,
                                                        center_x=ball.center_x, center_y=ball.center_y,
                                                        radius=ball.radius)


class CircularCushionElement(CushionElement):
    def __init__(self, xc, yc, r, phi_start, phi_end, color):
        super().__init__(color=color)

        self.xc, self.yc = xc, yc
        self.r = r
        self.phi_start = phi_start
        self.phi_end = phi_end
        self.color = color

    def on_draw(self):
        arcade.draw_arc_filled(center_x=self.xc, center_y=self.yc,
                               width=self.r, height=self.r,
                               start_angle=self.phi_start, end_angle=self.phi_end,
                               color=self.color)

    @staticmethod
    def _get_new_ball_direction(float center_x, float center_y,
                                float velocity_x, float velocity_y,
                                float xc, float yc):

        cdef float link_x = center_x - xc
        cdef float link_y = center_y - yc
        cdef float link_norm = math.sqrt(link_x**2 + link_y**2)
        link_x /= link_norm
        link_y /= link_norm
        cdef float proj = velocity_x * link_x + velocity_y * link_y
        cdef float velocity_x_proj = velocity_x - 2 * proj * link_x
        cdef float velocity_y_proj = velocity_y - 2 * proj * link_y
        return velocity_x_proj, velocity_y_proj

    def get_new_ball_direction(self, ball):

        return CircularCushionElement._get_new_ball_direction(center_x=ball.center_x, center_y=ball.center_y,
                                                              velocity_x=ball.velocity_x, velocity_y=ball.velocity_y,
                                                              xc=self.xc, yc=self.yc)

    @staticmethod
    def _collides_with_ball(float xc, float yc,
                            float center_x, float center_y,
                            float r, float radius):
        cdef float dx = xc - center_x
        cdef float dy = yc - center_y
        cdef float d_2 = dx * dx + dy * dy
        return d_2 < (r + radius)**2

    def collides_with_ball(self, ball):
        return CircularCushionElement._collides_with_ball(xc=self.xc, yc=self.yc,
                                                          center_x=ball.center_x, center_y=ball.center_y,
                                                          r=self.r, radius=ball.radius)


class Pocket(ABC):
    def __init__(self, color):
        self.color = color

    @abstractmethod
    def __contains__(self, ball):
        pass


class TriangularPocket(Pocket):
    def __init__(self, xa, ya, xb, yb, xc, yc, color):
        super().__init__(color)

        self.xa, self.ya = xa, ya
        self.xb, self.yb = xb, yb
        self.xc, self.yc = xc, yc

    def on_draw(self):
        arcade.draw_triangle_filled(x1=self.xa, y1=self.ya,
                                    x2=self.xb, y2=self.yb,
                                    x3=self.xc, y3=self.yc,
                                    color=self.color)

    @staticmethod
    def _contains_ball(float xa, float ya,
                       float xb, float yb,
                       float xc, float yc,
                       float center_x, float center_y):

        # Barycentric coordinates
        cdef float denominator = (yb - yc) * (xa - xc) + (xc - xb) * (ya - yc)
        cdef float u = ((yb - yc) * (center_x - xc) + (xc - xb) * (center_y - yc))/denominator
        cdef float v = ((yc - ya) * (center_x - xc) + (xa - xc) * (center_y - yc))/denominator
        cdef float w = 1 - u - v

        cdef bool u_in_range = 0 <= u <= 1
        cdef bool v_in_range = 0 <= v <= 1
        cdef bool w_in_range = 0 <= w <= 1

        cdef bool in_range = u_in_range and v_in_range and w_in_range

        return in_range

    def __contains__(self, ball):
        return TriangularPocket._contains_ball(xa=self.xa, ya=self.ya,
                                               xb=self.xb, yb=self.yb,
                                               xc=self.xc, yc=self.yc,
                                               center_x=ball.center_x, center_y=ball.center_y)


class RectangularPocket(Pocket):
    def __init__(self, xa, ya, width, height, color):
        super().__init__(color)

        self.xa, self.ya = xa, ya
        self.width = width
        self.height = height

    def on_draw(self):
        arcade.draw_rectangle_filled(center_x=self.xa, center_y=self.ya,
                                     width=self.width, height=self.height,
                                     color=self.color)

    @staticmethod
    def _contains_ball(float height, float width,
                       float xa, float ya,
                       float center_x, float center_y):
        cdef float w_half = width/2
        cdef float h_half = height/2
        cdef bool horizontally = xa - w_half <= center_x <= xa + w_half
        cdef bool vertically = ya - h_half <= center_y <= ya + h_half
        return horizontally and vertically

    def __contains__(self, ball):
        return RectangularPocket._contains_ball(height=self.height, width=self.width,
                                                xa=self.xa, ya=self.ya,
                                                center_x=ball.center_x, center_y=ball.center_y)


class PoolTable:
    def __init__(self, screen_width, screen_height, pocket_width, pocket_radius):

        self.pocket_width = pocket_width
        self.pocket_radius = pocket_radius

        # Definition of grid (used to make collision detection efficient)
        grid_step = 2 * pocket_width
        self.grid_x = np.arange(start=0, stop=screen_width, step=grid_step)
        self.grid_y = np.arange(start=0, stop=screen_height, step=grid_step)

        """
        TABLE GEOMETRY
        """
        screen_width = screen_width
        screen_height = screen_height
        color = BRITISH_RACING_GREEN
        pocket_color = BRONZE
        kwargs = {
            "pocket_radius": pocket_radius,
            "color": color,
            "screen_width": screen_width,
            "screen_height": screen_height
        }

        # Horizontal cushions
        lh = screen_width - 2 * pocket_radius
        self.cushion_bottom = LinearCushionElement(xa=2 * pocket_radius, ya=pocket_radius,
                                                   xb=lh, yb=pocket_radius,
                                                   **kwargs)
        self.cushion_top = LinearCushionElement(xa=2 * pocket_radius, ya=screen_height - pocket_radius,
                                                xb=lh, yb=screen_height - pocket_radius,
                                                **kwargs)

        # Vertical cushions
        lv = (screen_height - 6 * pocket_radius - pocket_width)/2.
        self.cushion_bottom_left = LinearCushionElement(xa=pocket_radius, ya=2 * pocket_radius,
                                                        xb=pocket_radius, yb=2 * pocket_radius + lv,
                                                        **kwargs)
        self.cushion_bottom_right = LinearCushionElement(xa=pocket_radius + lh, ya=2 * pocket_radius,
                                                         xb=pocket_radius + lh, yb=2 * pocket_radius + lv,
                                                         **kwargs)

        self.cushion_top_left = LinearCushionElement(xa=pocket_radius, ya=4 * pocket_radius + lv + pocket_width,
                                                     xb=pocket_radius, yb=screen_height - 2 * pocket_radius,
                                                     **kwargs)
        self.cushion_top_right = LinearCushionElement(xa=pocket_radius + lh, ya=4 * pocket_radius + lv + pocket_width,
                                                      xb=pocket_radius + lh, yb=screen_height - 2 * pocket_radius,
                                                      **kwargs)

        # Round cushions
        self.corner_bottom_west = CircularCushionElement(xc=2 * pocket_radius, yc=0,
                                                         r=pocket_radius,
                                                         phi_start=90, phi_end=180,
                                                         color=color)
        self.corner_bottom_east = CircularCushionElement(xc=lh, yc=0,
                                                         r=pocket_radius,
                                                         phi_start=0, phi_end=90,
                                                         color=color)

        self.corner_bottom_left_south = CircularCushionElement(xc=0, yc=2 * pocket_radius,
                                                               r=pocket_radius,
                                                               phi_start=270, phi_end=360,
                                                               color=color)
        self.corner_bottom_left_north = CircularCushionElement(xc=0, yc=2 * pocket_radius + lv,
                                                               r=pocket_radius,
                                                               phi_start=0, phi_end=90,
                                                               color=color)

        self.corner_top_left_south = CircularCushionElement(xc=0, yc=screen_height - 2 * pocket_radius - lv,
                                                            r=pocket_radius,
                                                            phi_start=270, phi_end=360,
                                                            color=color)
        self.corner_top_left_north = CircularCushionElement(xc=0, yc=screen_height - 2 * pocket_radius,
                                                            r=pocket_radius,
                                                            phi_start=0, phi_end=90,
                                                            color=color)

        self.corner_top_west = CircularCushionElement(xc=2 * pocket_radius, yc=screen_height,
                                                      r=pocket_radius,
                                                      phi_start=180, phi_end=270,
                                                      color=color)
        self.corner_top_east = CircularCushionElement(xc=lh, yc=screen_height,
                                                      r=pocket_radius,
                                                      phi_start=270, phi_end=360,
                                                      color=color)

        self.corner_top_right_south = CircularCushionElement(xc=screen_width, yc=screen_height - 2 * pocket_radius - lv,
                                                             r=pocket_radius,
                                                             phi_start=180, phi_end=270,
                                                             color=color)
        self.corner_top_right_north = CircularCushionElement(xc=screen_width, yc=screen_height - 2 * pocket_radius,
                                                             r=pocket_radius,
                                                             phi_start=90, phi_end=180,
                                                             color=color)

        self.corner_bottom_right_south = CircularCushionElement(xc=screen_width, yc=2 * pocket_radius + lv,
                                                                r=pocket_radius,
                                                                phi_start=90, phi_end=180,
                                                                color=color)
        self.corner_bottom_right_north = CircularCushionElement(xc=screen_width, yc=2 * pocket_radius,
                                                                r=pocket_radius,
                                                                phi_start=180, phi_end=270,
                                                                color=color)

        # All cushions
        self.cushions = [self.cushion_bottom,
                         self.cushion_bottom_left, self.cushion_bottom_right,
                         self.cushion_top_left, self.cushion_top_right,
                         self.cushion_top,
                         self.corner_bottom_west, self.corner_bottom_east,
                         self.corner_bottom_left_south, self.corner_bottom_left_north,
                         self.corner_top_left_south, self.corner_top_left_north,
                         self.corner_top_west, self.corner_top_east,
                         self.corner_top_right_south, self.corner_top_right_north,
                         self.corner_bottom_right_south, self.corner_bottom_right_north]

        # Pockets
        self.pocket_south_west = TriangularPocket(xa=0, ya=0,
                                                  xb=0, yb=2 * pocket_radius,
                                                  xc=2 * pocket_radius, yc=0,
                                                  color=pocket_color)
        self.pocket_south_east = TriangularPocket(xa=screen_width, ya=0,
                                                  xb=screen_width, yb=2 * pocket_radius,
                                                  xc=screen_width - 2 * pocket_radius, yc=0,
                                                  color=pocket_color)

        self.pocket_north_west = TriangularPocket(xa=0, ya=screen_height,
                                                  xb=2 * pocket_radius, yb=screen_height,
                                                  xc=0, yc=screen_height - 2 * pocket_radius,
                                                  color=pocket_color)
        self.pocket_north_east = TriangularPocket(xa=screen_width, ya=screen_height,
                                                  xb=screen_width, yb=screen_height - 2 * pocket_radius,
                                                  xc=screen_width - 2 * pocket_radius, yc=screen_height,
                                                  color=pocket_color)

        self.pocket_west = RectangularPocket(xa=0, ya=screen_height/2,
                                             width=pocket_radius/4, height=pocket_width,
                                             color=pocket_color)
        self.pocket_east = RectangularPocket(xa=screen_width, ya=screen_height/2,
                                             width=pocket_radius/4, height=pocket_width,
                                             color=pocket_color)

        self.pockets = [self.pocket_south_west, self.pocket_south_east,
                        self.pocket_north_west, self.pocket_north_east,
                        self.pocket_west, self.pocket_east]

    def on_draw(self):
        for pocket in self.pockets:
            pocket.on_draw()

        for cushion in self.cushions:
            cushion.on_draw()

    def is_in_pocket(self, ball):
        return any((ball in pocket) for pocket in self.pockets)

    def get_colliding_cushion(self, ball):

        for cushion in self.cushions:
            if cushion.collides_with_ball(ball):
                return cushion

        return None

    def assign_grid_cell(self, x_0, y_0):

        # Find interval
        i = bisect(self.grid_x, x_0) - 1
        j = bisect(self.grid_y, y_0) - 1

        return Cell(i=i, j=j)


class BilliardBall:
    def __init__(self, table, radius, color, reward):

        # Fixed
        self.table = table
        self.radius = radius
        self.four_times_radius_sq = 4 * radius**2
        self.color = color
        self.reward = reward

        # Dynamic
        self.is_on_table = True
        self.last_cushion = None
        self.last_ball = None
        self.center_x, self.center_y = None, None
        self.grid_cell = None

        # Past & present velocities
        self._velocity_x, self._velocity_y = 0, 0     # pixels/sec
        self._next_velocity_x, self._next_velocity_y = 0, 0

    @property
    def velocity_x(self):
        return self._velocity_x

    @property
    def velocity_y(self):
        return self._velocity_y

    @velocity_x.setter
    def velocity_x(self, value):
        self._next_velocity_x = value

    @velocity_y.setter
    def velocity_y(self, value):
        self._next_velocity_y = value

    @staticmethod
    def _is_moving(float velocity_x, float velocity_y):
        return velocity_x**2 + velocity_y**2 > MIN_VELOCITY_SQ

    def is_moving(self):
        return BilliardBall._is_moving(velocity_x=self.velocity_x, velocity_y=self.velocity_y)

    def exert_force(self, vx, vy):
        self._velocity_x = self._next_velocity_x = vx
        self._velocity_y = self._next_velocity_y = vy

    def on_draw(self):
        if self.is_on_table:
            arcade.draw_circle_filled(center_x=self.center_x, center_y=self.center_y,
                                      radius=self.radius, color=self.color)

    def update(self, dt):

        # Write back
        self._velocity_x = self._next_velocity_x
        self._velocity_y = self._next_velocity_y

        if self.is_moving():
            # Friction
            magnitude = np.linalg.norm([self._velocity_x, self._velocity_y], ord=2)
            self._velocity_x -= MU * G * dt * self._velocity_x/magnitude
            self._velocity_y -= MU * G * dt * self._velocity_y/magnitude

            # Apply velocity
            self.center_x += self._velocity_x * dt
            self.center_y += self._velocity_y * dt

            # Determine position within grid
            self.grid_cell = self.table.assign_grid_cell(self.center_x, self.center_y)
        else:
            self._velocity_x = 0
            self._velocity_y = 0

        self._next_velocity_x = self._velocity_x
        self._next_velocity_y = self._velocity_y

    @staticmethod
    def _get_new_ball_direction(float center_x, float center_y,
                                float other_center_x, float other_center_y,
                                float velocity_x, float velocity_y,
                                float other_velocity_x, float other_velocity_y):
        cdef float l_x = center_x - other_center_x
        cdef float l_y = center_y - other_center_y
        cdef float l_norm = math.sqrt(l_x * l_x + l_y * l_y)
        l_x /= l_norm
        l_y /= l_norm

        cdef float proj_1 = velocity_x * l_x + velocity_y * l_y
        cdef float v1_normal_x = proj_1 * l_x
        cdef float v1_normal_y = proj_1 * l_y

        cdef float proj_2 = other_velocity_x * l_x + other_velocity_y * l_y
        cdef float v2_normal_x = proj_2 * l_x
        cdef float v2_normal_y = proj_2 * l_y

        cdef float v_1_tangential_x = velocity_x - v1_normal_x
        cdef float v_1_tangential_y = velocity_y - v1_normal_y

        return v_1_tangential_x + v2_normal_x, v_1_tangential_y + v2_normal_y

    def get_new_ball_direction(self, other_ball):
        return BilliardBall._get_new_ball_direction(center_x=self.center_x, center_y=self.center_y,
                                                    other_center_x=other_ball.center_x, other_center_y=other_ball.center_y,
                                                    velocity_x=self.velocity_x, velocity_y=self.velocity_y,
                                                    other_velocity_x=other_ball.velocity_x, other_velocity_y=other_ball.velocity_y)

    @staticmethod
    def _is_colliding_with_ball(float center_x, float center_y,
                                float other_center_x, float other_center_y,
                                float four_times_radius_sq):
        cdef float d_x = center_x - other_center_x
        cdef float d_y = center_y - other_center_y
        cdef float d_2 = d_x * d_x + d_y * d_y

        return d_2 <= four_times_radius_sq

    def is_colliding_with_ball(self, other_ball):
        return BilliardBall._is_colliding_with_ball(center_x=self.center_x, center_y=self.center_y,
                                                    other_center_x=other_ball.center_x, other_center_y=other_ball.center_y,
                                                    four_times_radius_sq=self.four_times_radius_sq)

    def is_in_neighborhood(self, other_ball):
        horizontally = abs(self.grid_cell.i - other_ball.grid_cell.i) <= 1
        vertically = abs(self.grid_cell.j - other_ball.grid_cell.j) <= 1

        return horizontally and vertically

    def reset(self):
        self.__init__(radius=self.radius, color=self.color, reward=self.reward, table=self.table)


class BilliardGame(arcade.Window):
    def __init__(self, screen_width, screen_height, fps,
                 pocket_width,
                 num_balls, ball_radius,
                 standardize_rewards=True):

        # Unfortunately, arcade.Window does not expose the "visible" parameter in its constructor
        # Let us bypass this by calling pyglet.window.Window's constructor directly
        config = pyglet.gl.Config(major_version=3, minor_version=3, double_buffer=True)
        pyglet.window.Window.__init__(self, width=screen_width, height=screen_height, caption="Billard AI",
                                      resizable=False, config=config, visible=False)
        self.set_update_rate(1/fps)
        super().set_fullscreen(False)
        self.invalid = False
        set_window(self)
        set_viewport(0, self.width, 0, self.height)

        # Background color
        arcade.set_background_color(AIR_FORCE_BLUE)

        # Copy
        self.pocket_width = pocket_width
        self.pocket_radius = pocket_width / (2 * (math.sqrt(2) - 1))
        self.fps = fps

        # Game-specific
        self.num_balls = num_balls
        self.ball_radius = ball_radius

        # Table definition
        self.table = PoolTable(screen_width=screen_width, screen_height=screen_height,
                               pocket_width=pocket_width, pocket_radius=self.pocket_radius)

        # Scene construction
        self.balls = [BilliardBall(table=self.table, radius=ball_radius, color=WHITE_SMOKE, reward=None)]

        # Linearly interpolate colors
        start_color = np.array(RED)
        end_color = np.array(GREEN)
        x = np.linspace(start=0., stop=1., num=num_balls - 1)
        color = np.matmul(np.column_stack((1 - x, x)), np.row_stack((start_color, end_color)))

        # Reward definition
        self.rewards = x
        if standardize_rewards:
            self.rewards = 2 * self.rewards - 1

        for k in range(num_balls - 1):
            self.balls.append(BilliardBall(table=self.table,
                                           radius=ball_radius,
                                           color=color[k].astype(np.int32),
                                           reward=self.rewards[k]))

        # Assign ball positions
        self.assign_initial_ball_positions()

        # Keep track of steps
        self.step_counter = 0
        self.max_steps = (self.num_balls - 1) // 2
        self.done = False

    def assign_initial_ball_positions(self):

        # Enforce correct ball positions
        offset = self.pocket_radius + 2 * self.ball_radius

        for ball_id, ball in enumerate(self.balls):

            while True:
                # Proposal
                proposal_x = np.random.uniform(offset, self.width - offset)
                proposal_y = np.random.uniform(offset, self.height - offset)

                # Set position
                ball.center_x = proposal_x
                ball.center_y = proposal_y

                # Check if this causes a collision
                is_collision = False
                for other_ball in self.balls[:ball_id]:
                    is_collision = is_collision or ball.is_colliding_with_ball(other_ball)

                # Proceed with next ball if no collision has been detected
                if not is_collision:

                    # Assign initiali grid cell
                    ball.grid_cell = self.table.assign_grid_cell(x_0=proposal_x, y_0=proposal_y)
                    break

    def on_draw(self):
        self.clear()

        # Ball
        for ball in self.balls:
            ball.on_draw()

        # Table
        self.table.on_draw()

    def take_screenshot(self):
        # We have to execute the render loop a couple of times to guarantee consistency
        img = None
        while img is None or np.max(img) < 1:
            self.switch_to()
            self.dispatch_events()
            self.dispatch_event('on_draw')
            self.flip()
            img = np.array(arcade.draw_commands.get_image())[:, :, :3]    # [H, W, 3]
        img = np.transpose(img, axes=[2, 0, 1])     # [3, H, W]
        return img.astype(np.float32)/(2**8 - 1)

    def reset(self):

        # Reset balls
        for ball in self.balls:
            ball.reset()

        # Assign new positions
        self.assign_initial_ball_positions()

        # Reset step counter
        self.step_counter = 0
        self.done = False

        return Step(reward=0, new_observation=self.take_screenshot(), done=False)

    def step(self, phi_deg, velocity_magnitude):

        assert (self.step_counter < self.max_steps) and not self.done, "Episode is already over! Reset is necessary."

        # Radians
        phi_grad = np.pi/180. * phi_deg

        # Compute component-wise velocity
        velocity_x = np.cos(phi_grad) * velocity_magnitude
        velocity_y = np.sin(phi_grad) * velocity_magnitude

        # Exert force on cue ball
        self.balls[0].exert_force(vx=velocity_x, vy=velocity_y)

        # Count number of balls on the table before performing the shot
        balls_on_table_before = {ball for ball in self.balls[1:] if ball.is_on_table}

        # Update until movement stops or cue balls is pocketed
        delta_time = 1/self.fps
        while True:
            self.update(delta_time)

            # Cue ball
            if not self.balls[0].is_on_table:
                self.done = True
                return Step(reward=0, new_observation=None, done=True)

            # No balls are moving
            if all(not ball.is_moving() for ball in self.balls):
                break


        # Count number of balls on the table after performing the shot
        balls_on_table_after = {ball for ball in self.balls[1:] if ball.is_on_table}

        # Balls that were pocketed
        ball_pocketed = balls_on_table_before - balls_on_table_after

        # Compute reward
        reward = sum(ball.reward for ball in ball_pocketed)

        # Manage step counter
        self.step_counter += 1
        if self.step_counter == self.max_steps:
            self.done = True
            return Step(reward=reward,
                        new_observation=np.zeros(shape=[3, self.height, self.width], dtype=np.float32),
                        done=True)

        # Take screenshot
        screenshot = self.take_screenshot()

        # Episode is not finished
        return Step(reward=reward,
                    new_observation=screenshot,
                    done=False)

    def update(self, delta_time):

        # Ensure reproducibility (at the expense of more flickering)
        actual_delta_time = 1/self.fps

        # Game logic
        for ball in self.balls:
            if not ball.is_on_table:
                continue

            # Check if ball is in pocket
            if self.table.is_in_pocket(ball):
                ball.is_on_table = False
                ball.velocity_x = ball.velocity_y = 0
                continue

            # start = process_time_ns()
            # Check if a cushion-collision has occured
            colliding_cushion = self.table.get_colliding_cushion(ball)
            if colliding_cushion is not None and colliding_cushion is not ball.last_cushion:
                # Compute new direction
                velocity_x, velocity_y = colliding_cushion.get_new_ball_direction(ball)
                ball.velocity_x = velocity_x
                ball.velocity_y = velocity_y
                ball.last_cushion = colliding_cushion
                ball.last_ball = None
            # print("Took {} [msec] s2".format((process_time_ns() - start)/(10**6)))

            # Check if a ball-ball collision has occured
            # start = process_time_ns()
            for other_ball in self.balls:
                if ball is other_ball or not ball.is_in_neighborhood(other_ball):
                    continue

                if ball.is_colliding_with_ball(other_ball) and other_ball is not ball.last_ball:
                    velocity_x, velocity_y = ball.get_new_ball_direction(other_ball)
                    ball.velocity_x = velocity_x
                    ball.velocity_y = velocity_y
                    ball.last_ball = other_ball
                    ball.last_cushion = None
                    break
            # print("Took {} [msec] s3".format((process_time_ns() - start)/(10**6)))

        # Update ball position
        for ball in self.balls:
            ball.update(actual_delta_time)


