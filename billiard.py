import arcade
from arcade.window_commands import set_viewport
from arcade.window_commands import set_window
import pyglet
from argparse import ArgumentParser
import numpy as np
from abc import ABC, abstractmethod
from collections import namedtuple
import matplotlib.pyplot as plt
from skimage import io

Step = namedtuple("Step", ["reward", "new_observation", "done"])

# Environment constants
MU = 2.
G = 9.81
MIN_VELOCITY = 10

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
        self.o = np.array([xa, ya])

        # Normalized direction vector
        p = np.array([xb - xa, yb - ya])
        self.p_norm = np.linalg.norm(p, ord=2)
        self.p = p / self.p_norm

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

    def get_new_ball_direction(self, ball):
        velocity = np.array([ball.velocity_x, ball.velocity_y])
        n = np.array([-self.p[1], self.p[0]])
        return velocity - 2 * np.dot(velocity, n) * n

    def collides_with_ball(self, ball):
        """
        c.f. https://en.wikipedia.org/wiki/Line%E2%80%93sphere_intersection
        :param ball:
        :return:
        """

        # Ball coordinates
        xc, yc, rc = ball.center_x, ball.center_y, ball.radius
        c = np.array([xc, yc])

        # Origin (line) to center (ball)
        v = self.o - c

        # Existence of intersection point(s)
        discriminant = np.square(np.dot(self.p, v)) - np.dot(v, v) + rc * rc

        if discriminant < 0.:
            return False

        # Check where intersection occurs (could occur outside of line segment)
        d = -np.dot(self.p, v)

        return 0. < d < self.p_norm


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

    def get_new_ball_direction(self, ball):

        # Extract coordinates
        velocity = np.array([ball.velocity_x, ball.velocity_y])
        cushion_center = np.array([self.xc, self.yc])
        ball_center = np.array([ball.center_x, ball.center_y])

        # Direction of intersection (relative to cushion center)
        link = ball_center - cushion_center
        link /= np.linalg.norm(link, ord=2)

        return velocity - 2 * np.dot(velocity, link) * link

    def collides_with_ball(self, ball):
        cushion_center = np.array([self.xc, self.yc])
        ball_center = np.array([ball.center_x, ball.center_y])
        dist = np.linalg.norm(cushion_center - ball_center, ord=2)
        return dist < self.r + ball.radius


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
        pass
        # arcade.draw_triangle_filled(x1=self.xa, y1=self.ya,
        #                             x2=self.xb, y2=self.yb,
        #                             x3=self.xc, y3=self.yc,
        #                             color=self.color)

    def __contains__(self, ball):
        x1, y1 = self.xa, self.ya
        x2, y2 = self.xb, self.yb
        x3, y3 = self.xc, self.yc
        x, y = ball.center_x, ball.center_y

        # Barycentric coordinates
        denominator = (y2 - y3) * (x1 - x3) + (x3 - x2) * (y1 - y3)
        u = ((y2 - y3) * (x - x3) + (x3 - x2) * (y - y3))/denominator
        v = ((y3 - y1) * (x - x3) + (x1 - x3) * (y - y3))/denominator
        w = 1 - u - v

        return 0 <= u <= 1 and 0 <= v <= 1 and 0 <= w <= 1


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

    def __contains__(self, ball):
        horizontally = self.xa - self.width/2 <= ball.center_x <= self.xa + self.width/2
        vertically = self.ya - self.height/2 <= ball.center_y <= self.ya + self.height/2
        return horizontally and vertically


class PoolTable:
    def __init__(self, screen_width, screen_height, pocket_width, pocket_radius):

        self.pocket_width = pocket_width
        self.pocket_radius = pocket_radius

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


class BilliardBall:
    def __init__(self, radius, color, reward):

        # Fixed
        self.radius = radius
        self.color = color
        self.reward = reward

        # Dynamic
        self.is_on_table = True
        self.last_cushion = None
        self.last_ball = None
        self.center_x, self.center_y = None, None

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

    def is_moving(self):
        return np.linalg.norm([self.velocity_x, self.velocity_y], ord=2) > MIN_VELOCITY

    def exert_force(self, vx, vy):
        self._velocity_x = self._next_velocity_x = vx
        self._velocity_y = self._next_velocity_y = vy

    def on_draw(self):
        if self.is_on_table:
            arcade.draw_circle_filled(center_x=self.center_x, center_y=self.center_y,
                                      radius=self.radius, color=self.color)

    def update(self, dt):

        # Friction
        magnitude = np.linalg.norm([self._next_velocity_x, self._next_velocity_y], ord=2)
        if magnitude > MIN_VELOCITY:
            self._next_velocity_x -= MU * G * dt * self._next_velocity_x/magnitude
            self._next_velocity_y -= MU * G * dt * self._next_velocity_y/magnitude
        else:
            self._next_velocity_x = 0
            self._next_velocity_y = 0

        # Write back
        self._velocity_x = self._next_velocity_x
        self._velocity_y = self._next_velocity_y

        self.center_x += self.velocity_x * dt
        self.center_y += self.velocity_y * dt

    def get_new_ball_direction(self, other_ball):

        # Extract coordinates
        velocity = np.array([self.velocity_x, self.velocity_y])
        other_velocity = np.array([other_ball.velocity_x, other_ball.velocity_y])
        ball_center = np.array([self.center_x, self.center_y])
        other_ball_center = np.array([other_ball.center_x, other_ball.center_y])

        # Direction of intersection (relative to cushion center)
        link = ball_center - other_ball_center
        link /= np.linalg.norm(link, ord=2)

        # Project velocities on normal component
        v1_normal = np.dot(velocity, link) * link
        v2_normal = np.dot(other_velocity, link) * link

        # Project velocities on tangential component
        v1_tangential = velocity - v1_normal

        return v1_tangential + v2_normal

    def is_colliding_with_ball(self, other_ball):

        if self is other_ball:
            return False

        u = np.array([self.center_x, self.center_y])
        v = np.array([other_ball.center_x, other_ball.center_y])
        ru = self.radius
        rv = other_ball.radius

        # Center-to-center distance
        d = np.linalg.norm(u - v, ord=2)

        return d <= ru + rv

    def reset(self):
        self.__init__(radius=self.radius, color=self.color, reward=self.reward)


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

        # Copy
        self.pocket_width = pocket_width
        self.pocket_radius = pocket_width / (2 * (np.sqrt(2) - 1))
        self.fps = fps

        # Game-specific
        self.num_balls = num_balls
        self.ball_radius = ball_radius

        # Scene construction
        self.balls = [BilliardBall(radius=ball_radius, color=WHITE_SMOKE, reward=None)]

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
            self.balls.append(BilliardBall(radius=ball_radius, color=color[k].astype(np.int32), reward=self.rewards[k]))

        self.table = PoolTable(screen_width=screen_width, screen_height=screen_height,
                               pocket_width=pocket_width, pocket_radius=self.pocket_radius)

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
        return img

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
                        new_observation=None,
                        done=True)

        # Episode is not finished
        return Step(reward=reward,
                    new_observation=self.take_screenshot(),
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

            # Check if a cushion-collision has occured
            colliding_cushion = self.table.get_colliding_cushion(ball)
            if colliding_cushion is not None and colliding_cushion is not ball.last_cushion:
                # Compute new direction
                velocity_x, velocity_y = colliding_cushion.get_new_ball_direction(ball)
                ball.velocity_x = velocity_x
                ball.velocity_y = velocity_y
                ball.last_cushion = colliding_cushion
                ball.last_ball = None

            # Check if a ball-ball collision has occured
            for other_ball in self.balls:
                if ball is other_ball:
                    continue

                if ball.is_colliding_with_ball(other_ball) and other_ball is not ball.last_ball:
                    velocity_x, velocity_y = ball.get_new_ball_direction(other_ball)
                    ball.velocity_x = velocity_x
                    ball.velocity_y = velocity_y
                    ball.last_ball = other_ball
                    ball.last_cushion = None
                    break
        for ball in self.balls:
            # Update ball position
            ball.update(actual_delta_time)


def main():
    # Instantiate parser
    parser = ArgumentParser()

    parser.add_argument("--screen_width",
                        help="Width of the screen",
                        type=int,
                        default=600)

    parser.add_argument("--screen_height",
                        help="Height of the screen",
                        type=int,
                        default=900)

    parser.add_argument("--pocket_width",
                        help="How wide each pocket is",
                        type=float,
                        default=40.)

    parser.add_argument("--num_balls",
                        help="How many balls to use",
                        type=int,
                        default=30)

    parser.add_argument("--ball_radius",
                        help="Radius of the balls",
                        type=float,
                        default=10.)

    parser.add_argument("--fps",
                        help="How many frames should be rendered per second",
                        type=float,
                        default=60)

    parser.add_argument("--interactive", "-i",
                        help="Whether to perform a test run",
                        action="store_true",
                        default=False)

    # Parse
    args = parser.parse_args()

    # Input assertions
    assert args.screen_width > 0 and args.screen_height > 0, "Wrong dimension specification!"
    assert args.pocket_width >= 3 * args.ball_radius, "Pockets are not wide enough!"

    # Initialize game
    args = vars(args)
    interactive = args.pop("interactive")
    game = BilliardGame(**args)
    arcade.set_background_color(AIR_FORCE_BLUE)

    if interactive:
        # Make visible
        game.set_visible(True)

        # Apply force on cue ball
        game.balls[0].exert_force(vx=250, vy=250)

        # Render loop
        arcade.run()
    else:
        step = game.step(phi_deg=45, velocity_magnitude=300)
        io.imshow(step.new_observation)
        plt.show()


if __name__ == "__main__":
    main()
