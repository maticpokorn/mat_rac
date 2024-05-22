import arcade
from typing import Optional
import numpy as np
from numpy import random
from queue import Queue
import dqn
from time import sleep
from random import randint
from PIL import Image

# Constants
SCREEN_WIDTH = 1000
SCREEN_HEIGHT = 600

# Gravity
GRAVITY = 3000

# Damping - Amount of speed lost per second
DEFAULT_DAMPING = 1.0
PLAYER_DAMPING = 0.4

# Friction between objects
PLAYER_FRICTION = 1.0
WALL_FRICTION = 70
DYNAMIC_ITEM_FRICTION = 0.6

# Mass (defaults to 1)
PLAYER_MASS = 2.0

# Keep player from going too fast
PLAYER_MAX_HORIZONTAL_SPEED = 300
PLAYER_MAX_VERTICAL_SPEED = 1600

# Force applied while on the ground
PLAYER_MOVE_FORCE_ON_GROUND = 8000

# Force applied when moving left/right in the air
PLAYER_MOVE_FORCE_IN_AIR = 900

# Strength of a jump
PLAYER_JUMP_IMPULSE = 2400

PLAYER_ATTACK_COOLDOWN = 80

BULLET_SCALING = 0.15
BULLET_SPEED = 10


SCREEN_TITLE = "Game"

# Constants used to scale our sprites from their original size
CHARACTER_SCALING = 0.15
TILE_SCALING = 1
GRASS_SCALING = 0.1

N_FEATURES = 10
ACTION_SPACE_SIZE = 16
PLAYER_MEMORY = 50
STATE_SPACE_SIZE = PLAYER_MEMORY * 10 * 2
EPSILON_REDUCE = 0.9


class Player:

    def __init__(
        self,
        sprite,
        left_key,
        right_key,
        up_key,
        attack_key,
        name="Unnamed",
        mode="Player",
        color=None,
        epsilon=None,
        attack_cooldown=PLAYER_ATTACK_COOLDOWN,
        mass=PLAYER_MASS,
        damping=PLAYER_DAMPING,
        friction=PLAYER_FRICTION,
        max_horizontal_speed=PLAYER_MAX_HORIZONTAL_SPEED,
        max_vertical_speed=PLAYER_MAX_VERTICAL_SPEED,
        move_force_on_ground=PLAYER_MOVE_FORCE_ON_GROUND,
        move_force_in_air=PLAYER_MOVE_FORCE_IN_AIR,
        jump_impulse=PLAYER_JUMP_IMPULSE,
    ):

        self.sprite = sprite
        self.color = color
        self.im_dict = {}
        i = 1
        for side in ["left", "right"]:
            for move in ["sit", "jump", "run1", "run2"]:
                im = Image.open(f"images/{color}/{side}/{move}.png")
                print(f"images/{color}/{side}/{move}.png")
                label = f"{color}_{side}_{move}"
                texture = arcade.Texture(label, im)
                self.sprite.append_texture(texture)
                self.im_dict[label] = i
                i += 1

        self.left_key = left_key
        self.right_key = right_key
        self.up_key = up_key
        self.attack_key = attack_key

        self.left_pressed = False
        self.right_pressed = False
        self.up_pressed = False
        self.attack = False

        self.facing_right = True

        self.name = name
        self.mode = mode
        self.score = 100

        self.attack_cooldown = attack_cooldown
        self.attack_wait = 0

        self.mass = mass
        self.damping = damping
        self.friction = friction
        self.max_horizontal_speed = max_horizontal_speed
        self.max_vertical_speed = max_vertical_speed
        self.move_force_on_ground = move_force_on_ground
        self.move_force_in_air = move_force_in_air
        self.jump_impulse = jump_impulse

        self.step_counter = 0
        # features: keys pressed, x and y coords, atk cooldown, score (8 features)
        self.memory = []
        self.agent = None
        self.old_state = np.zeros((PLAYER_MEMORY * 2, N_FEATURES))
        self.last_action = 0
        # if mode is set to "Agent", we initialize the DQN agent
        if mode == "Agent":
            self.agent = dqn.DQN(n_states=STATE_SPACE_SIZE, n_actions=ACTION_SPACE_SIZE)
            self.epsilon = epsilon
            self.cumulative_reward = 0
        elif mode == "Agent load":
            self.agent = dqn.DQN(n_states=STATE_SPACE_SIZE, n_actions=ACTION_SPACE_SIZE)
            self.agent.load("agent")
            self.epsilon = epsilon
            self.cumulative_reward = 0
        if mode == "Random":
            self.action = 0

        self.bullet_list = arcade.SpriteList()

    def is_on_ground(self, physics_engine):
        return physics_engine.is_on_ground(self.sprite)

    def on_key_press(self, key):
        if key == self.left_key:
            self.left_pressed = True
            self.facing_right = False
        elif key == self.right_key:
            self.right_pressed = True
            self.facing_right = True
        elif key == self.up_key:
            self.up_pressed = True
        elif key == self.attack_key:
            self.attack = True

    def on_key_release(self, key):
        if key == self.left_key:
            self.left_pressed = False
        elif key == self.right_key:
            self.right_pressed = False
        elif key == self.up_key:
            self.up_pressed = False
        elif key == self.attack_key:
            self.attack = False

    def apply_force(self, physics_engine):
        movements = (
            self.left_pressed,
            self.right_pressed,
            self.is_on_ground(physics_engine),
        )
        if self.right_pressed and not self.left_pressed:
            self.facing_right = True
        elif self.left_pressed and not self.right_pressed:
            self.facing_right = False
        horizontal_force = {
            (True, False, True): -self.move_force_on_ground,
            (False, True, True): self.move_force_on_ground,
            (True, True, True): 0,
            (True, True, False): 0,
            (True, False, False): -self.move_force_in_air,
            (False, True, False): self.move_force_in_air,
            (False, False, True): 0,
            (False, False, False): 0,
        }
        physics_engine.apply_force(self.sprite, (horizontal_force[movements], 0))

    def set_friction(self, physics_engine):
        if self.left_pressed or self.right_pressed:
            physics_engine.set_friction(self.sprite, 0)
        else:
            physics_engine.set_friction(self.sprite, 1.0)

    def attempt_jump(self, physics_engine):
        jump = self.is_on_ground(physics_engine) and self.up_pressed
        if jump:
            impulse = (0, self.jump_impulse)
            physics_engine.apply_impulse(self.sprite, impulse)

    def attempt_attack(self, opponent):
        dist = np.sqrt(
            (self.sprite.center_x - opponent.sprite.center_x) ** 2
            + (self.sprite.center_x - opponent.sprite.center_x) ** 2
        )
        # FIRE BULLET OR ATTACK FROM CLOSE UP
        bullet_fired = False
        if self.attack and self.attack_wait == 0:
            if dist < 50:
                opponent.score -= 10

            else:
                if self.facing_right:
                    bullet = arcade.Sprite(
                        "images/fireball/right/fireball1.png", BULLET_SCALING
                    )
                    im = Image.open("images/fireball/right/fireball2.png")
                    texture = arcade.Texture("fireball2", im)
                    bullet.append_texture(texture)
                else:
                    bullet = arcade.Sprite(
                        "images/fireball/left/fireball1.png", BULLET_SCALING
                    )
                    im = Image.open("images/fireball/left/fireball2.png")
                    texture = arcade.Texture("fireball2_left", im)
                    bullet.append_texture(texture)
                bullet.center_x = self.sprite.center_x
                bullet.center_y = self.sprite.center_y
                bullet.angle = 0
                bullet.change_x = BULLET_SPEED if self.facing_right else -BULLET_SPEED
                bullet.change_y = 0
                bullet_fired = True
                self.bullet_list.append(bullet)

            self.attack_wait = self.attack_cooldown

        self.attack_wait = max(0, self.attack_wait - 1)

        return bullet_fired

    def on_update(self, physics_engine, opponent):
        self.bullet_list.update()
        self.attempt_jump(physics_engine)
        self.apply_force(physics_engine)
        self.set_friction(physics_engine)
        bullet_fired = self.attempt_attack(opponent)

        to_remove = []
        for bullet in self.bullet_list:
            if arcade.check_for_collision(bullet, opponent.sprite):
                to_remove.append(bullet)
                opponent.score -= 10
        for bullet in to_remove:
            self.bullet_list.remove(bullet)

        # update memory
        state = [
            int(self.left_pressed),
            int(self.right_pressed),
            int(self.up_pressed),
            int(self.attack),
            int(bullet_fired),
            int(self.facing_right),
            self.sprite.center_x / SCREEN_WIDTH,
            self.sprite.center_y / SCREEN_HEIGHT,
            self.attack_wait / PLAYER_ATTACK_COOLDOWN,
            self.score / 100,
        ]
        if len(self.memory) == PLAYER_MEMORY:
            self.memory.append(state)
            self.memory = self.memory[1:]
            if self.mode in ["Agent", "Agent load"]:
                self.step(opponent)
        else:
            self.memory.append(state)

        if self.mode == "Rule based":
            self.set_rule_based_movements(opponent)

        if self.mode == "Random":
            if self.step_counter % 20 == 0:
                self.action = np.random.randint(ACTION_SPACE_SIZE)
            self.set_movements(self.interpret_action(self.action))

        self.step_counter += 1
        self.set_image()

    def set_movements_randomly(self, change_probability=0.01):
        if random.rand() < change_probability:
            self.left_pressed = not self.left_pressed
        if random.rand() < change_probability:
            self.right_pressed = not self.right_pressed
        if random.rand() < change_probability:
            self.up_pressed = not self.up_pressed
        if random.rand() < change_probability:
            self.attack = not self.attack
        self.facing_right = self.right_pressed

    def set_rule_based_movements(self, opponent):
        m = np.array(opponent.memory)

        self.up_pressed = False
        for bullet in opponent.bullet_list:
            if (
                abs(bullet.center_x - self.sprite.center_x) < 100
                and abs(bullet.center_y - self.sprite.center_y) < 30
            ):
                self.up_pressed = True

        if self.attack_wait < 1:
            self.attack = True
        else:
            self.attack = False

        self.left_pressed = False
        self.right_pressed = False

        if opponent.sprite.center_x > self.sprite.center_x and not self.facing_right:
            self.left_pressed = False
            self.right_pressed = True

        if opponent.sprite.center_x < self.sprite.center_x and self.facing_right:
            self.left_pressed = True
            self.right_pressed = False

        if 0.2 * SCREEN_WIDTH < self.sprite.center_x < 0.5 * SCREEN_WIDTH:
            self.left_pressed = True
            self.right_pressed = False

        if 0.5 * SCREEN_WIDTH < self.sprite.center_x < 0.8 * SCREEN_WIDTH:
            self.left_pressed = False
            self.right_pressed = True

        if (
            opponent.sprite.center_x > SCREEN_WIDTH / 2
            and self.sprite.center_x > SCREEN_WIDTH / 2
        ):
            self.left_pressed = True
            self.right_pressed = False

        if (
            opponent.sprite.center_x < SCREEN_WIDTH / 2
            and self.sprite.center_x < SCREEN_WIDTH / 2
        ):
            self.left_pressed = False
            self.right_pressed = True

        if self.sprite.center_x < 0.05 * SCREEN_WIDTH:
            self.left_pressed = False
            self.right_pressed = True

        if self.sprite.center_x > 0.95 * SCREEN_WIDTH:
            self.left_pressed = True
            self.right_pressed = False

    def set_image(self):
        side = "right" if self.facing_right else "left"

        if self.up_pressed:
            move = "jump"
        elif self.right_pressed or self.left_pressed:
            if self.step_counter % 7 < 4:
                move = "run1"
            else:
                move = "run2"
        else:
            move = "sit"

        self.sprite.texture = self.sprite.textures[
            self.im_dict[f"{self.color}_{side}_{move}"]
        ]

        for bullet in self.bullet_list:
            bullet.texture = bullet.textures[int(self.step_counter % 16 > 7)]

    def step(self, opponent):
        if len(self.memory) < 2:
            reward = 0
        else:
            reward = (
                +100 * (self.memory[-1][-1] - self.memory[-2][-1])
                - 100 * (opponent.memory[-1][-1] - opponent.memory[-2][-1])
                - 0 * ((self.memory[-1][-4] - self.memory[-2][-4]) < 0.0005)
                + 0 * ((self.memory[-1][-4] - self.memory[-2][-4]) > 0.0005)
                - 1
                * int(
                    (self.sprite.center_x / SCREEN_WIDTH) > 0.9
                    or (self.sprite.center_x / SCREEN_WIDTH) < 0.1
                )
                - 0
                + 0 * (self.score - opponent.score)
                + 1000 * (opponent.score == 0)
            )
        self.cumulative_reward += reward
        # print(reward)
        state = np.concatenate(
            (np.array(self.memory), np.array(opponent.memory)), axis=0
        )
        # print(state.shape)
        # print(state.flatten().shape)
        self.agent.store_transition(
            self.old_state.flatten(), self.last_action, reward, state.flatten()
        )

        if self.agent.memory_counter > 5_000:
            self.agent.learn()

        action = self.agent.choose_action(state.flatten(), self.epsilon)
        self.set_movements(self.interpret_action(action))
        self.old_state = state
        self.last_action = action

    def interpret_action(self, action):
        binary_string = bin(action)[2:]
        # Pad with zeros to ensure desired length
        binary_string = binary_string.zfill(4)
        # Convert binary string to list of bits
        return [int(bit) for bit in binary_string]

    def set_movements(self, movements):
        self.left_pressed = movements[0]
        self.right_pressed = movements[1]
        self.up_pressed = movements[2]
        self.attack = movements[3]


class GameOverView(arcade.View):
    """View to show when game is over"""

    def __init__(self, epsilon):
        """This is run once when we switch to this view"""
        self.epsilon = epsilon
        super().__init__()
        self.texture = arcade.load_texture("images/game_over.png")

        # Reset the viewport, necessary if we have a scrolling game and we need
        # to reset the viewport back to the start so we can see what we draw.
        arcade.set_viewport(0, SCREEN_WIDTH - 1, 0, SCREEN_HEIGHT - 1)
        sleep(3)
        # if self.epsilon < 0.5:
        #    self.epsilon = 0
        game_view = GameView(epsilon=self.epsilon * EPSILON_REDUCE)
        game_view.setup()
        self.window.show_view(game_view)
        self.window.set_update_rate(1 / 30)

    def on_draw(self):
        """Draw this view"""
        self.clear()
        self.texture.draw_sized(
            SCREEN_WIDTH / 2, SCREEN_HEIGHT / 2, SCREEN_WIDTH, SCREEN_HEIGHT
        )

    def on_mouse_press(self, _x, _y, _button, _modifiers):
        """If the user presses the mouse button, re-start the game."""
        game_view = GameView(epsilon=self.epsilon * EPSILON_REDUCE)
        game_view.setup()
        self.window.show_view(game_view)


class GameMenuView(arcade.View):
    def __init__(self, mode1, mode2):
        super().__init__()
        # Reset the viewport, necessary if we have a scrolling game and we need
        # to reset the viewport back to the start so we can see what we draw.
        arcade.set_viewport(0, SCREEN_WIDTH - 1, 0, SCREEN_HEIGHT - 1)
        arcade.set_background_color(arcade.csscolor.WHITE)
        self.mode1 = mode1
        self.mode2 = mode2
        self.texture = arcade.load_texture("images/title.png")
        self.p1 = arcade.load_texture("images/player.png")
        self.p2 = arcade.load_texture("images/player.png")
        self.c1 = arcade.load_texture("images/computer.png")
        self.c2 = arcade.load_texture("images/computer.png")
        self.p1_select = True
        self.p2_select = True
        self.p1_decide = False
        self.p2_decide = False
        self.counter = 0

    def setup(self):
        a = 0

    def on_update(self, delta_time: float):
        if self.p1_select:
            self.c1 = arcade.load_texture("images/computer.png")
            if self.p1_decide or self.counter % 64 > 31:
                self.p1 = arcade.load_texture("images/player_select.png")
            else:
                self.p1 = arcade.load_texture("images/player.png")
        else:
            self.p1 = arcade.load_texture("images/player.png")
            if self.p1_decide or self.counter % 64 > 31:
                self.c1 = arcade.load_texture("images/computer_select.png")
            else:
                self.c1 = arcade.load_texture("images/computer.png")
        if self.p2_select:
            self.c2 = arcade.load_texture("images/computer.png")
            if self.counter % 64 > 31:
                self.p2 = arcade.load_texture("images/player_select.png")
            else:
                self.p2 = arcade.load_texture("images/player.png")
        else:
            self.p2 = arcade.load_texture("images/player.png")
            if self.counter % 64 > 31:
                self.c2 = arcade.load_texture("images/computer_select.png")
            else:
                self.c2 = arcade.load_texture("images/computer.png")

        if self.p2_decide and self.p2_decide:
            mode1 = "Player" if self.p1_select else "Rule based"
            mode2 = "Player" if self.p2_select else "Rule based"
            game_view = GameView(epsilon=0, mode1=mode1, mode2=mode2)
            game_view.setup()
            self.window.show_view(game_view)

        self.counter += 1

    def on_draw(self):
        """Draw this view"""
        scale = 0.1

        self.clear()
        self.texture.draw_sized(
            SCREEN_WIDTH / 2, SCREEN_HEIGHT / 2, SCREEN_WIDTH / 2, SCREEN_HEIGHT / 2
        )
        self.p1.draw_scaled(SCREEN_WIDTH / 4, SCREEN_HEIGHT / 4, scale=scale)
        self.c1.draw_scaled(SCREEN_WIDTH / 4, SCREEN_HEIGHT / 5, scale=scale)
        self.p2.draw_scaled(SCREEN_WIDTH * (3 / 4), SCREEN_HEIGHT / 4, scale=scale)
        self.c2.draw_scaled(SCREEN_WIDTH * (3 / 4), SCREEN_HEIGHT / 5, scale=scale)

        t = "Player 1: W and S to toggle, Q to select.\n Player 2: UP and DOWN to toggle, M to select."
        arcade.draw_text(
            t,
            20,
            10,
            arcade.csscolor.BLACK,
            10,
        )

    def on_key_press(self, symbol: int, modifiers: int):
        if symbol in [arcade.key.W, arcade.key.S]:
            self.p1_select = not self.p1_select
        if symbol in [arcade.key.UP, arcade.key.DOWN]:
            self.p2_select = not self.p2_select
        if symbol == arcade.key.Q:
            self.p1_decide = True
        if symbol == arcade.key.M:
            self.p2_decide = True


class GameView(arcade.View):
    """
    Main application class.
    """

    def __init__(self, epsilon, mode1, mode2):

        # Call the parent class and set up the window
        super().__init__()
        self.epsilon = epsilon
        self.mode1 = mode1
        self.mode2 = mode2

        # Our Scene Object
        self.scene = None

        # Player
        self.player1: Optional[Player] = None
        self.player2: Optional[Player] = None

        # Player sprite
        self.player_sprite1: Optional[arcade.Sprite] = None
        self.player_sprite2: Optional[arcade.Sprite] = None

        # Sprite lists we need
        self.player_list: Optional[arcade.SpriteList] = None
        self.wall_list: Optional[arcade.SpriteList] = None
        self.bullet_list: Optional[arcade.SpriteList] = None
        self.item_list: Optional[arcade.SpriteList] = None

        # Track the current state of what key is pressed
        self.left_pressed: bool = False
        self.right_pressed: bool = False
        self.up_pressed: bool = False

        # A Camera that can be used to draw GUI elements
        self.gui_camera = None

        # Keep track of the score
        self.score = 0

        arcade.set_background_color(arcade.csscolor.LIGHT_BLUE)

    def setup(self):
        """Set up the game here. Call this function to restart the game."""

        # Initialize Scene
        self.scene = arcade.Scene()

        # Create the Sprite lists
        self.scene.add_sprite_list("Player")
        self.scene.add_sprite_list("Walls", use_spatial_hash=True)

        # Create the Sprite lists
        self.player_list = arcade.SpriteList()
        self.wall_list = arcade.SpriteList(use_spatial_hash=True)

        rabbit1 = "images/red/right/sit.png"
        rabbit2 = "images/blue/left/sit.png"
        gray_square = "images/gray_square.png"
        grass = "images/grass/grass.png"
        self.player_sprite1 = arcade.Sprite(rabbit1, CHARACTER_SCALING)
        # self.player_sprite1.center_x = SCREEN_WIDTH / 8
        self.player_sprite1.center_x = randint(20, SCREEN_WIDTH - 20)
        self.player_sprite1.center_y = SCREEN_HEIGHT / 8
        self.player_list.append(self.player_sprite1)
        self.scene.add_sprite("Player1", self.player_sprite1)

        self.player_sprite2 = arcade.Sprite(rabbit2, CHARACTER_SCALING)
        # self.player_sprite2.center_x = SCREEN_WIDTH * (1 - 1 / 8)
        self.player_sprite2.center_x = randint(20, SCREEN_WIDTH - 20)
        self.player_sprite2.center_y = SCREEN_HEIGHT / 8
        self.player_list.append(self.player_sprite2)
        self.scene.add_sprite("Player2", self.player_sprite2)

        # Create the ground
        # This shows using a loop to place multiple sprites horizontally
        for x in range(0, SCREEN_WIDTH, 25):
            wall = arcade.Sprite(grass, GRASS_SCALING)
            wall.center_x = x
            wall.center_y = 0
            self.wall_list.append(wall)
            self.scene.add_sprite("Walls", wall)

        for x in range(200, SCREEN_WIDTH - 200, 25):
            wall = arcade.Sprite(grass, GRASS_SCALING)
            wall.center_x = x
            wall.center_y = SCREEN_HEIGHT / 5
            self.wall_list.append(wall)
            self.scene.add_sprite("Walls", wall)

        for y in range(0, SCREEN_HEIGHT + 100, 50):
            wall = arcade.Sprite(gray_square, TILE_SCALING)
            wall.center_x = 0
            wall.center_y = y
            self.wall_list.append(wall)
            self.scene.add_sprite("Walls", wall)

            wall = arcade.Sprite(gray_square, TILE_SCALING)
            wall.center_x = SCREEN_WIDTH
            wall.center_y = y
            self.wall_list.append(wall)
            self.scene.add_sprite("Walls", wall)

        self.physics_engine = arcade.PymunkPhysicsEngine(
            gravity=(0, -GRAVITY), damping=1.0, maximum_incline_on_ground=0.708
        )
        self.physics_engine.add_sprite(
            self.player_sprite1,
            friction=PLAYER_FRICTION,
            mass=PLAYER_MASS,
            moment=arcade.PymunkPhysicsEngine.MOMENT_INF,
            collision_type="player",
            max_horizontal_velocity=PLAYER_MAX_HORIZONTAL_SPEED,
            max_vertical_velocity=PLAYER_MAX_VERTICAL_SPEED,
        )
        self.physics_engine.add_sprite(
            self.player_sprite2,
            friction=PLAYER_FRICTION,
            mass=PLAYER_MASS,
            moment=arcade.PymunkPhysicsEngine.MOMENT_INF,
            collision_type="player",
            max_horizontal_velocity=PLAYER_MAX_HORIZONTAL_SPEED,
            max_vertical_velocity=PLAYER_MAX_VERTICAL_SPEED,
        )

        self.physics_engine.add_sprite_list(
            self.wall_list,
            friction=WALL_FRICTION,
            collision_type="wall",
            body_type=arcade.PymunkPhysicsEngine.STATIC,
        )

        self.player1 = Player(
            self.player_sprite1,
            left_key=arcade.key.A,
            right_key=arcade.key.D,
            up_key=arcade.key.W,
            attack_key=arcade.key.Q,
            name="Player 1",
            color="red",
            mode=self.mode1,
        )

        self.player2 = Player(
            self.player_sprite2,
            left_key=arcade.key.LEFT,
            right_key=arcade.key.RIGHT,
            up_key=arcade.key.UP,
            attack_key=arcade.key.M,
            name="Player 2",
            color="blue",
            mode=self.mode2,
            epsilon=self.epsilon,
        )

        # Set up the GUI Camera
        self.gui_camera = arcade.Camera(SCREEN_WIDTH, SCREEN_HEIGHT)

        # Keep track of the score
        self.score = 0

    def on_key_press(self, key, modifiers):
        """Called whenever a key is pressed."""
        self.player1.on_key_press(key)
        self.player2.on_key_press(key)

    def on_key_release(self, key, modifiers):
        """Called when the user releases a key."""
        self.player1.on_key_release(key)
        self.player2.on_key_release(key)

    def on_update(self, delta_time):
        """Movement and game logic"""
        self.player1.on_update(self.physics_engine, opponent=self.player2)
        # self.player2.set_movements_randomly()
        self.player2.on_update(self.physics_engine, opponent=self.player1)
        self.physics_engine.step()
        if (
            self.player1.score <= 0
            or self.player2.score <= 0
            or self.player1.step_counter > 10_000
            or self.player1.sprite.center_y < 0
            or self.player2.sprite.center_y < 0
        ):
            menu_view = GameMenuView(
                mode1=self.mode1,
                mode2=self.mode2,
            )
            menu_view.setup()
            self.window.show_view(menu_view)

    def on_draw(self):
        """Draw everything"""
        self.clear()
        self.wall_list.draw()
        # self.bullet_list.draw()
        # self.item_list.draw()
        self.player_list.draw()
        self.player1.bullet_list.draw()
        self.player2.bullet_list.draw()

        # Activate the GUI camera before drawing GUI elements
        self.gui_camera.use()

        # Draw our score on the screen, scrolling it with the viewport
        score_text1 = f"Health: {self.player1.score}"
        score_text2 = f"Health: {self.player2.score}"
        # reward = f"cumulative reward: {round(self.player2.cumulative_reward,2)}"
        arcade.draw_text(
            score_text1,
            10,
            10,
            arcade.csscolor.WHITE,
            18,
        )
        arcade.draw_text(
            score_text2,
            SCREEN_WIDTH - 200,
            10,
            arcade.csscolor.WHITE,
            18,
        )
        """arcade.draw_text(
            reward,
            SCREEN_WIDTH / 2 - 200,
            10,
            arcade.csscolor.WHITE,
            18,
        )"""


def main(epsilon, mode1, mode2):
    """Main function"""
    window = arcade.Window(SCREEN_WIDTH, SCREEN_HEIGHT, SCREEN_TITLE)
    start_view = GameMenuView(mode1, mode2)
    window.show_view(start_view)
    start_view.setup()
    arcade.run()


if __name__ == "__main__":
    epsilon = 1
    main(epsilon, "Rule based", "Player")
