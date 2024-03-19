import arcade
from typing import Optional

# Constants
SCREEN_WIDTH = 600
SCREEN_HEIGHT = 400

# Gravity
GRAVITY = 1500

# Damping - Amount of speed lost per second
DEFAULT_DAMPING = 1.0
PLAYER_DAMPING = 0.4

# Friction between objects
PLAYER_FRICTION = 1.0
WALL_FRICTION = 0.7
DYNAMIC_ITEM_FRICTION = 0.6

# Mass (defaults to 1)
PLAYER_MASS = 2.0

# Keep player from going too fast
PLAYER_MAX_HORIZONTAL_SPEED = 450
PLAYER_MAX_VERTICAL_SPEED = 1600

# Force applied while on the ground
PLAYER_MOVE_FORCE_ON_GROUND = 8000

# Force applied when moving left/right in the air
PLAYER_MOVE_FORCE_IN_AIR = 900

# Strength of a jump
PLAYER_JUMP_IMPULSE = 1800


SCREEN_TITLE = "Game"

# Constants used to scale our sprites from their original size
CHARACTER_SCALING = 0.1
TILE_SCALING = 0.5


class Player:

    def __init__(
        self,
        sprite,
        left_key,
        right_key,
        up_key,
        name="Unnamed",
        mass=PLAYER_MASS,
        damping=PLAYER_DAMPING,
        friction=PLAYER_FRICTION,
        max_horizontal_speed=PLAYER_MAX_HORIZONTAL_SPEED,
        max_vertical_speed=PLAYER_MAX_VERTICAL_SPEED,
        move_force_on_ground=PLAYER_MOVE_FORCE_ON_GROUND,
        move_force_in_air=PLAYER_MOVE_FORCE_IN_AIR,
        jump_impulse=PLAYER_JUMP_IMPULSE,
    ) -> None:
        self.sprite = sprite

        self.left_key = left_key
        self.right_key = right_key
        self.up_key = up_key

        self.left_pressed = False
        self.right_pressed = False
        self.up_pressed = False

        self.name = name

        self.mass = mass
        self.damping = damping
        self.friction = friction
        self.max_horizontal_speed = max_horizontal_speed
        self.max_vertical_speed = max_vertical_speed
        self.move_force_on_ground = move_force_on_ground
        self.move_force_in_air = move_force_in_air
        self.jump_impulse = jump_impulse

    def is_on_ground(self, physics_engine):
        return physics_engine.is_on_ground(self.sprite)

    def on_key_press(self, key):
        if key == self.left_key:
            self.left_pressed = True
        elif key == self.right_key:
            self.right_pressed = True
        elif key == self.up_key:
            self.up_pressed = True

    def on_key_release(self, key):
        if key == self.left_key:
            self.left_pressed = False
        elif key == self.right_key:
            self.right_pressed = False
        elif key == self.up_key:
            self.up_pressed = False

    def apply_force(self, physics_engine):
        movements = (
            self.left_pressed,
            self.right_pressed,
            self.is_on_ground(physics_engine),
        )
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

    def on_update(self, physics_engine):
        self.attempt_jump(physics_engine)
        self.apply_force(physics_engine)
        self.set_friction(physics_engine)


class MyGame(arcade.Window):
    """
    Main application class.
    """

    def __init__(self):

        # Call the parent class and set up the window
        super().__init__(SCREEN_WIDTH, SCREEN_HEIGHT, SCREEN_TITLE)

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

        arcade.set_background_color(arcade.csscolor.WHITE)

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

        image_source = "images/blue_square.png"
        self.player_sprite1 = arcade.Sprite(image_source, CHARACTER_SCALING)
        self.player_sprite1.center_x = SCREEN_WIDTH / 8
        self.player_sprite1.center_y = SCREEN_HEIGHT / 8
        self.player_list.append(self.player_sprite1)
        self.scene.add_sprite("Player1", self.player_sprite1)

        self.player_sprite2 = arcade.Sprite(image_source, CHARACTER_SCALING)
        self.player_sprite2.center_x = SCREEN_WIDTH * (1 - 1 / 8)
        self.player_sprite2.center_y = SCREEN_HEIGHT / 8
        self.player_list.append(self.player_sprite2)
        self.scene.add_sprite("Player2", self.player_sprite2)

        # Create the ground
        # This shows using a loop to place multiple sprites horizontally
        for x in range(-100, SCREEN_WIDTH + 100, 10):
            wall = arcade.Sprite("images/blue_square.png", TILE_SCALING)
            wall.center_x = x
            wall.center_y = -SCREEN_HEIGHT / 15
            self.wall_list.append(wall)
            self.scene.add_sprite("Walls", wall)

        # Create the 'physics engine'
        """self.physics_engine = arcade.PhysicsEnginePlatformer(
            self.player_sprite1, gravity_constant=GRAVITY, walls=self.scene["Walls"]
        )"""
        self.physics_engine = arcade.PymunkPhysicsEngine(
            gravity=(0, -GRAVITY), damping=1.0, maximum_incline_on_ground=0.708
        )
        # self.physics_engine.add_collision_handler()
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
        )

        self.player2 = Player(
            self.player_sprite2,
            left_key=arcade.key.LEFT,
            right_key=arcade.key.RIGHT,
            up_key=arcade.key.UP,
        )

        # Set up the GUI Camera
        self.gui_camera = arcade.Camera(self.width, self.height)

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
        self.player1.on_update(self.physics_engine)
        self.player2.on_update(self.physics_engine)
        self.physics_engine.step()

    def on_draw(self):
        """Render the screen."""

        """ Draw everything """
        self.clear()
        self.wall_list.draw()
        # self.bullet_list.draw()
        # self.item_list.draw()
        self.player_list.draw()

        # Activate the GUI camera before drawing GUI elements
        self.gui_camera.use()

        # Draw our score on the screen, scrolling it with the viewport
        score_text = f"Score: {self.score}"
        arcade.draw_text(
            score_text,
            10,
            10,
            arcade.csscolor.WHITE,
            18,
        )


def main():
    """Main function"""
    window = MyGame()
    window.setup()
    arcade.run()


if __name__ == "__main__":
    main()
