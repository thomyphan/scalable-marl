import pygame

BLACK = (0, 0, 0)
DARK_GRAY = (125, 125, 125)
GRAY = (175, 175, 175)
WHITE = (255, 255, 255)
RED = (255, 0, 0)
ORANGE = (255, 150, 0)
BLUE = (0, 0, 255)
LIGHT_BLUE = (51, 226, 253)
GREEN = (0, 255, 0)

class WarehouseViewer:
    def __init__(self, width, height, cell_size=20, fps=30):
        pygame.init()
        self.cell_size = cell_size
        self.width = cell_size*width
        self.height = cell_size*height
        self.clock = pygame.time.Clock()
        self.fps = fps
        pygame.display.set_caption("Warehouse Environment")
        self.screen = pygame.display.set_mode((self.width, self.height))
        pygame.event.set_blocked(pygame.MOUSEMOTION)  # we do not need mouse movement events
              
    def draw_state(self, env):
        self.screen.fill(BLACK)
        for position in env.locations.keys():
            x, y = position
            location = env.locations[position]
            if location.dropoff_type == 0:
                self.draw_pixel(x, y, ORANGE)
            elif location.dropoff_type == 1:
                self.draw_pixel(x, y, WHITE)
            else:
                self.draw_pixel(x, y, LIGHT_BLUE)
        for agent in env.agents:
            x, y = agent.position
        pygame.display.flip()
        self.clock.tick(self.fps)
        return self.check_for_interrupt()
    
    def draw_pixel(self, x, y, color):
        pygame.draw.rect(self.screen, color,
                        pygame.Rect(
                            x * self.cell_size+1,
                            y * self.cell_size+1,
                            self.cell_size-2,
                            self.cell_size-2),
                        0)
    
    def draw_circle(self, x, y, color):
        radius = int(self.cell_size/2)
        center_x = x * self.cell_size + radius
        center_y = y * self.cell_size + radius
        center = (center_x, center_y)
        pygame.draw.circle(self.screen, color, center, radius)

    def check_for_interrupt(self):
        key_state = pygame.key.get_pressed()
        for event in pygame.event.get():
            if event.type == pygame.QUIT or key_state[pygame.K_ESCAPE]:
                return True
        return False

    def close(self):
        pygame.quit()


def render(env, viewer):
    if viewer is None:
        viewer = WarehouseViewer(env.width, env.height)
    viewer.draw_state(env)
    return viewer

