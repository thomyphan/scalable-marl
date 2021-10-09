import pygame

WHITE = (255, 255, 255)
RED = (255, 0, 0)
BLUE = (0, 0, 255)

class BattleViewer:
    def __init__(self, width, height, cell_size=20, fps=30):
        pygame.init()
        self.cell_size = cell_size
        self.width = cell_size*height
        self.height = cell_size*width
        self.clock = pygame.time.Clock()
        self.fps = fps
        pygame.display.set_caption("Battle Environment")
        self.screen = pygame.display.set_mode((self.width, self.height))
        pygame.event.set_blocked(pygame.MOUSEMOTION)  # we do not need mouse movement events
              
    def draw_state(self, env):
        self.screen.fill(WHITE)
        global_state_array = env.global_state().view(env.global_state_space.shape).detach().numpy()
        self.draw_matrix(global_state_array[0], RED)
        self.draw_matrix(global_state_array[1], BLUE)
        pygame.display.flip()
        self.clock.tick(self.fps)
        return self.check_for_interrupt()
    
    def draw_matrix(self, matrix, color):
        for y, row in enumerate(matrix):
            for x, val in enumerate(row):
                if val:
                    pygame.draw.rect(
                        self.screen,
                        color,
                        pygame.Rect(
                            y * self.cell_size,
                            x * self.cell_size,
                            self.cell_size,
                            self.cell_size),
                        0)

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
        viewer = BattleViewer(env.width, env.height)
    viewer.draw_state(env)
    return viewer

