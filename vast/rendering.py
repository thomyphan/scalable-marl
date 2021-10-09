from vast.utils import get_param_or_default

class Renderer:

    def __init__(self, params):
        self.render_environment = get_param_or_default(params, "render_environment", False)
        self.render_pygame = get_param_or_default(params, "render_pygame", False)
        self.render_channel, self.render_stub = None, None

    def render(self, env):
        if self.render_environment or self.render_pygame:
            self.render_stub = env.render(self.render_stub)

    def close(self):
        if self.render_environment:
            self.render_channel.close()
        elif self.render_pygame:
            if self.render_stub is not None:
                self.render_stub.close()