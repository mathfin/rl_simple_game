import pygame


class Ball(pygame.sprite.Sprite):
    def __init__(self, x, speed, filename, group):
        pygame.sprite.Sprite.__init__(self)

        self.image = pygame.image.load(filename)
        self.image.set_colorkey((6, 6, 6))
        self.image = self.image.convert_alpha()

        self.rect = self.image.get_rect(center=(x, 0))
        self.speed = speed
        self.add(group)

    def update(self, *args) -> None:
        if self.rect.y < args[0]-50:
            self.rect.y += self.speed
        else:
            self.kill()
