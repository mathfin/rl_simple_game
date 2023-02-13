from random import randint

import pygame
from ball import Ball

pygame.init()
pygame.time.set_timer(pygame.USEREVENT, 2000)

BLACK = (0, 0, 0)
BLUE = (0, 0, 255)
GREEN = (0, 255, 0)
W, H = 1000, 570

sc = pygame.display.set_mode((W, H))
pygame.display.set_caption("Reinforcement Learning Lab")

clock = pygame.time.Clock()
FPS = 60

balls = pygame.sprite.Group()

portal = pygame.image.load('portal.png')
portal.set_colorkey((0, 0, 0))
portal = portal.convert_alpha()
p_rect = portal.get_rect(centerx=W // 2, bottom=H - 5)


def create_ball(group):
    x = randint(20, W - 20)
    speed = 1 # randint(1, 4)
    return Ball(x, speed, 'rick.jpg', group)


game_score = 0


def collide_balls():
    global game_score
    for ball in balls:
        if p_rect.collidepoint(ball.rect.center):
            game_score += 100
            ball.kill()


speed = 10
flRunning = True
create_ball(balls)
image_num = 0
while flRunning:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            exit()
        elif event.type == pygame.USEREVENT:
            create_ball(balls)
    keys = pygame.key.get_pressed()
    if keys[pygame.K_LEFT]:
        p_rect.x -= speed
        if p_rect.x < 0:
            p_rect.x = 0
    elif keys[pygame.K_RIGHT]:
        p_rect.x += speed
        if p_rect.x > W - p_rect.width:
            p_rect.x = W - p_rect.width
    collide_balls()
    sc.fill(BLACK)
    balls.draw(sc)
    sc.blit(portal, p_rect)
    text = pygame.font.Font(None, 24).render(f"Счет: {game_score}", 1, BLUE, GREEN)
    t_rect = text.get_rect(topleft=(0, 0))
    sc.blit(text, t_rect)
    pygame.display.update()
    #if image_num%30 == 0:
        #pygame.image.save(sc, f'./images/screen_{image_num}.jpg')
    #image_num += 1
    balls.update(H)
    clock.tick(FPS)
