import pygame
import random
import torch
import torch.optim as optim
import math

from torch import nn
from models import DQN, ReplayMemory, Transition
from constants import *
from random import randint
from ball import Ball

# Game initialize
pygame.init()
pygame.time.set_timer(pygame.USEREVENT, 2010)
sc = pygame.display.set_mode((W, H))
pygame.display.set_caption("Reinforcement Learning Lab")
clock = pygame.time.Clock()
balls = pygame.sprite.Group()
portal = pygame.image.load('portal.png')
portal.set_colorkey(BLACK)
portal = portal.convert_alpha()
p_rect = portal.get_rect(centerx=W // 2, bottom=H - 5)


def create_ball(group):
    x = randint(20, W - 20)
    speed = randint(1, 4)
    return Ball(x, speed, 'rick.jpg', group)


def collide_balls():
    reward = 0
    global game_score
    for ball in balls:
        if p_rect.collidepoint(ball.rect.center):
            game_score += 100
            reward += 100
            ball.kill()
        # reward -= (((ball.rect.x-p_rect.x)**2+(ball.rect.y-p_rect.y)**2)**(1/4))/1000
    return reward


speed = 10
flRunning = True
create_ball(balls)
seconds_from_start = 0

policy_net = torch.load("my_morty_net.pth") # DQN(n_observations, n_actions)
target_net = DQN(n_observations, n_actions)
target_net.load_state_dict(policy_net.state_dict())

optimizer = optim.AdamW(policy_net.parameters(), lr=LR, amsgrad=True)
memory = ReplayMemory(10000)

steps_done = 0


def get_state():
    """
        Для каждого рика три числа:
        две координаты и одна скорость,
        всего риков 5, получается 5*3=15
        + координата по x для портала
        итого тензор состояния состоит
        из 16 чисел
    """
    state_tensor = torch.zeros(21, dtype=torch.float32)
    state_tensor[-3] = p_rect.x
    state_tensor[-2] = p_rect.y
    state_tensor[-1] = 10
    for num, ball in enumerate(balls):
        if num <= 5:
            state_tensor[3 * num] = ball.rect.x
            state_tensor[3 * num + 1] = ball.rect.y
            state_tensor[3 * num + 2] = ball.speed
        else:
            print("To much balls")
    return state_tensor


def select_action(state):
    global steps_done
    sample = random.random()
    eps_threshold = EPS_END + (EPS_START - EPS_END) * \
                    math.exp(-1. * steps_done / EPS_DECAY)
    steps_done += 1
    if sample > eps_threshold:
        with torch.no_grad():
            return policy_net(state).max(1)[1].view(1, 1)
    else:
        return torch.tensor([[randint(0, 1)]], dtype=torch.long)


def optimize_model():
    if len(memory) < BATCH_SIZE:
        return
    transitions = memory.sample(BATCH_SIZE)
    batch = Transition(*zip(*transitions))

    non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                            batch.next_state)), dtype=torch.bool)
    non_final_next_states = torch.cat([s for s in batch.next_state
                                       if s is not None])
    state_batch = torch.cat(batch.state)
    action_batch = torch.cat(batch.action)
    reward_batch = torch.cat(batch.reward)

    state_action_values = policy_net(state_batch).gather(1, action_batch)

    next_state_values = torch.zeros(BATCH_SIZE)
    with torch.no_grad():
        next_state_values[non_final_mask] = target_net(non_final_next_states).max(1)[0]

    expected_state_action_values = (next_state_values * GAMMA) + reward_batch

    criterion = nn.SmoothL1Loss()
    loss = criterion(state_action_values, expected_state_action_values.unsqueeze(1))

    optimizer.zero_grad()
    loss.backward()

    torch.nn.utils.clip_grad_value_(policy_net.parameters(), 100)
    optimizer.step()


episode_durations = []


def run_game():
    global seconds_from_start
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            exit()
        elif event.type == pygame.USEREVENT:
            seconds_from_start += 2
            create_ball(balls)

    # Тут выберем действие и сделаем шаг игры
    state = get_state()
    state = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
    action = select_action(state)
    if action == 0:  # left
        p_rect.x -= speed
        if p_rect.x < 0:
            p_rect.x = 0
    elif action == 1:  # right
        p_rect.x += speed
        if p_rect.x > W - p_rect.width:
            p_rect.x = W - p_rect.width

    # Определяем столкновение и меняем счет
    reward = collide_balls()

    # Отрисовка
    sc.fill(BLACK)
    balls.draw(sc)
    sc.blit(portal, p_rect)
    text = pygame.font.Font(None, 24).render(f"Счет: {game_score} Награда {round(reward, 4)}", 1, BLUE, GREEN)
    t_rect = text.get_rect(topleft=(0, 0))
    sc.blit(text, t_rect)
    pygame.display.update()

    # Смещаем риков
    balls.update(H)
    clock.tick(FPS)

    # Тут можно уже будет рассчитать observation, reward, terminated, truncated
    observation = get_state()
    reward = torch.tensor([reward])

    next_state = torch.tensor(observation, dtype=torch.float32).unsqueeze(0)

    memory.push(state, action, next_state, reward)
    state = next_state
    optimize_model()

    target_net_state_dict = target_net.state_dict()
    policy_net_state_dict = policy_net.state_dict()
    for key in policy_net_state_dict:
        target_net_state_dict[key] = policy_net_state_dict[key] * TAU + target_net_state_dict[key] * (1 - TAU)
    target_net.load_state_dict(target_net_state_dict)
    return game_score


while seconds_from_start <= 1000:
    score = run_game()
print(score)
torch.save(policy_net, 'my_morty_net.pth')
exit()