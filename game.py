import pygame
import neat
import random
import os
import sys


# canvas setting
CANVAS_WIDTH = 500
CANVAS_HEIGHT = 800
BG_IMG = pygame.transform.scale2x(pygame.image.load(os.path.join(
    "assets/sprites", "bg-day.png")))
PIPE_IMG = pygame.transform.scale2x(pygame.image.load(os.path.join(
    "assets/sprites", "pipe.png")))
BIRD_IMGS = [pygame.transform.scale2x(pygame.image.load(os.path.join(
    "assets/sprites", "bird" + str(x) + ".png"))) for x in range(1, 4)]
BASE_IMG = pygame.transform.scale2x(pygame.image.load(os.path.join(
    "assets/sprites", "base.png")))
# STAT_FONT = [pygame.transform.scale2x(pygame.image.load(os.path.join(
#    "assets/sprites", str(x) + ".png"))) for x in range(0, 10)]
pygame.font.init()
STAT_FONT = pygame.font.SysFont("comicsans", 50)


class Bird:
    IMGS = BIRD_IMGS
    MAX_ROTATION = 25
    ROT_VEL = 20
    ANIMATION_TIME = 10

    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.tilt = 0
        self.tick_count = 0
        self.vel = 0
        self.height = self.y
        self.img_count = 0
        self.img = self.IMGS[0]

        self.score = 0

    def jump(self):
        self.vel = -10.5
        self.tick_count = 0
        self.height = self.y

    def move(self):
        # elapse time
        self.tick_count += 1

        # for downward acceleration
        displacement = self.vel*(self.tick_count) + 0.5 * \
            (3)*(self.tick_count)**2  # calculate displacement

        # terminal velocity
        if displacement >= 16:
            displacement = (displacement/abs(displacement)) * 16

        if displacement < 0:
            displacement -= 2

        self.y = self.y + displacement

        if displacement < 0 or self.y < self.height + 50:  # tilt up
            if self.tilt < self.MAX_ROTATION:
                self.tilt = self.MAX_ROTATION
        else:  # tilt down
            if self.tilt > -90:
                self.tilt -= self.ROT_VEL

    def draw(self, screen):
        # elapsed time
        self.img_count += 1

        count = self.img_count % 30

        # choose a img
        if count < self.ANIMATION_TIME:
            self.img = self.IMGS[0]
        elif count < self.ANIMATION_TIME*2:
            self.img = self.IMGS[1]
        elif count < self.ANIMATION_TIME*3:
            self.img = self.IMGS[2]

        # if rotate
        if self.tilt <= -80:
            self.img = self.IMGS[1]

        rotated_image = pygame.transform.rotate(self.img, self.tilt)
        new_rect = rotated_image.get_rect(
            center=self.img.get_rect(topleft=(self.x, self.y)).center)
        # draw
        screen.blit(rotated_image, new_rect)

    def get_mask(self):
        return pygame.mask.from_surface(self.img)


class Pipe:
    SPACE = 200
    VEL = 5

    def __init__(self, x):
        self.x = x
        self.height = 0
        self.top = 0
        self.bottom = 0
        self.TOP = pygame.transform.flip(PIPE_IMG, False, True)
        self.BOTTOM = PIPE_IMG
        self.passed = False
        self.width = PIPE_IMG.get_width()
        self.set_height()

    def set_height(self):
        self.height = random.randrange(50, 450)
        self.top = self.height - self.TOP.get_height()
        self.bottom = self.height + self.SPACE

    def move(self):
        self.x -= self.VEL

    def draw(self, screen):
        screen.blit(self.TOP, (self.x, self.top))
        screen.blit(self.BOTTOM, (self.x, self.bottom))

    def collide(self, bird):
        bird_mask = bird.get_mask()
        top_mask = pygame.mask.from_surface(self.TOP)
        bottom_mask = pygame.mask.from_surface(self.BOTTOM)

        top_offset = (self.x - bird.x, self.top - round(bird.y))
        bottom_offset = (self.x - bird.x, self.bottom - round(bird.y))

        b_point = bird_mask.overlap(bottom_mask, bottom_offset)
        t_point = bird_mask.overlap(top_mask, top_offset)

        if b_point or t_point:
            return True

        return False


class Base:
    VEL = 5
    WIDTH = BASE_IMG.get_width()
    IMG = BASE_IMG

    def __init__(self, y):
        self.y = y
        self.x1 = 0
        self.x2 = self.WIDTH

    def move(self):
        self.x1 -= self.VEL
        self.x2 -= self.VEL

        if self.x1 + self.WIDTH < 0:
            self.x1 = self.x2 + self.WIDTH

        if self.x2 + self.WIDTH < 0:
            self.x2 = self.x1 + self.WIDTH

    def draw(self, screen):
        screen.blit(self.IMG, (self.x1, self.y))
        screen.blit(self.IMG, (self.x2, self.y))


def delete_pipe(pipes):
    new_list = []
    for pipe in pipes:
        if not pipe.x + pipe.TOP.get_width() < 0:
            new_list.append(pipe)
    return new_list


def add_pipe():
    return Pipe(600)


def pass_check(pipes, bird):
    add = False
    for pipe in pipes:
        if not pipe.passed and (pipe.x + pipe.width) < bird.x:
            pipe.passed = True
            bird.score += 1
            add = True
    if add:
        pipes.append(add_pipe())

    return pipes, bird, add


def draw_window(screen, birds, pipes, base):
    screen.blit(BG_IMG, (0, 0))
    score = 0
    for pipe in pipes:
        pipe.draw(screen)
    base.draw(screen)

    for bird in birds:
        bird.draw(screen)
        score = bird.score if bird.score > score else score

    text = STAT_FONT.render("Score: " + str(score), 1, (255, 255, 255))
    screen.blit(text, (CANVAS_WIDTH - 10 - text.get_width(), 10))

    pygame.display.update()


def main(genomes, config):
    birds = []
    nets = []
    ge = []

    for _, g in genomes:
        net = neat.nn.FeedForwardNetwork.create(g, config)
        nets.append(net)
        birds.append(Bird(100, 350))
        g.fitness = 0
        ge.append(g)

    # bird = Bird(100, 350)
    base = Base(730)
    pipes = [add_pipe()]
    screen = pygame.display.set_mode((CANVAS_WIDTH, CANVAS_HEIGHT))
    clock = pygame.time.Clock()

    run = True
    while run:
        # clock.tick(30)
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                run = False
                pygame.quit()
                sys.exit()
                break

        pipes = delete_pipe(pipes)

        if len(birds) == 0:
            run = False
            break

        for index, bird in enumerate(birds):
            pipe_ind = 0
            if len(pipes) > 1 and birds[0].x + pipes[0].TOP.get_width():
                pipe_ind = 1

            # send bird location, top pipe location and bottom pipe location and determine from network whether to jump or not
            isJump = nets[birds.index(bird)].activate((bird.y, abs(
                bird.y - pipes[pipe_ind].height), abs(bird.y - pipes[pipe_ind].bottom), pipes[pipe_ind].passed))

            if isJump[0] > 0.5:
                bird.jump()

            ge[index].fitness += 0.1
            bird.move()
            base.move()

            pipes, bird, ge_update = pass_check(pipes, bird)

            if ge_update:
                for g in ge:
                    g.fitness += 1

            for pipe in pipes:
                if pipe.collide(bird):
                    ge[index].fitness -= 1

                    # remove bird
                    birds.pop(index)
                    nets.pop(index)
                    ge.pop(index)

            # base collide
            if bird.y + bird.img.get_height() >= 730 or bird.y < 0:
                ge[index].fitness -= 1

                # remove bird
                birds.pop(index)
                nets.pop(index)
                ge.pop(index)

        for pipe in pipes:
            pipe.move()

        draw_window(screen, birds, pipes, base)


def run(config_path):
    config = neat.config.Config(neat.DefaultGenome, neat.DefaultReproduction,
                                neat.DefaultSpeciesSet, neat.DefaultStagnation, config_path)

    popularion = neat.Population(config)

    popularion.add_reporter(neat.StdOutReporter(True))
    stats = neat.StatisticsReporter()
    popularion.add_reporter(stats)

    winner = popularion.run(main, 50)


if __name__ == "__main__":
    local_dir = os.path.dirname(__file__)
    config_path = os.path.join(local_dir, "config-feedforward.txt")
    run(config_path)
