import numpy as np
import pygame
import threading as th

import diffuzer as df
import game_content as gc

def normalize(a:np.array):
    a = complex(*a)
    r = abs(a)
    a = a/r if r !=0 else 0
    return np.array([a.real, a.imag], np.float32)

class pixeled_game:
    def __init__(self, canvas_size, world_size, screensize) -> None:
        self.screensize = np.array(screensize)
        self.camera_pos = np.array([*(300.0, 300.0)])

        self.canvas_size, self.world_size = canvas_size, world_size

        self.pg = pygame

        #----------camera----------
        self.scale = 4
        self.SX = self.screensize[0]//self.scale
        self.SY = self.screensize[1]//self.scale
        #--------------------------

        self.item_array = []

    def to_canvas_pos(self, screen_pos):
        return self.camera_pos[0] + int(self.SX*(screen_pos[0]/self.screensize[0] - 0.5)), self.camera_pos[1] + int(self.SY*(screen_pos[1]/self.screensize[1] - 0.5))

    def update_item_array(self):
        for i in self.item_array:
            if (i.life > 0):
                i.update(self)
                self.ligths.add_ligth(i.ligth, gc.from_loc_to_world(i.pos, self))
            else:
                self.item_array.remove(i)
                del i

    def draw_item_array(self, cv_surf, cv_ligth_surf):
        for i in self.item_array:
            i.draw(cv_surf, cv_ligth_surf, self)

    def get_world_image(self, canvas_pos):
        cv_im_surf, cv_ligth_surf = self.cv.get_image(
            (int(self.camera_pos[0]) - self.SX//2, int(self.camera_pos[1]) - self.SY//2), (self.SX, self.SY), canvas_pos, True)

        self.draw_item_array(cv_im_surf, cv_ligth_surf)
        del cv_ligth_surf

        cv_im_surf = self.cv.get_array_from_buff(
            cv_im_surf, (self.SX, self.SY))

        surf = self.pg.pixelcopy.make_surface(cv_im_surf)
        surf = self.pg.transform.scale(surf, self.screensize)
        return surf

    def run(self):
        #--------pygame------------
        self.sc = self.pg.display.set_mode(self.screensize, self.pg.FULLSCREEN)

        self.cv = df.canvas(
            gc.backgrounds["white_bag"], self.canvas_size, self.SX, self.SY, self.world_size)
        self.ligths = df.ligths

        self.cv.gen_cute_world()

        pl = gc.player((self.canvas_size[0], self.canvas_size[1]/2))
        CH = gc.chunk_updater(self.cv, pl, (20, 20), self.item_array)
        self.item_array.append(pl)
        self.cv.circle(pl.pos, 100, gc.bv['air'])

        TIME = 0
        bias = [0, 0]
        clock = self.pg.time.Clock()
        work = True
        while work:
            clock.tick(120)
            CH.update_chunk(self)
            TIME = (TIME + 1)%600

            self.cv.update_canvas(self.camera_pos, (500, 500), TIME)
            self.update_item_array()
            
            surf = self.get_world_image(CH.cv_pos)

            bias = [self.scale*(int(self.camera_pos[0]) - self.camera_pos[0]),
                    self.scale*(int(self.camera_pos[1]) - self.camera_pos[1])]
            
            self.sc.blit(surf, bias)
            self.ligths.clear()

            for ev in self.pg.event.get():
                if ev.type == self.pg.QUIT:
                    work = False
                    break

            key = self.pg.key.get_pressed()

            if key[self.pg.K_e]:
                s = np.array(self.to_canvas_pos(self.pg.mouse.get_pos()))
                k = 10*normalize(s - pl.pos)
                self.cv.circle(k*3 + pl.pos, 10, gc.bv['destroing_matter'], k)
            if key[self.pg.K_q]:
                self.cv.circle(self.to_canvas_pos(self.pg.mouse.get_pos()), 10, gc.bv['sand'])

            self.camera_pos += (pl.pos - self.camera_pos)/10

            self.pg.display.update()
