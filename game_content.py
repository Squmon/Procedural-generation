from PIL import Image
import numpy as np
import diffuzer as df

GRAVITY = 0.4
AIR_K = 0.99

textures = {
    "crab": Image.open("sprites/crab.png").convert("RGB"),
    "miner_sprite": Image.open("sprites/miner_sprite.png").convert("RGB"),
    "debug_bullet": Image.open("sprites/debug_bullet.png").convert("RGB"),
    "classic_bullet": Image.open("sprites/classic_bullet.png").convert("RGB"),
    "min_sprite": Image.open("sprites/min_sprite.png").convert("RGB")
}

backgrounds = {
    "basic_space": Image.open("backgrounds/basic_space.png").convert("RGB"),
    "white_bag": Image.open("backgrounds/white_bag.png").convert("RGB")
}

collisions = {
    "crab": Image.open("collisions/crab.png").convert("RGB"),
    "circle": Image.open("collisions/circle.png").convert("RGB"),
    "miner_coll":Image.open("collisions/miner_coll.png").convert("RGB"),
    "small_circle": Image.open("collisions/small_circle.png").convert("RGB")
}

bv = {
    "air": 0,
    "sand": 1,
    "stone": 2,
    "sandstone": 3,
    "destroing_matter": 4
}


def from_loc_to_world(pos, gctx):

    return pos - np.array(gctx.camera_pos) + np.array((gctx.SX, gctx.SY))/2


class collision_game_object:
    def __init__(self, pos, image_name="crab", collision_name="crab", ligth = [0, 0, 0]) -> None:
        self.coll = collisions[collision_name]
        self.coll_arr = np.asarray(self.coll, dtype=np.int32)

        self.SPRITE = textures[image_name]

        self.pos = np.array([*pos], dtype=np.float32)
        self.movement = np.array([0, 0], dtype=np.float32)
        self.rot_vect = 1j
        self.life = 1

        self.ligth = ligth

    def draw(self, camera_surf, cv_ligth_surf, gctx):
        gctx.cv.draw_image(from_loc_to_world(self.pos, gctx), (self.rot_vect.real, self.rot_vect.imag),
                           self.SPRITE, camera_surf, (gctx.SX, gctx.SY), cv_ligth_surf)

    def update(self, gctx):
        r = self.pos - np.array(gctx.camera_pos)
        if sum(r*r) < (gctx.scale*50)**2:
            self.movement[1] += 1
            self.movement *= AIR_K
            j = gctx.cv.get_force_collision(self.coll_arr, self.pos + self.movement)
            j1 = gctx.cv.get_force_collision(self.coll_arr, self.pos)
            if (j[0] == 0) and (j[1] == 0):
                self.pos += np.array(self.movement, dtype=np.int32)
            else:
                self.movement *= 0

            if (j1[0] != 0) or (j1[1] != 0):
                self.pos += j1

class weapon:
    def __init__(self, owner) -> None:
        self.owner = owner
    
    def update(self, gctx, mouse):
        if mouse[0]:
            self.fire()

    def fire(self, *args):
        pass

class unrun(weapon):
    def __init__(self, owner) -> None:
        super().__init__(owner)
        self.charge = 0

    def update(self, gctx, direction, mouse):
        if mouse[0]:
            self.add_charge()

        else:
            d = gctx.to_canvas_pos(gctx.pg.mouse.get_pos()) - self.owner.pos
            self.fire(gctx, d)

    def add_charge(self):
        self.charge = min(self.charge + 0.8, 50)

    def fire(self, gctx, direction, * args):
        if self.charge >= 1:
            self.charge = max(10, self.charge)
            self.stage = -1
            d = direction
            d /= np.sqrt(np.dot(d, d))
            gctx.item_array.append(debug_bullet(
                self.owner.pos, (self.charge)*d, int(self.charge)))
            self.charge = 0

class classic_gun(weapon):
    def __init__(self, owner) -> None:
        super().__init__(owner)
        self.delay = 10
        self.t = 1
    
    def fire(self, gctx, direction, * args):
        d = direction
        d /= np.sqrt(np.dot(d, d))
        gctx.item_array.append(classic_bullet(
            self.owner.pos, d*11, 10))

    def update(self, gctx, direction, mouse):
        if mouse[0] and (self.t == 0):
            self.fire(gctx, direction)
            #self.owner.movement -= direction*5
            self.t = (self.t + 2) % self.delay
        if (self.t == 1):
            self.t = 0
        else:
            self.t = (self.t + 1) % self.delay

class player(collision_game_object):
    def __init__(self, pos) -> None:
        super().__init__(pos, "crab", "circle", [1, 0, 0])
        self.life = 1

        self.w:weapon = classic_gun(self)
        self.w1: weapon = unrun(self)

    def update(self, gctx):
        self.movement[1] += GRAVITY
        key = gctx.pg.key.get_pressed()
        S = 0.3
        k = -1j*complex(*self.movement)
        qq = abs(k)
        self.rot_vect = k/qq if qq != 0 else 1
        if key[gctx.pg.K_a]:
            self.movement[0] -= S*1.1
        if key[gctx.pg.K_d]:
            self.movement[0] += S*1.1
        if key[gctx.pg.K_w]:
            self.movement[1] -= S*2.1
        if key[gctx.pg.K_s]:
            self.movement[1] += S


        mouse = gctx.pg.mouse.get_pressed()
        d = gctx.to_canvas_pos(gctx.pg.mouse.get_pos()) - self.pos
        self.w.update(gctx, d, mouse)
        self.w1.update(gctx, d, [mouse[2], mouse[1], mouse[0]])
        
        r = self.pos - np.array(gctx.camera_pos)
        if sum(r*r) < (gctx.scale*50)**2:
            self.movement *= AIR_K
            j = gctx.cv.get_force_collision(self.coll_arr, self.pos + self.movement)
            j1 = gctx.cv.get_force_collision(self.coll_arr, self.pos)
            if (j[0] == 0) and (j[1] == 0):
                self.pos += np.array(self.movement, dtype=np.int32)
            else:
                self.movement *= 0

            if (j1[0] != 0) or (j1[1] != 0):
                self.pos += j1


class classic_bullet(collision_game_object):
    def __init__(self, pos, movement, power = 10) -> None:
        super().__init__(pos, "classic_bullet", "small_circle", [0.5, 0.5, 0])
        self.movement = movement
        self.power = power
        self.life = 4

    def update(self, gctx):
        self.movement[1] += GRAVITY
        k = 1j*complex(*self.movement)
        qq = abs(k)
        self.rot_vect = k/qq if qq != 0 else 1
        r = self.pos - np.array(gctx.camera_pos)
        if sum(r*r) < (gctx.scale*200)**2:
            j = gctx.cv.get_force_collision(
                self.coll_arr, self.pos + self.movement)
            j1 = gctx.cv.get_force_collision(
                self.coll_arr, self.pos)
            if (j[1] == 0) and (j[0] == 0):
                self.pos += np.array(self.movement, dtype=np.int32)
            else:
                self.life -= 1
                G = self.pos + np.array(self.movement, dtype=np.int32)
                gctx.cv.circle(G, 1, bv["sand"], self.movement)
                self.movement += j
                self.pos -= j/2
                #gctx.cv.force_circle(G, self.power, self.power)

        else:
            self.life = -100


class debug_bullet(collision_game_object):
    def __init__(self, pos, movement, power = 10) -> None:
        super().__init__(pos, "debug_bullet", "circle", [1, 0, 1])
        self.movement = movement
        self.power = power

    def update(self, gctx):
        self.movement[1] += GRAVITY
        k = 1j*complex(*self.movement)
        qq = abs(k)
        self.rot_vect = k/qq if qq != 0 else 1
        r = self.pos - np.array(gctx.camera_pos)
        if sum(r*r) < (gctx.scale*200)**2:
            j = gctx.cv.get_force_collision(
                self.coll_arr, self.pos + self.movement)
            j1 = gctx.cv.get_force_collision(self.coll_arr, self.pos)
            if (j[0] == 0) and (j[1] == 0):
                self.pos += np.array(self.movement, dtype=np.int32)
            else:
                self.life = -100
                G = self.pos + np.array(self.movement, dtype=np.int32)
                gctx.cv.circle(G, self.power, bv["destroing_matter"], self.movement)

            if (j1[0] != 0) or (j1[1] != 0):
                self.pos += j1
        else:
            self.life = -100


class chunk_updater():
    def __init__(self, canvas, pl, cv_pos, item_array) -> None:
        self.pl: player = pl
        self.CV: df.canvas = canvas
        self.item_array = item_array

        self.cv_pos = np.array(cv_pos, float)

        self.ws = np.array(self.CV.world_size)
        self.cs = np.array(self.CV.canvas_size)

        self.CV.get_canvas_from_world(cv_pos)

    def update_chunk(self, gctx):
        k = self.pl.pos - self.cs/2
        if sum(k*k) > (self.cs[0]**2)/18:
            self.CV.update_world_from_canvas(self.cv_pos - self.cs/2)
            self.cv_pos += k
            self.CV.get_canvas_from_world(self.cv_pos - self.cs/2)
            b = gctx.camera_pos - self.pl.pos

            for i in self.item_array:
                if (i != self.pl):
                    i.pos -= k

            self.pl.pos = self.cs/2
            gctx.camera_pos = self.pl.pos + b
