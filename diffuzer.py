import pyopencl as cl
import numpy as np
from pyopencl import mem_flags as mf


class ligths:
    ligths_array = []
    def add_ligth(color, pos):
        ligths.ligths_array += [*pos, *color]
    def clear():
        ligths.ligths_array.clear()
    def get_numpy_array():
        if len(ligths.ligths_array) != 0:
            return np.array(ligths.ligths_array, np.float32)
        return np.array([1]*5, np.float32)
        


class canvas:
    def __init__(self, bg_image, size, SX, SY, world_size=(1000, 1000)) -> None:
        self.ctx = cl.create_some_context(0)
        self.queue = cl.CommandQueue(self.ctx)
        with open('kernels.cl', 'r') as j:
            source = j.read()

        self.particle_param_count = 1 + 2

        self.world_size = world_size
        self.world_size_buff = cl.Buffer(self.ctx, mf.COPY_HOST_PTR | mf.READ_ONLY, hostbuf=np.array([*self.world_size, self.particle_param_count], dtype=np.int32))
        self.world_buff = cl.Buffer(self.ctx, mf.COPY_HOST_PTR, hostbuf = np.zeros((self.world_size[0]*self.world_size[1] * self.particle_param_count), np.int32))

        self.wall_buffer = cl.Buffer(self.ctx, mf.COPY_HOST_PTR, hostbuf = np.zeros((self.world_size[0]*self.world_size[1]*3), np.float32))

        a = np.asarray(bg_image, dtype=np.int32)
        self.bg_buffer = cl.Buffer(self.ctx, mf.COPY_HOST_PTR | mf.READ_ONLY, hostbuf= a.reshape(a.shape[0]*a.shape[1]*3))
        self.bg_size_buffer = cl.Buffer(self.ctx, mf.COPY_HOST_PTR | mf.READ_ONLY, hostbuf= np.array((a.shape[0], a.shape[1], 3)))

        #важные функции
        self.program = cl.Program(self.ctx, source).build()
        self.__update_canvas = self.program.update_canvas
        self.__get_image = self.program.get_image
        self.__get_ligth = self.program.get_ligth_map
        self.__draw_image = self.program.draw_image
        self.__copy_canvas = self.program.copy_canvas
        self.__fill_zeros = self.program.fill_zeros
        
        self.__get_canvas_from_world = self.program.get_canvas_from_world
        self.__update_world_from_canvas = self.program.update_world_from_canvas

        #частные функции
        self.__get_coll = self.program.get_force
        self.__explotion = self.program.explotion
        self.__fill_canvas = self.program.fill_canvas
        self.__circle = self.program.circle
        self.__force_circle = self.program.force_circle

        self.__gen_cute_world = self.program.gen_cute_world
        self.__gen_noise = self.program.noise_generator
        self.__blur = self.program.blur
        self.__dust_filter = self.program.dust_filter
        self.__errosion_filter = self.program.errosion_filter
        self.__map_mul = self.program.map_mul

        
        self.canvas_size = (*size,)
        self.screensize = (SX, SY)

        self.im = np.zeros((self.screensize[0]*self.screensize[1]*3, ), dtype=np.int32)
        self.im_buff = cl.Buffer(self.ctx, mf.COPY_HOST_PTR | mf.WRITE_ONLY, hostbuf=self.im)
        self.ligth_map_buff = cl.Buffer(self.ctx, mf.COPY_HOST_PTR | mf.WRITE_ONLY, hostbuf=self.im)


        self.canvas_size_buff = cl.Buffer(self.ctx, mf.COPY_HOST_PTR | mf.READ_ONLY, hostbuf=np.array([*self.canvas_size, self.particle_param_count], dtype=np.int32))

        id_array = np.array(np.random.sample((self.canvas_size[0]*self.canvas_size[1], )) + 0.15, dtype=np.int32)

        u_array_x = np.array(1.5*(np.random.sample((self.canvas_size[0]*self.canvas_size[1], ))*2 - 1), dtype=np.int32)
        u_array_y = np.array(1.5*(np.random.sample((self.canvas_size[0]*self.canvas_size[1], ))*2 - 1), dtype=np.int32)

        self.canvas_s = cl.Buffer(self.ctx, mf.COPY_HOST_PTR, hostbuf = np.array(np.stack((id_array, u_array_x, u_array_y), axis = 1), dtype=np.int32))

        self.next_canvas = cl.Buffer(self.ctx, mf.COPY_HOST_PTR, hostbuf = np.zeros((self.canvas_size[0]*self.canvas_size[1]*self.particle_param_count, ), dtype=np.int32))

    def update_canvas(self, center, sizes, time = 0.0):
        upzb = cl.Buffer(self.ctx, mf.COPY_HOST_PTR | mf.READ_ONLY, hostbuf = np.array(sizes, np.int32))
        cb = cl.Buffer(self.ctx, mf.COPY_HOST_PTR | mf.READ_ONLY, hostbuf = np.array(center, np.int32))

        self.__copy_canvas(self.queue, self.canvas_size, None, self.canvas_s, self.next_canvas, self.canvas_size_buff)
        self.__fill_zeros(self.queue, sizes, None, self.next_canvas, self.canvas_size_buff, cb, upzb)

        self.__update_canvas(self.queue, sizes, None, self.canvas_s, self.next_canvas, self.canvas_size_buff, cl.Buffer(self.ctx, mf.COPY_HOST_PTR | mf.READ_ONLY, hostbuf = np.array(time, dtype=np.float32)), 
        cb, upzb)

        self.canvas_s, self.next_canvas = self.next_canvas, self.canvas_s

    def get_canvas(self):
        a = np.empty((self.canvas_size[0]*self.canvas_size[1]*self.particle_param_count, ), dtype = np.int32)
        cl.enqueue_copy(self.queue, a, self.canvas_s)
        return a

    def get_canvas_from_world(self, position):
        self.__get_canvas_from_world(self.queue, self.canvas_size, None, self.world_buff, self.world_size_buff, self.canvas_s, self.canvas_size_buff, 
            cl.Buffer(self.ctx, mf.COPY_HOST_PTR | mf.READ_ONLY, hostbuf = np.array(position, dtype=np.int32))
        )

    def update_world_from_canvas(self, position):
        self.__update_world_from_canvas(self.queue, self.canvas_size, None, self.world_buff, self.world_size_buff, self.canvas_s, self.canvas_size_buff, 
            cl.Buffer(self.ctx, mf.COPY_HOST_PTR | mf.READ_ONLY, hostbuf = np.array(position, dtype=np.int32))
        )

    def calculate_pos(self, pos_x, pos_y, i):
        pos_x = int(pos_x)
        pos_y = int(pos_y)
        return pos_x*self.particle_param_count + self.canvas_size[0]*self.particle_param_count*pos_y + i


    def set_cavnas(self, new_canvas):
        self.canvas_s = cl.Buffer(self.ctx, mf.COPY_HOST_PTR, hostbuf=np.array(
            new_canvas, dtype=np.int32))

    def get_image(self, position, image_size, canvas_pos, return_buff=False):    
        dynamic_ligth = cl.Buffer(self.ctx, mf.COPY_HOST_PTR | mf.WRITE_ONLY, hostbuf=ligths.get_numpy_array())
        dynamic_ligth_len = cl.Buffer(self.ctx, mf.COPY_HOST_PTR | mf.WRITE_ONLY, hostbuf=np.array([len(ligths.ligths_array)], np.int32))

        pos_buff = cl.Buffer(self.ctx, mf.COPY_HOST_PTR | mf.READ_ONLY, hostbuf=np.array(canvas_pos, np.int32))

        self.__get_ligth(self.queue, image_size, None, self.canvas_s, self.canvas_size_buff, self.ligth_map_buff,
                         cl.Buffer(self.ctx, mf.COPY_HOST_PTR | mf.READ_ONLY,
                                   hostbuf=np.array(image_size, dtype=np.int32)),
                         cl.Buffer(self.ctx, mf.COPY_HOST_PTR | mf.READ_ONLY,
                                   hostbuf=np.array(position, dtype=np.int32)),
                         self.wall_buffer, self.world_size_buff, pos_buff,
                         self.bg_buffer, self.bg_size_buffer, dynamic_ligth, dynamic_ligth_len)

        self.__get_image(self.queue, image_size, None, self.canvas_s, self.canvas_size_buff, self.im_buff, self.ligth_map_buff,
                         cl.Buffer(self.ctx, mf.COPY_HOST_PTR | mf.READ_ONLY, hostbuf=np.array(image_size, dtype=np.int32)),
                         cl.Buffer(self.ctx, mf.COPY_HOST_PTR | mf.READ_ONLY, hostbuf=np.array(position, dtype=np.int32)),
                          self.wall_buffer, self.world_size_buff, pos_buff,
                          self.bg_buffer, self.bg_size_buffer)
        if not return_buff:
            cl.enqueue_copy(self.queue, im, self.im_buff)
            im = im.reshape((image_size[0], image_size[1], 3))
            return np.array(im, dtype=np.uint8)
        else:
            return self.im_buff, self.ligth_map_buff

    def get_array_from_buff(self, buff, image_size):
        im = np.zeros((image_size[0]*image_size[1]*3, ), dtype=np.int32)
        cl.enqueue_copy(self.queue, im, buff)
        im = im.reshape((image_size[0], image_size[1], 3))
        return np.array(im, dtype=np.uint8)

    def draw_image(self, center, rot_vect, image, canvas_image_buff, canvas_image_sizes, cv_ligth_surf):
        a = np.asarray(image, dtype=np.int32)
        
        sp_buff = cl.Buffer(self.ctx, mf.COPY_HOST_PTR | mf.READ_ONLY, hostbuf = a.reshape(a.shape[0]*a.shape[1]*3))
        cimbuff = cl.Buffer(self.ctx, mf.COPY_HOST_PTR | mf.READ_ONLY, hostbuf=np.array(canvas_image_sizes, dtype=np.int32))
        spimsizebuff = cl.Buffer(self.ctx, mf.COPY_HOST_PTR | mf.READ_ONLY, hostbuf=np.array(image.size, dtype=np.int32))
        cenbuff = cl.Buffer(self.ctx, mf.COPY_HOST_PTR | mf.READ_ONLY, hostbuf=np.array(center, dtype=np.int32))
        rotbuff = cl.Buffer(self.ctx, mf.COPY_HOST_PTR | mf.READ_ONLY, hostbuf=np.array(rot_vect, dtype=np.float32))
        
        self.__draw_image(self.queue, image.size, None, canvas_image_buff, cimbuff, sp_buff, spimsizebuff, cenbuff, rotbuff, cv_ligth_surf)

    def get_force_collision(self, collision, center):
        f = np.zeros([2], dtype=np.int32)
        force_buff = cl.Buffer(self.ctx, mf.COPY_HOST_PTR, hostbuf = f)

        center_buff = cl.Buffer(self.ctx, mf.COPY_HOST_PTR | mf.READ_ONLY, hostbuf = np.array(center, dtype = np.int32))

        im_size = np.array((collision.shape[0], collision.shape[1]), dtype = np.int32)
        im_size_buff = cl.Buffer(self.ctx, mf.COPY_HOST_PTR | mf.READ_ONLY, hostbuf = im_size)
        colli = np.reshape(collision, [im_size[0]*im_size[1]*3])
        coll_buff = cl.Buffer(self.ctx, mf.COPY_HOST_PTR | mf.READ_ONLY, hostbuf = colli)
        self.__get_coll(self.queue, im_size, None, self.canvas_s, self.canvas_size_buff, coll_buff, im_size_buff, center_buff, force_buff)
        cl.enqueue_copy(self.queue, f, force_buff)
        return f

    def explotion(self, position, radius, force = 1):
        self.__explotion(self.queue, (radius*2, radius*2), None, self.canvas_s, self.canvas_size_buff,
         cl.Buffer(self.ctx, mf.COPY_HOST_PTR | mf.READ_ONLY,hostbuf=np.array(radius, dtype=np.int32)),
          cl.Buffer(self.ctx, mf.COPY_HOST_PTR | mf.READ_ONLY,hostbuf=np.array(position, dtype=np.int32)),
          cl.Buffer(self.ctx, mf.COPY_HOST_PTR | mf.READ_ONLY, hostbuf=np.array(force, dtype=np.int32)))

    
    def circle(self, position, radius, block, force = [0, 0]):
        self.__circle(self.queue, (2*radius, 2*radius), None, self.canvas_s, self.canvas_size_buff,
                         cl.Buffer(self.ctx, mf.COPY_HOST_PTR | mf.READ_ONLY,hostbuf=np.array(radius, dtype=np.int32)),
                         cl.Buffer(self.ctx, mf.COPY_HOST_PTR | mf.READ_ONLY, hostbuf=np.array(position, dtype=np.int32)),
                         cl.Buffer(self.ctx, mf.COPY_HOST_PTR | mf.READ_ONLY, hostbuf=np.array(block, dtype=np.int32)),
                      cl.Buffer(self.ctx, mf.COPY_HOST_PTR | mf.READ_ONLY, hostbuf=np.array(force, dtype=np.int32)))

    def force_circle(self, position, radius, force):
        self.__force_circle(self.queue, (2*radius, 2*radius), None, self.canvas_s, self.canvas_size_buff,
                         cl.Buffer(self.ctx, mf.COPY_HOST_PTR | mf.READ_ONLY,hostbuf=np.array(radius, dtype=np.int32)),
                         cl.Buffer(self.ctx, mf.COPY_HOST_PTR | mf.READ_ONLY, hostbuf=np.array(position, dtype=np.int32)),
                            cl.Buffer(self.ctx, mf.COPY_HOST_PTR | mf.READ_ONLY, hostbuf=np.array(force, dtype=np.int32)))
    
    def fill_canvas(self, block):
        self.__fill_canvas(self.queue, self.canvas_size, None, self.canvas_s, self.canvas_size_buff, cl.Buffer(self.ctx, mf.COPY_HOST_PTR | mf.READ_ONLY, hostbuf = np.array([block], dtype= np.int32)))

    def gen_cute_world(self):
        caves_map = self.get_caves_map(self.world_size)
        caves_map = self.map_mul(self.get_blured_map(self.world_size, 20, 2), caves_map, self.world_size)
        caves_map = self.map_mul(self.get_blured_map(self.world_size, 40, 1), caves_map, self.world_size)
        caves_map = self.get_blured_map(self.world_size, 2, 3, caves_map)
        caves_wall_map = self.get_caves_map(self.world_size)
        self.dust_filter(10)
        self.__gen_cute_world(self.queue, self.world_size, None, self.world_buff, self.world_size_buff, caves_map, caves_wall_map, self.wall_buffer)
        #self.errosion_filter(10)

    def dust_filter(self, i = 1):
        for _ in range(i):self.__dust_filter(self.queue, self.world_size, None, self.world_buff, self.world_size_buff, self.wall_buffer)
    
    def errosion_filter(self, i = 1):
        for _ in range(i):self.__errosion_filter(self.queue, self.world_size, (1, 1), self.world_buff, self.world_size_buff, self.wall_buffer)\
        
    def map_mul(self, a, b, sizes):
        size_buff = cl.Buffer(self.ctx, mf.COPY_HOST_PTR | mf.READ_ONLY, hostbuf=np.array(sizes, np.int32))
        self.__map_mul(self.queue, sizes, None, a, b, size_buff)
        return b


    def get_blured_map(self, sizes, r = 10, i = 1, start = None): 
        if type(start) == type(None):
            b1 = cl.Buffer(self.ctx, mf.COPY_HOST_PTR, hostbuf=np.array(np.random.sample(sizes), np.float32))
        else:
            b1 = start
        b2 = cl.Buffer(self.ctx, mf.COPY_HOST_PTR, hostbuf=np.zeros(sizes, np.float32))
        size_buff = cl.Buffer(self.ctx, mf.COPY_HOST_PTR | mf.READ_ONLY, hostbuf=np.array(sizes, np.int32))
        radius_buff = cl.Buffer(self.ctx, mf.COPY_HOST_PTR | mf.READ_ONLY, hostbuf=np.array([r], np.int32))

        for _ in range(i):
            #self.__gen_noise(self.queue, sizes, None, b1, size_buff)
            self.__blur(self.queue, sizes, None, b1, b2, size_buff, radius_buff)
            b1, b2 = b2, b1

        return b1

    def get_caves_map(self, sizes):
        b1 = cl.Buffer(self.ctx, mf.COPY_HOST_PTR, hostbuf=np.array(np.random.sample(sizes), np.float32))
        b2 = cl.Buffer(self.ctx, mf.COPY_HOST_PTR, hostbuf=np.zeros(sizes, np.float32))
        size_buff = cl.Buffer(self.ctx, mf.COPY_HOST_PTR | mf.READ_ONLY, hostbuf=np.array(sizes, np.int32))
        radius_buff = cl.Buffer(self.ctx, mf.COPY_HOST_PTR | mf.READ_ONLY, hostbuf=np.array([30], np.int32))

        for _ in range(2):
            #self.__gen_noise(self.queue, sizes, None, b1, size_buff)
            self.__blur(self.queue, sizes, None, b1, b2, size_buff, radius_buff)
            b1, b2 = b2, b1

        return b1
