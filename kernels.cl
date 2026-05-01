#define get_current_pixel get_pixel(pixel_pos, canvas_size)
#define image_pixel(i) image[x*3 + y*canvas_size[0]*3 + i]

#define gravity 1
#define AIR 0
#define SAND 1
#define STONE 2
#define SAND_STONE 3 
#define DESTROING_MATTER 4
#define UNBREKABLE 11

uint random(uint2 randoms, int b){
    uint seed = randoms.x + b;
    uint t = seed ^ (seed << 11);  
    return randoms.y ^ (randoms.y >> 19) ^ (t ^ (t >> 8));
}

int get_pixel(int2 coords, int*canvas_size){
    return (coords.y*canvas_size[0] + coords.x)*canvas_size[2];
}

int cut(int value, int minv, int maxv){
    return min(max(value, minv), maxv - 1);
}

float2 complex_mul(float2 a, float2 b){
    return (float2)(a.x*b.x - a.y*b.y, a.x*b.y + a.y*b.x);
}

__kernel void fill_canvas(__global int* canvas, __global const int* canvas_size, __global const int* number){
    int2 pixel_pos = (int2)(get_global_id(0), get_global_id(1));
    canvas[get_current_pixel] = number[0];
}

void move_pixel(int id, int ux, int uy, int2 next_pos, int* output_canvas, int*canvas_size){
    output_canvas[get_pixel(next_pos, canvas_size) + 1] += ux;
    output_canvas[get_pixel(next_pos, canvas_size) + 2] += uy + gravity;
    output_canvas[get_pixel(next_pos, canvas_size) + 0] = id;
}


__kernel void noise_generator(__global float* im, __global const int* im_size){
    int2 pixel_pos = (int2)(get_global_id(0), get_global_id(1));

    int cave_step1 = 30;
    int cave_step2 = 60;

    float2 p2 = 20000.0f*((float2)((float) (cave_step1*(pixel_pos.x/cave_step1)), (float) (cave_step1*(pixel_pos.y/cave_step1))) + (float2) (300.0f, 2000.0f));
    float2 p3 = 200.0f*((float2)((float) (cave_step2*(pixel_pos.x/cave_step2)), (float) (cave_step2*(pixel_pos.y/cave_step2))) + (float2) (30033.0f, 200.0f));
    float r = (sin(dot(p2, p2)) + sin(dot(p3, p3)))/2.0f;
    if (r > -0.2f){
        im[pixel_pos.x + pixel_pos.y * im_size[0]] = r;
    }
}


__kernel void blur(__global float* im1, __global float* im2, __global const int* im_size, __global const int* radius){
    int2 pixel_pos = (int2)(get_global_id(0), get_global_id(1));
    float k = 0.0f;
    int r = *radius;
    for (int lx = pixel_pos.x - r; lx <= pixel_pos.x + r; lx++){
        for (int ly = pixel_pos.y - r; ly <= pixel_pos.y + r; ly++){
            if ((lx > 0 && lx < im_size[0]) && (ly > 0 && ly < im_size[1])){
                k += im1[lx + ly * im_size[0]];
            }
        }
    }

    im2[pixel_pos.x + pixel_pos.y * im_size[0]] = k/((r*2 + 1)*(r*2 + 1));
}

__kernel void map_mul(__global float* im1, __global float* im2, __global const int* im_size){
    int2 pixel_pos = (int2)(get_global_id(0), get_global_id(1));
    int q = pixel_pos.x + pixel_pos.y * im_size[0];
    im2[q] = sqrt(im2[q]*im1[q]);
}


__kernel void gen_cute_world(__global int* canvas, __global const int* canvas_size, __global const float* caves_map, __global const float* caves_wall_map, __global float* wall_array){
    int2 pixel_pos = (int2)(get_global_id(0), get_global_id(1));
    wall_array[canvas_size[0]*3*pixel_pos.y + pixel_pos.x*3 + 0] = -1.0f;

    float r = caves_map[pixel_pos.x + pixel_pos.y * canvas_size[0]];
    float r1 = caves_wall_map[pixel_pos.x + pixel_pos.y * canvas_size[0]];
    if (r > 0.2f){
        if (r < 0.5f){
            canvas[get_current_pixel] = STONE;
            //wall_array[canvas_size[0]*3*pixel_pos.y + pixel_pos.x*3 + 0] = 0.2f;
            //wall_array[canvas_size[0]*3*pixel_pos.y + pixel_pos.x*3 + 1] = 0.2f;
            //wall_array[canvas_size[0]*3*pixel_pos.y + pixel_pos.x*3 + 2] = 0.2f;
        }
        else{
            canvas[get_current_pixel] = AIR;
        }
    }
    if (r1 > 0.2f && r1 < 0.5f){
        wall_array[canvas_size[0]*3*pixel_pos.y + pixel_pos.x*3 + 0] = 0.2f;
        wall_array[canvas_size[0]*3*pixel_pos.y + pixel_pos.x*3 + 1] = 0.2f;
        wall_array[canvas_size[0]*3*pixel_pos.y + pixel_pos.x*3 + 2] = 0.2f;
    }
}



__kernel void dust_filter(__global int* canvas, __global const int* canvas_size, __global float* wall_array){
    int2 pixel_pos = (int2)(get_global_id(0), get_global_id(1));

    if ((canvas[get_pixel(pixel_pos + (int2)(0, -1), canvas_size)] == STONE || canvas[get_pixel(pixel_pos + (int2)(0, -1), canvas_size)] == SAND_STONE) && canvas[get_current_pixel] == AIR){
        canvas[get_current_pixel] = SAND_STONE;
    }
}

__kernel void errosion_filter(__global int* canvas, __global const int* canvas_size, __global float* wall_array){
    int2 pixel_pos = (int2)(get_global_id(0), get_global_id(1));
    int I = 0;

    

    for (int dx = pixel_pos.x - 1; dx <= pixel_pos.x + 1; dx++){
        for (int dy = pixel_pos.y - 1; dy <= pixel_pos.y + 1; dy++){
            if (dx > 0 && dx < canvas_size[0] && dy > 0 && dy < canvas_size[1]){
                if (canvas[get_pixel((int)(dx, dy), canvas_size)] == AIR){
                    I++;
                }
            }
        }
    }
    if (canvas[get_current_pixel] == STONE){
        /*I = 0;
        I += canvas[get_pixel(pixel_pos + (int)(1, 0), canvas_size)] == AIR;
        I += canvas[get_pixel(pixel_pos + (int)(0, 1), canvas_size)] == AIR;
        I += canvas[get_pixel(pixel_pos + (int)(-1, 0), canvas_size)] == AIR;
        I += canvas[get_pixel(pixel_pos + (int)(0, -1), canvas_size)] == AIR;
        I += canvas[get_pixel(pixel_pos + (int)(1, 1), canvas_size)] == AIR;
        I += canvas[get_pixel(pixel_pos + (int)(-1, -1), canvas_size)] == AIR;
        I += canvas[get_pixel(pixel_pos + (int)(1, -1), canvas_size)] == AIR;
        I += canvas[get_pixel(pixel_pos + (int)(-1, 1), canvas_size)] == AIR;*/

        if (I >= 4){
            canvas[get_current_pixel] = AIR;
            wall_array[canvas_size[0]*3*pixel_pos.y + pixel_pos.x*3 + 0] = 1.0f;
        }
    }
    
}


__kernel void update_canvas(__global const int* input_canvas, __global int* output_canvas, __global const int* canvas_size, __global const float* time, __global const int* center, __global const int* update_zone_scale){
    int2 pixel_pos = (int2)(get_global_id(0), get_global_id(1)) - (int2)(update_zone_scale[0]/2, update_zone_scale[1]/2) + (int2)(center[0], center[1]);
    if (pixel_pos.x > 0 && pixel_pos.y > 0 && pixel_pos.x < canvas_size[0] && pixel_pos.y < canvas_size[1]){
        int value = input_canvas[get_current_pixel];

        int2 u = (int2)(input_canvas[get_current_pixel + 1], input_canvas[get_current_pixel + 2]);
        int2 next_pos = pixel_pos + u;
        int a = 0;
        int2 lp;
        next_pos.x = cut(next_pos.x, 0, canvas_size[0]);
        next_pos.y = cut(next_pos.y, 0, canvas_size[1]);
        int g = 1;
        int2 up = pixel_pos + (int2)(0, -1);
        up.x = cut(up.x, 0, canvas_size[0]);
        up.y = cut(up.y, 0, canvas_size[1]);
        int2 down = pixel_pos + (int2)(0, 1);
        down.x = cut(down.x, 0, canvas_size[0]);
        down.y = cut(down.y, 0, canvas_size[1]);

        switch (value){
            case AIR:
                break;
            case SAND_STONE:
            case SAND:
                next_pos.x = cut(next_pos.x + (2*((int)random((uint)(next_pos.x, next_pos.y), next_pos.x*next_pos.y + (uint) (*time)) % 2) - 1), 0, canvas_size[0]);
                next_pos.y = cut(next_pos.y, 0, canvas_size[1]); 
                if (input_canvas[get_pixel(next_pos, canvas_size)] != 0 || output_canvas[get_pixel(next_pos, canvas_size)] != 0){
                    next_pos = pixel_pos;
                    u *= STONE == output_canvas[get_pixel(next_pos, canvas_size)];
                    output_canvas[get_pixel(next_pos, canvas_size) + 1] += u.x;
                    output_canvas[get_pixel(next_pos, canvas_size) + 2] += u.y;
                    g = 2;
                }
                move_pixel(value, u.x/g, u.y/g, next_pos, output_canvas, canvas_size);
                break;
            case STONE:
                if (input_canvas[get_pixel(next_pos, canvas_size)] != 0 || output_canvas[get_pixel(next_pos, canvas_size)] != 0){
                    next_pos = pixel_pos;
                }
                if (u.x*u.x + u.y*u.y > 16 || (input_canvas[get_pixel(up, canvas_size)] == AIR && input_canvas[get_pixel(down, canvas_size)] == AIR)){
                    move_pixel(SAND_STONE, 0, 0, next_pos, output_canvas, canvas_size);
                }
                else{
                    move_pixel(STONE, 0, -gravity, next_pos, output_canvas, canvas_size);
                }
                break;
            case DESTROING_MATTER:
                for (int lx = pixel_pos.x - 1; lx <= pixel_pos.x + 1; lx++){
                    for (int ly = pixel_pos.y - 1; ly <= pixel_pos.y + 1; ly++){
                        lp = (int2)(cut(lx, 0, canvas_size[0]), cut(ly, 0, canvas_size[1]));
                        if (input_canvas[get_pixel(lp, canvas_size)] == DESTROING_MATTER){
                            a++;
                        }
                        else if(input_canvas[get_pixel(lp, canvas_size)] != UNBREKABLE && input_canvas[get_pixel(lp, canvas_size)] != AIR){
                            output_canvas[get_pixel(lp, canvas_size) + 1] += -(pixel_pos - lp).x;
                            output_canvas[get_pixel(lp, canvas_size) + 2] += -(pixel_pos - lp).y;
                        }
                    }
                }
                if (a < 8){
                    move_pixel(AIR, u.x, u.y, next_pos, output_canvas, canvas_size);
                    }
                else{
                    move_pixel(DESTROING_MATTER, u.x, u.y, next_pos, output_canvas, canvas_size);
                }
                break;
            case UNBREKABLE:
                output_canvas[get_pixel(pixel_pos, canvas_size) + 0] = UNBREKABLE;
            default:
                break;
        }
    }
}


__kernel void copy_canvas(__global const int* input_canvas, __global int* output_canvas, __global const int* canvas_size){
    int2 pixel_pos = (int2)(get_global_id(0), get_global_id(1));
    output_canvas[get_current_pixel + 0] = input_canvas[get_current_pixel + 0];
    output_canvas[get_current_pixel + 1] = input_canvas[get_current_pixel + 1];
    output_canvas[get_current_pixel + 2] = input_canvas[get_current_pixel + 2];
}



__kernel void fill_zeros(__global int* canvas, __global const int* canvas_size, __global const int* centeri, __global const int* sizes){
    int x = get_global_id(0);
    int y = get_global_id(1);
    int2 pixel_pos = (int2)(x - sizes[0]/2, y - sizes[1]/2) + (int2)(centeri[0], centeri[1]);;

    pixel_pos.x = cut(pixel_pos.x, 0, canvas_size[0]);
    pixel_pos.y = cut(pixel_pos.y, 0, canvas_size[1]);

    canvas[get_current_pixel + 0] = 0;
    canvas[get_current_pixel + 1] = 0;
    canvas[get_current_pixel + 2] = 0;
}



__kernel void get_canvas_from_world(__global const int* world, __global const int* world_size, __global int* canvas, __global const int* canvas_size, __global const int *left_up){
    int x = get_global_id(0);
    int y = get_global_id(1);
    int2 pixel_pos = (int2)(x + left_up[0], y + left_up[1]);
    int3 params = (int3)(UNBREKABLE, 0, 0);
    if (pixel_pos.x < world_size[0] && pixel_pos.y < world_size[1] && pixel_pos.x > 0 && pixel_pos.y > 0){
        int id = pixel_pos.x*world_size[2] + pixel_pos.y*world_size[2]*world_size[0];
        params.r = world[id + 0];
        params.g = world[id + 1];
        params.b = world[id + 2];
    }
    canvas[x*3 + y*3*canvas_size[0] + 0] = params.r; 
    canvas[x*3 + y*3*canvas_size[0] + 1] = params.g; 
    canvas[x*3 + y*3*canvas_size[0] + 2] = params.b;
}


__kernel void update_world_from_canvas(__global int* world, __global const int* world_size, __global const int* canvas, __global const int* canvas_size, __global const int *left_up){
    int x = get_global_id(0);
    int y = get_global_id(1);
    int2 pixel_pos = (int2)(x + left_up[0], y + left_up[1]);
    int3 params = 0;
    if (pixel_pos.x < world_size[0] && pixel_pos.y < world_size[1] && pixel_pos.x > 0 && pixel_pos.y > 0){
        params.r = canvas[x*3 + y*3*canvas_size[0] + 0];
        params.g = canvas[x*3 + y*3*canvas_size[0] + 1];
        params.b = canvas[x*3 + y*3*canvas_size[0] + 2];
        int id = pixel_pos.x*3 + pixel_pos.y*3*world_size[0];
        world[id + 0] = params.r; 
        world[id + 1] = params.g;
        world[id + 2] = params.b;
    }
}



__kernel void get_image(__global const int* canvas, __global const int* canvas_size, __global int* image, __global int* ligth_map, __global const int* image_size,
 __global const int *left_up, __global const float* walls, __global const int* walls_array_size, __global const int* canvas_pos,
 __global const int* background, __global const int* background_sizes){
    int x = get_global_id(0);
    int y = get_global_id(1);
    int2 pixel_pos = (int2)(x + left_up[0], y + left_up[1]);
    float3 color = (float3)(1.0f, 0.0f, 1.0f);

    int2 gp = pixel_pos + (int2)(canvas_pos[0], canvas_pos[1]) - (int2)(canvas_size[0], canvas_size[1])/2;
    
    int wall_id = walls_array_size[0]*3*gp.y + 3*gp.x;
    float rand = ((float)(random((uint2)((uint) pixel_pos.x, (uint) pixel_pos.y), pixel_pos.x*pixel_pos.y)%100))/100.0f;

    if (pixel_pos.x < canvas_size[0] && pixel_pos.y < canvas_size[1] && pixel_pos.x > 0 && pixel_pos.y > 0 &&
        gp.x < walls_array_size[0] && gp.y < walls_array_size[1] && gp.x > 0 && gp.y > 0){
        int value = canvas[get_current_pixel];
        if (value != AIR || walls[wall_id] >= 0.0f){
            switch (value){
                case AIR:
                    color.r = walls[wall_id + 0];
                    color.g = walls[wall_id + 1];
                    color.b = walls[wall_id + 2];
                    break;
                case SAND:
                    color.r = 240.0f/255.0f;
                    color.g = 200.0f/255.0f;
                    color.b = 100.0f/255.0f;
                    color *= 0.5f*rand + 0.5f;
                    break;
                case STONE:
                    color = 0.5f * (0.2f*rand + 0.8f);
                    break;
                case SAND_STONE:
                    color = 0.7f;
                    break;
                case DESTROING_MATTER:
                    color = 0.f;
                    color.r = 1.0f;
                    color.b = 1.0f;
                    break;

                case UNBREKABLE:
                    color = 0;
                    color.r = 1.0f;
                    break;
            }
        }
        else{
            //color = 0.01f;
            //int id = 3*(gp.x%background_sizes[0] + gp.y%background_sizes[1] * background_sizes[0]);
            int id = 3*((x + gp.x/10)%background_sizes[0] + (y + gp.y/10)%background_sizes[1] * background_sizes[0]);
            color.r = ((float)background[id + 0])/255.0f;
            color.g = ((float)background[id + 1])/255.0f;
            color.b = ((float)background[id + 2])/255.0f;
        }
    }
    int iid = y*3 + x*3*image_size[1];
    image[iid + 0] = (int)(ligth_map[iid + 0]*color.r); 
    image[iid + 1] = (int)(ligth_map[iid + 1]*color.g); 
    image[iid + 2] = (int)(ligth_map[iid + 2]*color.b);
}

float3 get_ligth_intensivity(int id){
    switch (id){
        case SAND:
            return 5.0f*(float3)(240.0f/255.0f, 200.0f/255.0f, 100.0f/255.0f);
            break;
        case DESTROING_MATTER:
            return 4.0f * (float3) (1.0f, 0.0f, 1.0f);
        default:
            return 0.f;
    }
}
float3 get_shadow_intensivity(int id){
    switch (id){
        case AIR:
            return 0.4f;
        default:
            return 1.f;
    }
}

__kernel void get_ligth_map(__global const int* canvas, __global const int* canvas_size, __global int* ligth_map, __global const int* ligth_map_size,
 __global const int *left_up, __global const float* walls, __global const int* walls_array_size, __global const int* canvas_pos,
 __global const int* background, __global const int* background_sizes, __global const float* dynamic_ligths, __global const int* dynamic_ligths_count){
    int x = get_global_id(0);
    int y = get_global_id(1);
    int2 pixel_pos = (int2)(x + left_up[0], y + left_up[1]);
    float3 ligth = 0.0f;
    float3 ligth1 = 0.f;//length(((float2)((float)x, (float)y)) - ((float2)((float)ligth_map_size[0], (float)ligth_map_size[1]))/2.0f)/100.0f;
    for (int i = 0; i < dynamic_ligths_count[0]; i++){
        int lid = i*5;
        float2 pos = (float2)(dynamic_ligths[lid + 0], dynamic_ligths[lid + 1]);
        float3 color = (float3)(dynamic_ligths[lid + 2], dynamic_ligths[lid + 3], dynamic_ligths[lid + 4]);
        ligth1 += color*max(1.0f - length(((float2)((float)x, (float)y)) - pos)/100.0f, 0.f);
    }

    ligth1 = min(ligth1, 1.f);

    int2 gp = pixel_pos + (int2)(canvas_pos[0], canvas_pos[1]) - (int2)(canvas_size[0], canvas_size[1])/2;
    
    int wall_id = walls_array_size[0]*3*gp.y + 3*gp.x;

    if (pixel_pos.x < canvas_size[0] && pixel_pos.y < canvas_size[1] && pixel_pos.x > 0 && pixel_pos.y > 0 &&
        gp.x < walls_array_size[0] && gp.y < walls_array_size[1] && gp.x > 0 && gp.y > 0){
        int value = canvas[get_current_pixel];
        if (value != AIR || walls[wall_id] >= 0.0f){
            int2 lp = 0;
            int2 lgp = 0;
            float3 ligth2 = 0.0f;
            for (int lx = pixel_pos.x - 10; lx <= pixel_pos.x + 10; lx++){
                for (int ly = pixel_pos.y - 10; ly <= pixel_pos.y + 10; ly++){
                    lp = (int2)(cut(lx, 0, canvas_size[0]), cut(ly, 0, canvas_size[1]));
                    lgp = (int2)(cut(lp.x + canvas_pos[0] - canvas_size[0]/2, 0, walls_array_size[0]), cut(lp.y + canvas_pos[1] - canvas_size[1]/2, 0, walls_array_size[1]));
                    if (canvas[get_pixel(lp, canvas_size)] != 0 || walls[lgp.x*3 + lgp.y * 3 * walls_array_size[0]] > 0.0f){
                        ligth2 -= 1.0f;
                    }
                    if (canvas[get_pixel(lp, canvas_size)] != 0){
                        ligth += 1.0f;
                    }
                    ligth2 += get_ligth_intensivity(canvas[get_pixel(lp, canvas_size)]);
                    //ligth += get_ligth_intensivity(canvas[get_pixel(lp, canvas_size)]);
                }
            }
            ligth2 = 1.0f + ligth2/441.0f;
            ligth = 1.0f - ligth/441.0f;
            ligth1 *= ligth1;
            ligth = (ligth*(ligth1) + ligth2)/2.0f;
            ligth = min(ligth, 1.0f);
        }
        else{
            ligth = 1.0f;
        }
    }

    ligth_map[y*3 + x*3*ligth_map_size[1] + 0] = (int)(255.0f*ligth.r); 
    ligth_map[y*3 + x*3*ligth_map_size[1] + 1] = (int)(255.0f*ligth.g); 
    ligth_map[y*3 + x*3*ligth_map_size[1] + 2] = (int)(255.0f*ligth.b);
}



__kernel void draw_image(__global int* image, __global const int* image_size, __global const int* sprite,
 __global const int* sprite_size, __global const int* position , __global const float* rotation_vec, __global const int* ligth_map){
    int2 gid = (int2)(get_global_id(0), get_global_id(1));
    float2 H = complex_mul((float2)((float) gid.x - sprite_size[0]/2,
     (float) gid.y - sprite_size[1]/2), (float2)(rotation_vec[0], rotation_vec[1]));
    int2 tgid = (int2)((int) H.x + sprite_size[0]/2, (int) H.y + sprite_size[1]/2);
    
    int2 center = (int2)(position[0], position[1]);
    int2 pos = tgid + center - (int2)(sprite_size[0], sprite_size[1])/2;
    if((pos.x > 0 && pos.y > 0) && (pos.x < image_size[0] && pos.y < image_size[1])){
        int sp_ind = gid.x*3 + gid.y*3*sprite_size[0];
        int im_ind = pos.y*3 + pos.x*3*image_size[1];
        int3 color = 0;
        color.r = sprite[sp_ind + 0];
        color.g = sprite[sp_ind + 1];
        color.b = sprite[sp_ind + 2];
        
        if (color.r != 0 || color.g != 0 || color.b != 0){
            image[im_ind + 0] = (int)((float)color.r)*(((float)ligth_map[im_ind + 0])/255.0f);
            image[im_ind + 1] = (int)((float)color.g)*(((float)ligth_map[im_ind + 1])/255.0f);
            image[im_ind + 2] = (int)((float)color.b)*(((float)ligth_map[im_ind + 2])/255.0f);
        }
    }
}


__kernel void explotion(__global int* canvas, __global const int* canvas_size, __global const int* radius, __global const int *position, __global const int* force){
    int x = get_global_id(0);
    int y = get_global_id(1);
    int r = radius[0];
    int2 center = (int2)(position[0], position[1]);
    int2 pixel_pos = (int2)(x - r, y - r) + center;

    pixel_pos.x = cut(pixel_pos.x, 0, canvas_size[0]);
    pixel_pos.y = cut(pixel_pos.y, 0, canvas_size[1]);

    center.x = cut(center.x, 0, canvas_size[0]);
    center.y = cut(center.y, 0, canvas_size[1]);

    int2 u = pixel_pos - center;
    int l = (u.x*u.x + u.y*u.y)/force[0];
    if (l != 0){
        u = u/l;
    }
    if (l < r*r){
        canvas[get_current_pixel + 1] += u.x;
        canvas[get_current_pixel + 2] += u.y;
    }

}


__kernel void circle(__global int* canvas, __global const int* canvas_size, __global const int* radius, __global const int *position, __global const int* block, __global const int* force){
    int x = get_global_id(0);
    int y = get_global_id(1);
    int r = radius[0];
    int2 center = (int2)(position[0], position[1]);
    int2 pixel_pos = (int2)(x - r, y - r) + center;

    pixel_pos.x = cut(pixel_pos.x, 0, canvas_size[0]);
    pixel_pos.y = cut(pixel_pos.y, 0, canvas_size[1]);

    center.x = cut(center.x, 0, canvas_size[0]);
    center.y = cut(center.y, 0, canvas_size[1]);

    int2 u = pixel_pos - center;
    int l = (u.x*u.x + u.y*u.y);
    if (l < r*r){
        canvas[get_current_pixel] = block[0];
        canvas[get_current_pixel + 1] = force[0];
        canvas[get_current_pixel + 2] = force[1];
    }
}

__kernel void force_circle(__global int* canvas, __global const int* canvas_size, __global const int* radius, __global const int *position, __global const int* force){
    int x = get_global_id(0);
    int y = get_global_id(1);
    int r = radius[0];
    int2 center = (int2)(position[0], position[1]);
    int2 pixel_pos = (int2)(x - r, y - r) + center;

    pixel_pos.x = cut(pixel_pos.x, 0, canvas_size[0]);
    pixel_pos.y = cut(pixel_pos.y, 0, canvas_size[1]);

    center.x = cut(center.x, 0, canvas_size[0]);
    center.y = cut(center.y, 0, canvas_size[1]);

    int2 u = pixel_pos - center;
    int l = (u.x*u.x + u.y*u.y);
    if (l < r*r){
        canvas[get_current_pixel + 1] += force[1];
        canvas[get_current_pixel + 2] += force[2];
    }
}

__kernel void big_block(__global int* canvas, __global const int* canvas_size, __global const int* radius, __global const int *position, __global const int* block){
    int x = get_global_id(0);
    int y = get_global_id(1);
    int r = radius[0];
    int2 center = (int2)(position[0], position[1]);
    int2 pixel_pos = (int2)(x - r/2, y - r/2) + center;

    pixel_pos.x = cut(pixel_pos.x, 0, canvas_size[0]);
    pixel_pos.y = cut(pixel_pos.y, 0, canvas_size[1]);

    int2 u = pixel_pos - center;
    int l = (u.x*u.x + u.y*u.y);
    if (l < r*r){
        canvas[get_current_pixel] = block[0];
    }
}


__kernel void get_force(__global int* canvas, __global const int* canvas_size, __global const int* image, __global const int* image_size, __global const int *center, __global int* force){
    int x = get_global_id(0);
    int y = get_global_id(1);
    int2 pixel_pos = (int2)(x + center[0] - image_size[0]/2, y + center[1] - image_size[1]/2);
    int2 c = (int2)(center[0], center[1]);
    
    if (pixel_pos.x < canvas_size[0] && pixel_pos.y < canvas_size[1] && pixel_pos.x > 0 && pixel_pos.y > 0){
        if (image[x*3 + y*3*image_size[0]] != 0){
            if (canvas[get_current_pixel] != 0){
                int2 f = (int2)(x - image_size[0]/2, y - image_size[1]/2);
                force[0] -= f.x;
                force[1] -= f.y;
            }
        }
    }
}