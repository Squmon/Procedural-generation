import game_context as gctx
import game_content as gc
game = gctx.pixeled_game((1000, 1000), (3000, 3000), (1920, 1080))
gc.gctx = game
game.run()