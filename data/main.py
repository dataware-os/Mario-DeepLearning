__author__ = 'justinarmstrong'

from . import setup,tools,deep_analyzer
from .states import main_menu,load_screen,level1
from . import constants as c
import threading

def main(deep=False):
    """Add states to control here."""
    if deep is True:
        deep_controller = deep_analyzer.DeepLearningAnalyzer()
        deep_controller.start()

        run_it = tools.Control(setup.DEEP_LEARNING_CAPTION)            
        state_dict = {c.MAIN_MENU: main_menu.Menu(deep_mode=deep),
                    c.LOAD_SCREEN: load_screen.LoadScreen(deep_mode=deep),
                    c.TIME_OUT: load_screen.TimeOut(),
                    c.GAME_OVER: load_screen.GameOver(deep_mode=deep),
                    c.LEVEL1: level1.Level1(deep_mode=deep)}

        run_it.setup_states(state_dict, c.MAIN_MENU)
        run_it.main(deep_controller)

    else:
        run_it = tools.Control(setup.ORIGINAL_CAPTION)
        state_dict = {c.MAIN_MENU: main_menu.Menu(),
                    c.LOAD_SCREEN: load_screen.LoadScreen(),
                    c.TIME_OUT: load_screen.TimeOut(),
                    c.GAME_OVER: load_screen.GameOver(),
                    c.LEVEL1: level1.Level1()}

        run_it.setup_states(state_dict, c.MAIN_MENU)
        run_it.main()



