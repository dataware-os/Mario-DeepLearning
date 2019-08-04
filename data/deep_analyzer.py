__author__ = "lukasz.ksiezak"
import threading
import time

class DeepLearningAnalyzer(threading.Thread):    
    
    _end_signal = True

    def __init__(self):
        threading.Thread.__init__(self)
    
    def run(self):        
        print("Starting thread with deep learning analyzer")
        self._end_signal = False
        while(self._end_signal is False):
            pass
    
    def stop(self):
        print("Killing deep learning api mode")
        self._end_signal = True
