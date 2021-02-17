'''
author: bg
goal: 
type: Logger and Reports worker 
how: 
ref: 
refactors: 
'''

from datetime import datetime

## TODO: refactor @ worker, design pattern, logger module, fileIO, colored, ZPdDataStats
class ZReporter:
    worker = None 
    class ZWorker(list):
        def __init__(self, name, elements=None):
            super().__init__(self, elements)
            self.name = name
    @staticmethod
    def start(name=None): 
        if ZReporter.worker is None:
            ZReporter.worker = ZWorker(name) 
        ZReporter.worker.start()
    @staticmethod
    def add_log_entry(rec):        
        if ZReporter.worker is not None:
            ZReporter.worker.append( rec ) 
        d = datetime.today().strftime("%d-%m-%Y %H:%M:%S")
        print( f"{d}: {rec}") 


if __name__ == "__main__":
    ZReporter.add_log_entry("We're heading for the stars and the moon is enroute!")
    
