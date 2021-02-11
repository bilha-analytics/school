'''
author: bg
goal: track & manage performance[acc,time/cost] cascade, log to file worker, 
type: util 
how: 
ref: 
refactors: TODO: entire module after tryzex
'''
#### ===== TODO: refactor 
## callback to que 
class ReportableHandler:
    def log_entry(rec):
        print(rec) 

## cascade data object 
class ReportableSrc(list): ##DataObject 
    def __init__(self, itemz):
        super().__init__(itemz)

## Runnable/Worker + Logger 
class ZReportablezLogger:
    def __init__(self):
        pass
