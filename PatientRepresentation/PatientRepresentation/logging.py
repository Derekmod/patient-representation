"""A rudimentary logging tool which I'm developing."""

import os
from datetime import datetime

NEXT_ID = 0
ROOT = None
ACTIVE_NODE = None
class LoggerNode(object):

    def __init__(self, parent=None):
        global NEXT_ID
        self._id = NEXT_ID
        NEXT_ID += 1

        global ACTIVE_NODE
        ACTIVE_NODE = self

        self._statements = []
        self._children = []

        self._parent = parent
        if parent is not None:
            parent._children += [self]

    def addStatement(self, text, args=None):
        statement = Statement(text, args)
        self._statements += [Statement(text, args)]
        return statement

    def show(self, all=False, tab=0):
        start = max(0, len(self._statements)-50)
        for message in self._statements[start:]:
            time = message[0]
            hour_string = str(time.hour+100)[1:]
            minute_string = str(time.minute+100)[1:]
            second_string = str(time.second+100)[1:]
            milli_string = str(time.microsecond/1000+1000)[1:]
            time_string = '  '*tab + '{}:{}:{}.{}:: '.format(hour_string, minute_string, second_string, milli_string)
            statement_string = str(message[1])
            if message[2] is not None:
                statement_string = statement_string % tuple(message[2])
            print time_string + statement_string

        if all:
            for child in self._children:
                child.show(all=all, tab=tab+1)

class Statement(object):
    def __init__(self, text, args):
        self.time = datetime.now()
        self.text = text
        self.args = args

        global NEXT_ID
        self.id = NEXT_ID
        NEXT_ID += 1

def clear():
    #os.system('clear')
    os.system('cls')

def log(statement, args=None):
    global ACTIVE_NODE
    global ROOT
    statement = ACTIVE_NODE.addStatement(statement, args)
        
    refresh()
    return statement

def newNode():
    global ACTIVE_NODE
    ACTIVE_NODE = LoggerNode(ACTIVE_NODE)
    return ACTIVE_NODE._id

def closeNode():
    global ACTIVE_NODE
    ACTIVE_NODE = ACTIVE_NODE._parent
    return ACTIVE_NODE._id

def exit():
    global ROOT
    os.system('pause')
    clear()
    ROOT.show(all=True)

def refresh():
    ancestry = [ACTIVE_NODE]
    while ancestry[-1]._parent is not None:
        ancestry += [ancestry[-1]._parent]
    ancestry.reverse()

    clear()

    for i in range(len(ancestry)):
        node = ancestry[i]
        node.show(tab=i)


#setup
ROOT = LoggerNode()


