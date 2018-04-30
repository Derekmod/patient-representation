"""A logging tool which I'm developing."""

import os
import sys
from datetime import datetime
import collapse_logging as logging

from threading import Timer


PRINT_INTERVAL = 5
WRITE_INTERVAL = 30


CRITICAL = 4
ERROR = 3
WARNING = 2
DEBUG = 1
INFO = 0
VERBOSITY = DEBUG


NEXT_ID = 0
ROOT = None
_ACTIVE_NODE = None
LOGFILE = None
COOLDOWN_PRINT = None
PENDING_PRINT = False
COOLDOWN_WRITE = None
PENDING_WRITE = False
class LoggerNode(object):
    '''A group of logging statements and sub-nodes.
    Indents the children statements and otherwise manages them.
    Attributes:
        _statements [Statement]: logging statements directly under this node.
        _children [LoggerNode]: nodes nested under this one.
        _dependents ?: ?
        _log_items [(bool, obj)]: statements AND nodes under this node...
            first item of key specifies if object is a NODE
    '''
    def __init__(self, parent=None, verb=None):
        if verb is None:
            verb = logging.VERBOSITY
        self._id = logging.NEXT_ID
        logging.NEXT_ID += 1

        logging._ACTIVE_NODE = self

        self._statements = []
        self._children = []
        self._log_items = []
        self._dependents = []
        self._verb = verb

        self._parent = parent
        if parent is not None:
            parent._children += [self]
            parent._log_items += [(True, self)]

    def addStatement(self, text, args=None, getter=None, verb=None):
        if verb is None:
            verb = self._verb
        statement = Statement(text, args, getter, self, verb=verb)
        return self.addCustomStatement(statement)

    def addCustomStatement(self, statement):
        message = (datetime.now(), statement)
        self._statements += [message]
        self._log_items += [(False, message)]

        statement.node = self
        return statement

    def show(self, tab=0, max_items=50):
        ''' ONLY SHOWS STATEMENTS.'''
        for message in self._statements[-max_items:]:
            self.showStatement(message, tab)

    def showStatement(self, message, tab=0, stream=None):
        ''' Display a single Statement.'''
        time, statement = message
        hour_string = str(time.hour+100)[1:]
        minute_string = str(time.minute+100)[1:]
        second_string = str(time.second+100)[1:]
        milli_string = str(time.microsecond/1000+1000)[1:]
        time_string = '  '*tab + '{}:{}:{}.{}:: '.format(hour_string, minute_string, second_string, milli_string)
        statement_string = statement.text
        if statement.args is not None:
            statement_string = statement_string % tuple(statement.getter(statement.args))
        if stream:
            stream.write(time_string + statement_string + '\n')
        else:
            print time_string + statement_string


    def showAll(self, tab=0, max_items=20, stream=None):
        ''' Show ALL child nodes, and some statements.'''
        for i in range(len(self._log_items)-max_items):
            isNode, item = self._log_items[i]
            if isNode:
                item.showAll(tab=tab+1, max_items=max_items, stream=stream)

        for isNode, item in self._log_items[-max_items:]:
            if isNode:
                item.showAll(tab=tab+1, max_items=max_items, stream=stream)
            else:
                self.showStatement(item, tab, stream=stream)

    def remove(self, statement):
        ''' Remove given node from this node.'''
        for i in range(len(self._statements)):
            time, s = self._statements[i]
            if s is statement:
                del self._statements[i]
                break

        for i in range(len(self._log_items)):
            is_node, message = self._log_items[i]
            if not is_node:
                time, s = message
                if s is statement:
                    del self._log_items[i]
                    break

    def _delete(self):
        ''' Delete this node and remove it from parent.'''
        if self._parent is None:
            return
        global _ACTIVE_NODE
        if self is _ACTIVE_NODE:
            _ACTIVE_NODE = self._parent
        for i in range(len(self._parent._log_items)):
            is_node, node = self._parent._log_items[i]
            if is_node and node is self:
                del self._parent._log_items[i]
                break

    def delete(self):
        garbage = set([self])
        unchecked = set([self])
        while len(unchecked):
            item = unchecked.pop()
            garbage.add(item)
            if isinstance(item, LoggerNode):
                unchecked |= set(item._dependents) - garbage

        for item in garbage:
            item._delete()

    def addDependent(self, other):
        self._dependents += [other]


class Statement(object):
    def __init__(self, text, args=None, getter=None, node=None, verb=None):
        if verb is None:
            verb = logging.VERBOSITY

        self.time = datetime.now()
        self.text = text
        self.args = args
        self.getter = getter
        if getter is None:
            self.getter = lambda(data): data if type(data) is list else [data]
        self.node = node
        self.verb = verb

        self.id = logging.NEXT_ID
        logging.NEXT_ID += 1

    def delete(self):
        if self.node:
            self.node.remove(self)

    def _delete(self):
        self.delete()

class ProgressStatement(Statement):
    def __init__(self, task_name, count, show_time=False, node=None, verb=None):
        #TODO: implement time
        self._start = datetime.now()
        self._count = count
        text = task_name + ' %d/' + str(count)
        args = [0]
        if show_time:
            text += ': %s'
            args += ['? remaining']
        Statement.__init__(self, text, args, node=node, verb=verb)

    def step(self):
        self.args[0] += 1
        if len(self.args) > 1:
            elapsed = datetime.now() - self._start
            if self.args[0] >= self._count:
                self.args[1] = str(elapsed) + ' to complete'
            else:
                estimated = elapsed * (self._count - self.args[0])/(self.args[0])
                self.args[1] = str(estimated) + ' remaining'
        refresh()


def clear():
    #os.system('clear')
    #os.system('cls')
    print('\033[H\033[J')
    #os.system('cls' if os.name == 'nt' else 'clear')
    #print '\n'*3

def log(text, args=None, getter=None, verb=None):
    if verb is None:
        verb = logging.VERBOSITY
    text = str(text)
    statement = logging._ACTIVE_NODE.addStatement(text, args, getter)
        
    refresh()
    return statement

def logProgress(task_name, count, show_time=True):
    global _ACTIVE_NODE
    statement = ProgressStatement(task_name, count, show_time, node=_ACTIVE_NODE)
    _ACTIVE_NODE.addCustomStatement(statement)

    refresh()
    return statement

def _task_descriptor(node_ptr):
    node = node_ptr[0]
    description = 'in progress'
    if node is not _ACTIVE_NODE:
        description = 'DONE'
    if node is None:
        description = 'starting'

    return [description]

def addNode(task_name=None, count=0):
    statement = None
    if count > 0:
        if task_name is None:
            task_name = ''
        statement = logProgress(task_name, count)
    elif task_name is not None:
        args = [None]
        getter = _task_descriptor
        statement = log(task_name + ' %s', args, getter)

    global _ACTIVE_NODE
    _ACTIVE_NODE = LoggerNode(_ACTIVE_NODE)
    if statement:
        _ACTIVE_NODE.addDependent(statement)
    if task_name is not None and not statement:
        args[0] = _ACTIVE_NODE
    return statement
    #return _ACTIVE_NODE._id

def replaceNode(task_name=None, count=0):
    logging.closeNode()
    logging.addNode(task_name, count)

def closeNode():
    global _ACTIVE_NODE
    _ACTIVE_NODE = _ACTIVE_NODE._parent
    return _ACTIVE_NODE._id

def deleteNode(key=0):
    global _ACTIVE_NODE
    next = _ACTIVE_NODE._parent
    next._children.remove(_ACTIVE_NODE)
    next._log_items.remove((True, _ACTIVE_NODE))
    del _ACTIVE_NODE
    _ACTIVE_NODE = next

def exit():
    print 'EXITING'
    #clear()
    #logging.ROOT.showAll()
    logging.COOLDOWN_WRITE = None
    logging._refreshWrite()
    sys.exit()

def refresh():
    logging._refreshPrint()
    logging._refreshWrite()

def _refreshPrint():
    if PRINT_INTERVAL < 0:
        return
    if logging.COOLDOWN_PRINT is not None:
        logging.PENDING_PRINT = True
        return

    logging.COOLDOWN_PRINT = Timer(logging.PRINT_INTERVAL, logging._resetPrintCooldown)
    ancestry = [_ACTIVE_NODE]
    while ancestry[-1]._parent is not None:
        ancestry += [ancestry[-1]._parent]
    ancestry.reverse()

    clear()
    
    for i in range(len(ancestry)):
        node = ancestry[i]
        node.show(tab=i)

    logging.COOLDOWN_PRINT.start()
    logging.PENDING_PRINT = False


def _refreshWrite():
    if WRITE_INTERVAL < 0:
        return

    if not logging.LOGFILE:
        return

    if logging.COOLDOWN_WRITE is not None:
        logging.PENDING_WRITE = True
        return

    logging.COOLDOWN_WRITE = Timer(logging.WRITE_INTERVAL, logging._resetWriteCooldown)

    #if logging.LOGFILE == logging.LOGFILE1:
    #    logging.LOGFILE = logging.LOGFILE2
    #else:
    #    logging.LOGFILE = logging.LOGFILE1

    stream = open(logging.LOGFILE, 'w')
    logging.ROOT.showAll(stream=stream, max_items=200)
    stream.close()

    logging.COOLDOWN_WRITE.start()
    logging.PENDING_WRITE = False


def set_logfile(logfile=LOGFILE):
    logging.LOGFILE = logfile

def _resetPrintCooldown():
    logging.COOLDOWN_PRINT = None
    if logging.PENDING_PRINT:
        logging._refreshPrint()

def _resetWriteCooldown():
    logging.COOLDOWN_WRITE = None
    if logging.PENDING_WRITE:
        logging._refreshWrite()


#setup
ROOT = LoggerNode()


