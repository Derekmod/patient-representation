"""A rudimentary logging tool which I'm developing."""

import os
from datetime import datetime

NEXT_ID = 0
ROOT = None
_ACTIVE_NODE = None
class LoggerNode(object):

    def __init__(self, parent=None):
        global NEXT_ID
        self._id = NEXT_ID
        NEXT_ID += 1

        global _ACTIVE_NODE
        _ACTIVE_NODE = self

        self._statements = []
        self._children = []
        self._log_items = []
        self._dependents = []

        self._parent = parent
        if parent is not None:
            parent._children += [self]
            parent._log_items += [(True, self)]

    def addStatement(self, text, args=None, getter=None):
        statement = Statement(text, args, getter, self)
        return self.addCustomStatement(statement)

    def addCustomStatement(self, statement):
        message = (datetime.now(), statement)
        self._statements += [message]
        self._log_items += [(False, message)]

        statement.node = self
        return statement

    def show(self, tab=0, max_items=50):
        for message in self._statements[-max_items:]:
            self.showStatement(message, tab)

    def showStatement(self, message, tab=0):
        time, statement = message
        hour_string = str(time.hour+100)[1:]
        minute_string = str(time.minute+100)[1:]
        second_string = str(time.second+100)[1:]
        milli_string = str(time.microsecond/1000+1000)[1:]
        time_string = '  '*tab + '{}:{}:{}.{}:: '.format(hour_string, minute_string, second_string, milli_string)
        statement_string = statement.text
        if statement.args is not None:
            statement_string = statement_string % tuple(statement.getter(statement.args))
        print time_string + statement_string


    def showAll(self, tab=0, max_items=20):
        for i in range(len(self._log_items)-max_items):
            isNode, item = self._log_items[i]
            if isNode:
                item.showAll(tab=tab+1, max_items=max_items)

        for isNode, item in self._log_items[-max_items:]:
            if isNode:
                item.showAll(tab=tab+1, max_items=max_items)
            else:
                self.showStatement(item, tab)

    def remove(self, statement):
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
    def __init__(self, text, args=None, getter=None, node=None):
        self.time = datetime.now()
        self.text = text
        self.args = args
        self.getter = getter
        if getter is None:
            self.getter = lambda(data): data if type(data) is list else [data]
        self.node = node

        global NEXT_ID
        self.id = NEXT_ID
        NEXT_ID += 1

    def delete(self):
        if self.node:
            self.node.remove(self)

    def _delete(self):
        self.delete()

class ProgressStatement(Statement):
    def __init__(self, task_name, count, show_time=False, node=None):
        #TODO: implement time
        self._start = datetime.now()
        self._count = count
        text = task_name + ' %d/' + str(count)
        args = [0]
        if show_time:
            text += ': %s'
            args += ['? remaining']
        Statement.__init__(self, text, args, node=node)

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
    #print('\033[H\033[J')
    os.system('cls' if os.name == 'nt' else 'clear')

def log(text, args=None, getter=None):
    global _ACTIVE_NODE
    statement = _ACTIVE_NODE.addStatement(text, args, getter)
        
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
    global ROOT
    print 'EXITING'
    os.system('pause')
    clear()
    ROOT.showAll()

def refresh():
    ancestry = [_ACTIVE_NODE]
    while ancestry[-1]._parent is not None:
        ancestry += [ancestry[-1]._parent]
    ancestry.reverse()

    clear()

    for i in range(len(ancestry)):
        node = ancestry[i]
        node.show(tab=i)


#setup
ROOT = LoggerNode()


