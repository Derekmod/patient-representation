import collapse_logging as logging
import os
import sys


def playGame():
    turn = [0]
    state = [0, 0, 0]
    logging.log('Player %d turn!', turn, lambda(turn): [turn[0] % 2])
    logging.log('')
    logging.log('state:\n\t%d\n\t%d\n\t%d', state)
    
    for _ in range(3):
        prompt = logging.log('Waiting for input.')
        cmd = int(raw_input('>>> '))
        state[cmd] += 1
        prompt.delete()
        confirmation = logging.log('got input: %d', cmd)
        confirmation.delete()


if __name__ == '__main__':
    logging.addNode('Game 1')
    game1_log = logging._ACTIVE_NODE
    playGame()
    logging.closeNode()
    logging.addNode('Game 2')
    game2_log = logging._ACTIVE_NODE
    playGame()
    logging.closeNode()
    breadth = [len(logging.ROOT._children)]
    logging.log('ROOT breadth %d', breadth)
    os.system('pause')
    game1_log.delete()
    game2_log.delete()
    breadth = [len(logging.ROOT._children)]
    progress = logging.addNode('testing progress', 5)
    #progress = logging.logProgress('sample progress log', 5, True)
    for _ in range(5):
        os.system('pause')
        progress.step()
    logging.closeNode()
    logging.log('sample log')
    logging.exit()