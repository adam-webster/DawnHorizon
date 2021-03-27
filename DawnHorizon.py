"""
Author: Adam W
This will replicate results from Nowak & May
1993 paper 'The Spatial Dilemmas of Evolution'
"""

import sys, argparse
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.colors import LinearSegmentedColormap
from progress.bar import IncrementalBar

#to format the help screen nicely
class CustomHelpFormatter(argparse.HelpFormatter):
    def _format_action_invocation(self, action):
        if not action.option_strings or action.nargs == 0:
            return super()._format_action_invocation(action)
        default = self._get_default_metavar_for_optional(action)
        args_string = self._format_args(action, default)
        return ', '.join(action.option_strings) + ' ' + args_string

    def __init__(self, prog):
        super().__init__(prog, max_help_position=40, width=80)

COOP = 1
DEFEC = 0
vals = [COOP, DEFEC]

def randomGrid(N):
    """returns a grid of NxN random values.
       Probability of cooperator is 0.9"""
    return np.random.choice(vals, N*N, p=[0.9, 0.1]).reshape(N, N)

def addGlider(i, j, grid):
    """adds a glider with top left cell at (i, j)"""
    glider = np.array([[1, 1, 1, 1, 1],
                       [1, 1, 1, 1, 0],
                       [1, 0, 0, 0, 0],
                       [1, 0, 0, 0, 0]])
    grid[i:i+4, j:j+5] = glider

def addRotator(i, j, grid):
    '''Adds a rotator (as defined in Nowak & May)
       with top left cell at (i, j)'''
    rotator = np.array([[1, 1],
                       [1, 1],
                       [1, 0],
                       [1, 0]])
    grid[i:i+4, j:j+2] = rotator

def addGrower(i, j, grid):
    '''Adds a grower (as defined in Nowak & May)
       with top left cell at (i, j)'''
    grower = np.array([[1, 1, 1, 1],
                       [0, 0, 1, 1],
                       [0, 0, 1, 1],
                       [0, 0, 1, 1]])
    grid[i:i+4, j:j+4] = grower

def kings_neighbours(N, i, j, grid, scoreGrid, scoring=False):
    '''This will generate a list of the 8 nearest neighbours,
       in kings move format, to location (i, j) on the grid.
       Boundary conditions are toroidal.
       Scoring parameter to be declared if you want to access the
       grid of scores'''
    if not scoring:
        neighbours = [grid[i, (j-1)%N], grid[i, (j+1)%N],
           grid[(i-1)%N, j], grid[(i+1)%N, j],
           grid[(i-1)%N, (j-1)%N], grid[(i-1)%N, (j+1)%N],
           grid[(i+1)%N, (j-1)%N], grid[(i+1)%N, (j+1)%N]]
    if scoring:
        neighbours = [scoreGrid[i, (j-1)%N], scoreGrid[i, (j+1)%N],
           scoreGrid[(i-1)%N, j], scoreGrid[(i+1)%N, j],
           scoreGrid[(i-1)%N, (j-1)%N], scoreGrid[(i-1)%N, (j+1)%N],
           scoreGrid[(i+1)%N, (j-1)%N], scoreGrid[(i+1)%N, (j+1)%N]]
    return neighbours


def fixed_bc_kings_neighbours(N, i, j, grid, scoreGrid, scoring=False):
    '''This will generate a list of nearest 8 neighbours,
       in kings move format, to location (i, j) on the grid.
       Boundary conditions are now fixed.
       Scoring parameter to be declared if you want to access the
       grid of scores'''
    if not scoring:
        if i >= 1 and i < N-1:
            if j >= 1 and j < N-1:
                neighbours = [grid[i, (j-1)%N], grid[i, (j+1)%N],
                    grid[(i-1)%N, j], grid[(i+1)%N, j],
                    grid[(i-1)%N, (j-1)%N], grid[(i-1)%N, (j+1)%N],
                    grid[(i+1)%N, (j-1)%N], grid[(i+1)%N, (j+1)%N]]
            elif j == 0:
                neighbours = [grid[i, (j+1)%N], grid[(i-1)%N, j],
                    grid[(i+1)%N, j], grid[(i-1)%N, (j+1)%N],
                    grid[(i+1)%N, (j+1)%N]]
            elif j == N-1:
                neighbours = [grid[i, (j-1)%N], grid[(i-1)%N, j],
                    grid[(i+1)%N, j], grid[(i-1)%N, (j-1)%N],
                    grid[(i+1)%N, (j-1)%N]]
        elif i == 0:
            if j >= 1 and j < N-1:
                neighbours = [grid[i, (j-1)%N], grid[i, (j+1)%N],
                   grid[(i+1)%N, j], grid[(i+1)%N, (j-1)%N],
                   grid[(i+1)%N, (j+1)%N]]
            elif j == 0:
                neighbours = [grid[i, (j+1)%N], grid[(i+1)%N, j],
                   grid[(i+1)%N, (j+1)%N]]
            elif j == N-1:
                neighbours = [grid[i, (j-1)%N], grid[(i+1)%N, j],
                   grid[(i+1)%N, (j-1)%N]]
        elif i == N-1:
            if j >= 1 and j < N-1:
                neighbours = [grid[i, (j-1)%N], grid[i, (j+1)%N],
                    grid[(i-1)%N, j], grid[(i-1)%N, (j-1)%N],
                    grid[(i-1)%N, (j+1)%N]]
            elif j == 0:
                neighbours = [grid[i, (j+1)%N], grid[(i-1)%N, j],
                    grid[(i-1)%N, (j+1)%N]]
            elif j == N-1:
                neighbours = [grid[i, (j-1)%N], grid[(i-1)%N, j],
                    grid[(i-1)%N, (j-1)%N]]

    if scoring:
        if i >= 1 and i < N-1:
            if j >= 1 and j < N-1:
                neighbours = [scoreGrid[i, (j-1)%N], scoreGrid[i, (j+1)%N],
                    scoreGrid[(i-1)%N, j], scoreGrid[(i+1)%N, j],
                    scoreGrid[(i-1)%N, (j-1)%N], scoreGrid[(i-1)%N, (j+1)%N],
                    scoreGrid[(i+1)%N, (j-1)%N], scoreGrid[(i+1)%N, (j+1)%N]]
            elif j == 0:
                neighbours = [scoreGrid[i, (j+1)%N], scoreGrid[(i-1)%N, j],
                    scoreGrid[(i+1)%N, j], scoreGrid[(i-1)%N, (j+1)%N],
                    scoreGrid[(i+1)%N, (j+1)%N]]
            elif j == N-1:
                neighbours = [scoreGrid[i, (j-1)%N], scoreGrid[(i-1)%N, j],
                    scoreGrid[(i+1)%N, j], scoreGrid[(i-1)%N, (j-1)%N],
                    scoreGrid[(i+1)%N, (j-1)%N]]
        elif i == 0:
            if j >= 1 and j < N-1:
                neighbours = [scoreGrid[i, (j-1)%N], scoreGrid[i, (j+1)%N],
                   scoreGrid[(i+1)%N, j], scoreGrid[(i+1)%N, (j-1)%N],
                   scoreGrid[(i+1)%N, (j+1)%N]]
            elif j == 0:
                neighbours = [scoreGrid[i, (j+1)%N], scoreGrid[(i+1)%N, j],
                   scoreGrid[(i+1)%N, (j+1)%N]]
            elif j == N-1:
                neighbours = [scoreGrid[i, (j-1)%N], scoreGrid[(i+1)%N, j],
                   scoreGrid[(i+1)%N, (j-1)%N]]
        elif i == N-1:
            if j >= 1 and j < N-1:
                neighbours = [scoreGrid[i, (j-1)%N], scoreGrid[i, (j+1)%N],
                    scoreGrid[(i-1)%N, j], scoreGrid[(i-1)%N, (j-1)%N],
                    scoreGrid[(i-1)%N, (j+1)%N]]
            elif j == 0:
                neighbours = [scoreGrid[i, (j+1)%N], scoreGrid[(i-1)%N, j],
                    scoreGrid[(i-1)%N, (j+1)%N]]
            elif j == N-1:
                neighbours = [scoreGrid[i, (j-1)%N], scoreGrid[(i-1)%N, j],
                    scoreGrid[(i-1)%N, (j-1)%N]]
    return neighbours

def freq_calc(grid, N, b, bound_cond, generations):
    '''This is designed to calculate the frequency of cooperators
       at each timestep, then plot how this changes over time'''

    time_list = np.arange(generations+1)
    coop_freq_list = []
    defec_freq_list = []
    # present a progress bar in terminal for confidence in long running simulations
    bar = IncrementalBar('Running', max=len(time_list),
          suffix='%(percent).1f%% complete - Time elapsed: %(elapsed)ds - Estimated time remaining: %(eta)ds')
    for timestep, time in enumerate(time_list):
        # copy grid since we require 8 neighbours for calculation
        newGrid = grid.copy()
        scoreGrid = np.zeros(N*N).reshape(N, N)

        # a cooperator in the grid is denoted by a 1, defector by a 0
        # so we count the amount of 1s in the grid at each step
        # then convert this to a frequency and add it to the list
        player, count = np.unique(grid, return_counts=True)

        # this set of condtionals handles cases when counts go to zero
        # when the counts goes to zero it is not included in the list
        # so this set will insert those values into the list
        if len(count) == 1:
            if player[0] == 0:
                count = np.insert(count, 1, 0)
            if player[0] == 1:
                count = np.insert(count, 0, 0)


        defec_freq = count[0] / (N*N) # index 0 is always defectors
        defec_freq_list.append(defec_freq)
        coop_freq = count[1] / (N*N) # index 1 is always cooperators
        coop_freq_list.append(coop_freq)


        # scoring loop
        # each player plays their neighbour in turn
        # and we go line by line
        for i in range(N):
            for j in range(N):
                # reset the current players score
                score = 0
                # initiate a list of all their neighbours
                if bound_cond == 'fixed':
                    neighbours = fixed_bc_kings_neighbours(
                                 N, i, j, grid, scoreGrid)
                else:
                    neighbours = kings_neighbours(N, i, j, grid, scoreGrid)
                # apply game rules
                # coop playing themself will always get payoff 1
                # no other self-interaction gives payoff
                # remove this line to remove self-interaction

                if grid[i, j] == 1:
                    score += 1

                # play against each neighbour
                for index, elem in enumerate(neighbours):
                    if elem == grid[i, j] and elem == COOP:
                        score += 1
                    elif elem == grid[i, j] and elem == DEFEC:
                        score += 0
                    elif grid[i,j] == COOP and elem == DEFEC:
                        score += 0
                    elif grid[i,j] == DEFEC and elem == COOP:
                        score += b
                    # after playing each neighbour,
                    # update the score on the grid
                    if index == len(neighbours)-1:
                        scoreGrid[i, j] = score

        # now each location has a score
        # we compare each score to its neighbours
        # and go line by line again
        for i in range(N):
            for j in range(N):
                # create a list of the neighbours scores to compare with
                if bound_cond == 'fixed':
                    scoreNeighbours = fixed_bc_kings_neighbours(
                                 N, i, j, grid, scoreGrid, scoring=True)
                else:
                    scoreNeighbours = kings_neighbours(
                                 N, i, j, grid, scoreGrid, scoring=True)
                # get location of highest score obtained by neighbours
                sort_scoreNeighbours = sorted(scoreNeighbours)
                hscore_index = scoreNeighbours.index(sort_scoreNeighbours[-1])

                # compare current players score with highest neighbours score
                # if current players score is higher, they stay as the player
                # in the next round
                if scoreGrid[i, j] >= sort_scoreNeighbours[-1]:
                    newGrid[i, j] = grid[i, j]

                # if a neighbour has a higher score, we find out which player
                # and that player takes over this cell in the next round
                elif scoreGrid[i, j] < sort_scoreNeighbours[-1]:
                    if bound_cond == 'fixed':
                        select = fixed_bc_kings_neighbours(
                                 N, i, j, grid, scoreGrid)
                    else:
                        select = kings_neighbours(N, i, j, grid, scoreGrid)
                    newGrid[i, j] = select[hscore_index]


        grid[:] = newGrid[:] # replace current grid with new grid
        bar.next() # update progress bar
    bar.finish()
    # now sort out the plotting, make it pretty
    plt.style.use('ggplot') # pretty format
    plt.figure(figsize=(9,5))
    plt.plot(time_list, coop_freq_list, 'b', linewidth=0.7, label='Cooperator Frequency')
    plt.plot(time_list, defec_freq_list, 'r', linewidth=0.7, label='Defector Frequency')
    plt.title('b = ' + str(b) + '   N = ' + str(N))
    plt.legend()
    plt.xlabel('Time')
    plt.ylabel('Frequency of player')
    plt.xlim(-1, len(time_list))
    plt.show()


def update(frameNum, img, grid, N, b, bound_cond):
    # copy grid since we require 8 neighbors for calculation
    newGrid = grid.copy()
    scoreGrid = np.zeros(N*N).reshape(N, N)

    # update data
    # in format for visualisation
    # do this first to avoid frame loss
    img.set_data(newGrid)

    # scoring loop
    # each player plays their neighbour in turn
    # and we go line by line
    for i in range(N):
        for j in range(N):
            # reset the current players score
            score = 0
            # initiate a list of all their neighbours
            if bound_cond == 'fixed':
                neighbours = fixed_bc_kings_neighbours(N, i, j, grid, scoreGrid)
            else:
                neighbours = kings_neighbours(N, i, j, grid, scoreGrid)
            # apply game rules
            # coop playing themself will always get payoff 1
            # no other self-interaction gives payoff
            if grid[i, j] == 1:
                score += 1
            # play against each neighbour
            for index, elem in enumerate(neighbours):
                if elem == grid[i, j] and elem == COOP:
                    score += 1
                elif elem == grid[i, j] and elem == DEFEC:
                    score += 0
                elif grid[i,j] == COOP and elem == DEFEC:
                    score += 0
                elif grid[i,j] == DEFEC and elem == COOP:
                    score += b
                # after playing each neighbour,
                # update the score on the grid
                if index == len(neighbours)-1:
                    scoreGrid[i, j] = score

    # now each location has a score
    # we compare each score to its neighbours
    # and go line by line again
    for i in range(N):
        for j in range(N):
            # create a list of the neighbours scores to compare with
            if bound_cond == 'fixed':
                scoreNeighbours = fixed_bc_kings_neighbours(
                                  N, i, j, grid, scoreGrid, scoring=True)
            else:
                scoreNeighbours = kings_neighbours(
                                  N, i, j, grid, scoreGrid, scoring=True)
            # get location of highest score obtained by neighbours
            sort_scoreNeighbours = sorted(scoreNeighbours)
            hscore_index = scoreNeighbours.index(sort_scoreNeighbours[-1])

            # compare current players score with highest neighbours score
            # if current players score is higher, they stay as the player
            # in the next round
            if scoreGrid[i, j] >= sort_scoreNeighbours[-1]:
                newGrid[i, j] = grid[i, j]

            # if a neighbour has a higher score, we find out which player
            # and that player takes over this cell in the next round
            elif scoreGrid[i, j] < sort_scoreNeighbours[-1]:
                if bound_cond == 'fixed':
                    select = fixed_bc_kings_neighbours(N, i, j, grid, scoreGrid)
                else:
                    select = kings_neighbours(N, i, j, grid, scoreGrid)
                newGrid[i, j] = select[hscore_index]


    grid[:] = newGrid[:]
    return img,


def main():
    # Command line args are in sys.argv[1], sys.argv[2] ..
    # sys.argv[0] is the script name itself and can be ignored
    # parse arguments
    parser = argparse.ArgumentParser(
    description='''Produces the results from Nowak & May
    1993 paper 'The Spatial Dilemmas of Evolution'. ''',
    formatter_class=CustomHelpFormatter)
    boundary_args = parser.add_mutually_exclusive_group(required=True)
    # add arguments
    parser.add_argument('--grid-size', '-N', dest='N', required=True,
    help='Size of grid to run the simulation on. This will generate a N x N square grid.')
    parser.add_argument('--movfile', dest='movfile', required=False,
                        help='''Saves the animation to filename/path you provide.
                        Saves as <arg>_<b arg>.gif or
                        as <arg>_<b arg>.png if used with -f''')
    parser.add_argument('--interval', '-i', dest='interval', required=False,
                        help='''Interval between animation updates in milliseconds.
                        200 is default if this switch is not specified.
                        If using with --frequency/-f this will specify the
                        amount of generations. Default otherwise is 200
                        generations''')
    parser.add_argument('--glider', '-g', dest='glider',
                        action='store_true', required=False,
                        help='Places a glider strucure in an otherwise empty grid')
    parser.add_argument('--rotator', '-r', dest='rotator',
                        action='store_true', required=False,
                        help='Places a rotator strucure in an otherwise empty grid')
    parser.add_argument('--grower', '-G', dest='grower',
                        action='store_true', required=False,
                        help='Places a grower strucure in an otherwise empty grid')
    parser.add_argument('-b', dest='b', required=True,
                        help='This is the advantage for defectors.')
    parser.add_argument('--frequency', '-f', dest='frequency',
                        action='store_true', required=False,
                        help='Produce frequency of cooperators graph (no visualisation)')
    parser.add_argument('--lone-defector', '-ld', dest='lone_d',
                        action='store_true', required=False,
                        help='Introduce a lone defector in the middle of the grid')
    boundary_args.add_argument('--fixed-boundaries', '-fbc',
                        dest='bound_cond', action='store_const', const='fixed',
                        help='Use fixed boundary condtions')
    boundary_args.add_argument('--periodic-boundaries', '-pbc',
                        dest='bound_cond', action='store_const', const='periodic',
                        help='Use periodic (toroidal) boundary condtions')
    args = parser.parse_args()

    #intitate inputs
    b = float(args.b)
    N = int(args.N)
    bound_cond = args.bound_cond

    # set animation update interval or frequency generations
    updateInterval = generations = 200
    if args.interval:
        updateInterval = generations = int(args.interval)

    # intitate grid and check for specific setups declared e.g. glider
    grid = np.array([])
    if args.glider:
        grid = np.zeros(N*N).reshape(N, N)
        addGlider(15, 5, grid)
    elif args.rotator:
        grid = np.zeros(N*N).reshape(N, N)
        addRotator(10, 10, grid)
    elif args.grower:
        grid = np.zeros(N*N).reshape(N, N)
        addGrower(8, 8, grid)
    elif args.lone_d:
        grid = np.ones(N*N).reshape(N, N)
        grid[int(N/2), int(N/2)] = 0
    else:
        grid = randomGrid(N)


    # this will make the color scheme of the visual plots
    # identical to that used in the paper
    # just add cmap=cm arg to imshow
    colors = [(1, 0, 0), (0, 0, 1)]
    cmap_name = 'nowak_scheme'
    cm = LinearSegmentedColormap.from_list(cmap_name, colors, N=2)

    if args.frequency:
        freq_calc(grid, N, b, bound_cond, generations)
        if args.movfile:
            plt.savefig(args.movfile+'_b='+str(b)+'.png')
    else:
        # set up animation
        fig, ax = plt.subplots()
        img = ax.imshow(grid, vmin=0.0, vmax=1.0, interpolation='nearest', cmap='binary')
        plt.colorbar(img, ax=ax)
        plt.title('Black denotes Cooperator')
        # display animation length in seconds is frames * interval /1000
        ani = animation.FuncAnimation(fig, update,
                                      fargs=(img, grid, N, b, bound_cond,),
                                      frames = 30,
                                      interval=updateInterval,
                                      save_count=50)

        # saved animation duration in seconds is frames * (1 / fps)
        # set output file
        writergif = animation.PillowWriter(fps=5)
        if args.movfile:
            ani.save(args.movfile+'.gif',writer=writergif)

        plt.show()


# call main
if __name__ == '__main__':
    main()
