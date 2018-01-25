import pygmo as pg
"""
Okay, so the lesson learned here is: not sure :-D
"""
if __name__ == "__main__":
    prob = pg.problem(pg.rosenbrock(dim = 5))
    pool_creator = pg.mp_island()
    # pool_creator.resize_pool(1)
    pool_creator.init_pool(1)
    island = pg.island(udi=pool_creator, algo=pg.sga(gen = 200), pop=pg.population(prob,200))
    island.evolve()
    island.wait()
    print ("island: ***** \n", island)
