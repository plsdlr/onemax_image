# onemax problem / reproduces an image out of noise
# every generations best individual array gets saved as an image
# slightly visible image after 1600 generations
# example mostly take from the DAEP Documentation


import random

from deap import base
from deap import creator
from deap import tools
from numpy import interp
import cv2
import numpy as np

img = cv2.imread('othertest.jpg',0)


newarray=[]

for i in range(img.shape[0]):
    for b in range(img.shape[1]):
        newarray.append(img.item(i,b))


creator.create("FitnessMax", base.Fitness, weights=(1.0,))
creator.create("Individual", list, fitness=creator.FitnessMax)

toolbox = base.Toolbox()


toolbox.register("attr_bool", random.randint, 0, 255)

toolbox.register("individual", tools.initRepeat, creator.Individual,
    toolbox.attr_bool, 10000)


toolbox.register("population", tools.initRepeat, list, toolbox.individual)


def evalOneMax(individual):

    return sum(individual),


def evaluationthing2(individual):
    #resault = [40,90,0,80,17,200]
    global newarray
    #print len(newarray)
    score = 0
    for i in range(len(newarray)):
        score = score + interp(abs(newarray[i]-individual[i]),[0,255],[1,0])
    return score,



toolbox.register("evaluate", evaluationthing2)
toolbox.register("mate", tools.cxTwoPoint)
toolbox.register("mutate", tools.mutFlipBit, indpb=0.05)
toolbox.register("select", tools.selTournament, tournsize=3)

#----------

def main():
    random.seed(64)


    pop = toolbox.population(n=300)


    CXPB, MUTPB, NGEN = 0.5, 0.5, 600

    print("Start of evolution")


    fitnesses = list(map(toolbox.evaluate, pop))
    for ind, fit in zip(pop, fitnesses):
        ind.fitness.values = fit

    print("  Evaluated %i individuals" % len(pop))


    for g in range(NGEN):
        print("-- Generation %i --" % g)


        offspring = toolbox.select(pop, len(pop))

        offspring = list(map(toolbox.clone, offspring))


        for child1, child2 in zip(offspring[::2], offspring[1::2]):


            if random.random() < CXPB:
                toolbox.mate(child1, child2)
                del child1.fitness.values
                del child2.fitness.values

        for mutant in offspring:

            if random.random() < MUTPB:
                toolbox.mutate(mutant)
                del mutant.fitness.values

        invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
        fitnesses = map(toolbox.evaluate, invalid_ind)
        for ind, fit in zip(invalid_ind, fitnesses):
            ind.fitness.values = fit

        print("  Evaluated %i individuals" % len(invalid_ind))

        pop[:] = offspring

        fits = [ind.fitness.values[0] for ind in pop]

        length = len(pop)
        mean = sum(fits) / length
        sum2 = sum(x*x for x in fits)
        std = abs(sum2 / length - mean**2)**0.5

        print("  Min %s" % min(fits))
        print("  Max %s" % max(fits))
        print("  Avg %s" % mean)
        print("  Std %s" % std)

        bestone = tools.selBest(pop, 1)
        print "write in image"
        count = 0
        for i in range(img.shape[0]):
            for b in range(img.shape[1]):
                img.itemset((i,b),bestone[0][count])
                count += 1

        filename = "Generation" + str(g) +".png"
        cv2.imwrite(filename,img)
        print "save file"


    print("-- End of (successful) evolution --")

    best_ind = tools.selBest(pop, 1)[0]

if __name__ == "__main__":
    main()
